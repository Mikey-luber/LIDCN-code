#!/usr/bin/env python3

import argparse
import copy
import math
import os
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sigmoid_pred_from_logit(z: torch.Tensor) -> torch.Tensor:
    return (z > 0).long()


def _parse_int_tuple(s: str) -> Tuple[int, ...]:
    s = str(s)
    xs = []
    for x in s.split(','):
        x = str(x).strip()
        if x == '':
            continue
        xs.append(int(x))
    return tuple(xs)


def _parse_attack_from_npz_name(npz_path: Path) -> str:
    name = npz_path.stem
    if name.startswith('lid_resnet_'):
        # lid_resnet_{attack}_...
        rest = name[len('lid_resnet_'):]
        parts = rest.split('_')
        if len(parts) >= 2:
            # Some pipelines embed the feature mode into the filename:
            #   lid_resnet_{featmode}_{attack}_...
            # e.g. lid_resnet_flatten_pgd_...
            feat_mode_tokens = {
                'flatten',
                'avgpool',
            }
            if parts[0] in feat_mode_tokens:
                return parts[1]
        return parts[0]
    if name.startswith('lid_clip_'):
        # lid_clip_{modeltoken}_{attack}_...
        rest = name[len('lid_clip_'):]
        parts = rest.split('_')
        if len(parts) >= 2:
            return parts[1]
    return name


def _safe_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return bool(x.item())
        if x.size == 1:
            return bool(x.reshape(-1)[0])
    return bool(x)


def _np_logits_to_single_z(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim == 0:
        return logits.reshape(1).astype(np.float32, copy=False)
    if logits.ndim == 1:
        return logits.astype(np.float32, copy=False).reshape(-1)
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits.astype(np.float32, copy=False).reshape(-1)
    if logits.ndim == 2 and logits.shape[1] == 2:
        z = logits[:, 1] - logits[:, 0]
        return np.asarray(z, dtype=np.float32).reshape(-1)
    raise ValueError(f"Unsupported logits shape for single-logit z: {tuple(logits.shape)}")


@dataclass
class NpzAttackData:
    attack: str
    lid_clean: np.ndarray
    lid_adv: np.ndarray
    z_clean: np.ndarray
    z_adv: np.ndarray
    y_clean_eff: np.ndarray
    y_adv_eff: np.ndarray
    attack_success_mask: np.ndarray
    attack_valid_mask: Optional[np.ndarray]


def _subset_attack_data(it: NpzAttackData, idx: np.ndarray) -> NpzAttackData:
    idx = np.asarray(idx)
    return NpzAttackData(
        attack=it.attack,
        lid_clean=it.lid_clean[idx],
        lid_adv=it.lid_adv[idx],
        z_clean=it.z_clean[idx],
        z_adv=it.z_adv[idx],
        y_clean_eff=it.y_clean_eff[idx],
        y_adv_eff=it.y_adv_eff[idx],
        attack_success_mask=it.attack_success_mask[idx],
        attack_valid_mask=(None if it.attack_valid_mask is None else it.attack_valid_mask[idx]),
    )


def _split_attack_indices(attack: str, n: int, train_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n = int(n)
    if n <= 1:
        raise ValueError(f"Cannot split attack={attack} with n={n}")
    train_frac = float(train_frac)
    if not (0.0 < train_frac <= 1.0):
        raise ValueError(f"train_frac must be in (0, 1], got {train_frac}")

    if train_frac >= 1.0:
        idx = np.arange(n, dtype=np.int64)
        return idx, idx

    # Deterministic per-attack split.
    atk_hash = int(zlib.crc32(str(attack).encode('utf-8')) & 0xFFFFFFFF)
    rng = np.random.RandomState(int(seed) ^ atk_hash)
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    n_train = max(1, min(n - 1, n_train))
    idx_train = np.sort(perm[:n_train]).astype(np.int64)
    idx_test = np.sort(perm[n_train:]).astype(np.int64)
    return idx_train, idx_test


def _resolve_input_mode(input_mode: str, use_logits_input_flag: bool) -> str:
    input_mode = str(input_mode).strip().lower()
    if input_mode == '' or input_mode == 'auto':
        return 'lid_logit' if bool(use_logits_input_flag) else 'lid_only'
    if input_mode not in {'lid_logit', 'lid_only', 'logit_only'}:
        raise ValueError(f"Unknown input_mode: {input_mode}")
    return input_mode


def _apply_input_mode_to_attack_data(it: NpzAttackData, input_mode: str) -> NpzAttackData:
    input_mode = str(input_mode).strip().lower()
    if input_mode != 'logit_only':
        return it
    lid_clean = np.asarray(it.lid_clean)
    lid_adv = np.asarray(it.lid_adv)
    lid_clean = lid_clean[:, :0]
    lid_adv = lid_adv[:, :0]
    return NpzAttackData(
        attack=it.attack,
        lid_clean=lid_clean,
        lid_adv=lid_adv,
        z_clean=it.z_clean,
        z_adv=it.z_adv,
        y_clean_eff=it.y_clean_eff,
        y_adv_eff=it.y_adv_eff,
        attack_success_mask=it.attack_success_mask,
        attack_valid_mask=it.attack_valid_mask,
    )


def _resolve_struct_mode(struct_mode: str, use_gate_flag: bool, use_flip_flag: bool) -> str:
    struct_mode = str(struct_mode).strip().lower()
    if struct_mode == '' or struct_mode == 'auto':
        if bool(use_gate_flag) and bool(use_flip_flag):
            return 'full'
        if bool(use_gate_flag):
            return 'gate'
        return 'base'
    if struct_mode not in {'base', 'gate', 'flip', 'full'}:
        raise ValueError(f"Unknown struct_mode: {struct_mode}")
    return struct_mode


def _load_npz_for_calibration(npz_path: Path, require_pairing: bool = True) -> NpzAttackData:
    data = np.load(npz_path, allow_pickle=True)

    features = np.asarray(data['features'], dtype=np.float32)
    n_neg = int(data['n_neg_samples'])

    clean_effective_labels = np.asarray(data['clean_effective_labels'], dtype=np.int64).reshape(-1)
    adv_true_labels = np.asarray(data['adv_true_labels'], dtype=np.int64).reshape(-1)

    z_clean = _np_logits_to_single_z(np.asarray(data['clean_model_logits']))
    z_adv = _np_logits_to_single_z(np.asarray(data['adv_model_logits']))

    flip_attack_labels = _safe_bool(data.get('flip_attack_labels', False))
    if flip_attack_labels:
        clean_effective_labels = 1 - clean_effective_labels
    y_adv_eff = adv_true_labels

    attack_success_mask = np.asarray(data.get('attack_success_mask', np.zeros_like(y_adv_eff, dtype=bool)), dtype=bool).reshape(-1)
    attack_valid_mask = None
    if 'attack_valid_mask' in data:
        attack_valid_mask = np.asarray(data['attack_valid_mask'], dtype=bool).reshape(-1)

    n_clean = int(len(clean_effective_labels))
    n_adv = int(len(y_adv_eff))

    if len(z_clean) != n_clean:
        raise ValueError(
            f"clean_model_logits length mismatch: expected n_clean={n_clean} (from clean_effective_labels) "
            f"but got len(z_clean)={len(z_clean)} in {npz_path}"
        )
    if len(z_adv) != n_adv:
        raise ValueError(
            f"adv_model_logits length mismatch: expected n_adv={n_adv} (from adv_true_labels) "
            f"but got len(z_adv)={len(z_adv)} in {npz_path}"
        )

    if require_pairing and (n_adv != n_clean):
        raise ValueError(f"Expected paired clean/adv logits. Got n_clean={n_clean} n_adv={n_adv} in {npz_path}")

    lid_clean = features[:n_clean]
    lid_adv = features[n_neg:n_neg + n_adv]

    if require_pairing and (len(lid_adv) != n_clean):
        raise ValueError(f"Expected paired clean/adv LID. Got len(lid_clean)={len(lid_clean)} len(lid_adv)={len(lid_adv)} in {npz_path}")

    attack = _parse_attack_from_npz_name(npz_path)

    return NpzAttackData(
        attack=attack,
        lid_clean=lid_clean,
        lid_adv=lid_adv,
        z_clean=z_clean,
        z_adv=z_adv,
        y_clean_eff=clean_effective_labels,
        y_adv_eff=y_adv_eff,
        attack_success_mask=attack_success_mask,
        attack_valid_mask=attack_valid_mask,
    )


class _CalibDataset(Dataset):
    def __init__(
        self,
        lid: np.ndarray,
        z: np.ndarray,
        y: np.ndarray,
        teacher_z: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray],
        gate_y: Optional[np.ndarray],
        gate_weight: Optional[np.ndarray],
        succ: Optional[np.ndarray] = None,
        valid: Optional[np.ndarray] = None,
    ):
        self.lid = np.asarray(lid, dtype=np.float32)
        self.z = np.asarray(z, dtype=np.float32).reshape(-1, 1)
        self.y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.teacher_z = None if teacher_z is None else np.asarray(teacher_z, dtype=np.float32).reshape(-1, 1)
        self.sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32).reshape(-1, 1)
        if gate_y is None:
            self.gate_y = None
        else:
            self.gate_y = np.asarray(gate_y, dtype=np.float32).reshape(-1, 1)
        if gate_weight is None:
            self.gate_weight = None
        else:
            self.gate_weight = np.asarray(gate_weight, dtype=np.float32).reshape(-1, 1)

        if succ is None:
            self.succ = None
        else:
            self.succ = np.asarray(succ, dtype=np.float32).reshape(-1, 1)

        if valid is None:
            self.valid = None
        else:
            self.valid = np.asarray(valid, dtype=np.float32).reshape(-1, 1)

        n = len(self.lid)
        if len(self.z) != n:
            raise ValueError("lid and z length mismatch")
        if self.y is not None and len(self.y) != n:
            raise ValueError("y length mismatch")
        if self.teacher_z is not None and len(self.teacher_z) != n:
            raise ValueError("teacher_z length mismatch")
        if self.sample_weight is not None and len(self.sample_weight) != n:
            raise ValueError("sample_weight length mismatch")
        if self.gate_y is not None and len(self.gate_y) != n:
            raise ValueError("gate_y length mismatch")
        if self.gate_weight is not None and len(self.gate_weight) != n:
            raise ValueError("gate_weight length mismatch")
        if self.succ is not None and len(self.succ) != n:
            raise ValueError("succ length mismatch")
        if self.valid is not None and len(self.valid) != n:
            raise ValueError("valid length mismatch")

    def __len__(self):
        return len(self.lid)

    def __getitem__(self, idx: int):
        out = {
            'lid': torch.from_numpy(self.lid[idx]),
            'z': torch.from_numpy(self.z[idx]),
            'y': torch.from_numpy(self.y[idx]),
        }
        if self.teacher_z is not None:
            out['teacher_z'] = torch.from_numpy(self.teacher_z[idx])
        if self.sample_weight is not None:
            out['w'] = torch.from_numpy(self.sample_weight[idx])
        if self.gate_y is not None:
            out['gate_y'] = torch.from_numpy(self.gate_y[idx])
        if self.gate_weight is not None:
            out['gate_w'] = torch.from_numpy(self.gate_weight[idx])
        if self.succ is not None:
            out['succ'] = torch.from_numpy(self.succ[idx])
        if self.valid is not None:
            out['valid'] = torch.from_numpy(self.valid[idx])
        return out


def _build_model_from_config(config: Dict[str, object]) -> nn.Module:
    method = str(config['method'])
    use_gate = bool(config.get('use_gate', False))
    use_flip_expert = bool(config.get('use_flip_expert', False))
    use_logits_input = bool(config.get('use_logits_input', False))
    gate_use_logits_input = bool(config.get('gate_use_logits_input', False))

    lid_dim = int(config['lid_dim'])
    input_dim = int(lid_dim) + (1 if use_logits_input else 0)

    hidden = config.get('hidden', ())
    if isinstance(hidden, str):
        hidden = _parse_int_tuple(hidden)
    hidden = tuple(int(x) for x in list(hidden))

    base = LogitCalibrator(
        input_dim=input_dim,
        method=method,
        hidden=hidden,
        a_range=float(config.get('a_range', 0.5)),
        b_range=float(config.get('b_range', 2.0)),
        delta_range=float(config.get('delta_range', 2.0)),
    )

    model: nn.Module = base
    if use_flip_expert and (not use_gate):
        flip = FlipAffineExpert(
            init_scale=float(config.get('flip_init_scale', 1.0)),
            init_bias=float(config.get('flip_init_bias', 0.0)),
        )
        model = FlipOnlyCalibrator(
            calibrator=base,
            flip_expert=flip,
            alpha_init=float(config.get('flip_only_alpha_init', 0.5)),
        )
        return model

    gate_hidden = config.get('gate_hidden', (64, 64))
    if isinstance(gate_hidden, str):
        gate_hidden = _parse_int_tuple(gate_hidden)
    gate_hidden = tuple(int(x) for x in list(gate_hidden))

    if use_gate:
        gate_input_dim = int(lid_dim) + (1 if gate_use_logits_input else 0)
        gate = GateNet(input_dim=gate_input_dim, hidden=gate_hidden)
        if use_flip_expert:
            flip = FlipAffineExpert(
                init_scale=float(config.get('flip_init_scale', 1.0)),
                init_bias=float(config.get('flip_init_bias', 0.0)),
            )
            model = FlipMoECalibrator(
                calibrator=base,
                gate=gate,
                flip_expert=flip,
                epsilon_gate=float(config.get('epsilon_gate', 0.1)),
                gate_use_logits_input=gate_use_logits_input,
            )
        else:
            model = GatedCalibrator(
                calibrator=base,
                gate=gate,
                epsilon_gate=float(config.get('epsilon_gate', 0.1)),
                gate_use_logits_input=gate_use_logits_input,
            )
    return model


def _save_checkpoint(path: Path, model: nn.Module, config: Dict[str, object]) -> None:
    payload = {
        'state_dict': model.state_dict(),
        'config': dict(config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def _load_checkpoint(path: Path, map_location: str = 'cpu') -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    obj = torch.load(str(path), map_location=map_location)
    if isinstance(obj, dict) and ('state_dict' in obj):
        cfg = obj.get('config', {})
        sd = obj['state_dict']
        return cfg, sd
    if isinstance(obj, dict):
        return {}, obj
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _infer_checkpoint_lid_dim(sd: Dict[str, torch.Tensor], use_logits_input: bool) -> Optional[int]:
    candidates = []
    for key in ('calibrator.net.0.weight', 'net.0.weight'):
        w = sd.get(key)
        if torch.is_tensor(w) and w.dim() == 2:
            candidates.append(int(w.shape[1]) - (1 if bool(use_logits_input) else 0))
    gate_w = sd.get('gate.net.0.weight')
    if torch.is_tensor(gate_w) and gate_w.dim() == 2:
        candidates.append(int(gate_w.shape[1]) - 1)
    candidates = [int(x) for x in candidates if int(x) >= 0]
    if len(candidates) == 0:
        return None
    return int(max(candidates))


def _config_for_loaded_checkpoint(
    cfg_loaded: Dict[str, object],
    sd: Dict[str, torch.Tensor],
    fallback_cfg: Dict[str, object],
    data_lid_dim: int,
    load_path: Path,
) -> Dict[str, object]:
    cfg = dict(fallback_cfg)
    cfg.update(dict(cfg_loaded or {}))

    cfg.setdefault('use_logits_input', bool(fallback_cfg.get('use_logits_input', False)))
    cfg.setdefault('gate_use_logits_input', bool(fallback_cfg.get('gate_use_logits_input', False)))

    if 'gate.net.0.weight' in sd and torch.is_tensor(sd['gate.net.0.weight']):
        gate_in = int(sd['gate.net.0.weight'].shape[1])
        if gate_in == int(data_lid_dim) + 1:
            cfg['gate_use_logits_input'] = True
        elif gate_in == int(data_lid_dim):
            cfg['gate_use_logits_input'] = False
        else:
            raise ValueError(
                f"Checkpoint gate input dim mismatch for {load_path}: gate.net.0.weight expects {gate_in}, "
                f"but current data lid_dim={int(data_lid_dim)} gives valid gate dims "
                f"{int(data_lid_dim)} or {int(data_lid_dim) + 1}."
            )

    inferred_lid_dim = _infer_checkpoint_lid_dim(sd, use_logits_input=bool(cfg.get('use_logits_input', False)))
    ckpt_lid_dim = int(cfg.get('lid_dim', inferred_lid_dim if inferred_lid_dim is not None else data_lid_dim))
    if ckpt_lid_dim != int(data_lid_dim):
        raise ValueError(
            f"Checkpoint lid_dim mismatch for {load_path}: checkpoint expects lid_dim={ckpt_lid_dim}, "
            f"but current feature data has lid_dim={int(data_lid_dim)}."
        )
    cfg['lid_dim'] = int(data_lid_dim)
    return cfg


def average_checkpoints(ckpt_paths: List[Path], output_path: Path, map_location: str = 'cpu') -> None:
    """Average multiple checkpoint state_dicts and save to output_path.
    
    This helps stabilize the calibrator by averaging weights from multiple
    training rounds, reducing oscillation in adversarial training.
    """
    if len(ckpt_paths) == 0:
        raise ValueError("No checkpoints to average")
    
    if len(ckpt_paths) == 1:
        # Just copy the single checkpoint
        cfg, sd = _load_checkpoint(ckpt_paths[0], map_location=map_location)
        _save_checkpoint(output_path, _build_model(cfg), cfg)
        model = _build_model(cfg)
        model.load_state_dict(sd)
        _save_checkpoint(output_path, model, cfg)
        return
    
    # Load all checkpoints
    all_cfgs = []
    all_sds = []
    for p in ckpt_paths:
        cfg, sd = _load_checkpoint(p, map_location=map_location)
        all_cfgs.append(cfg)
        all_sds.append(sd)
    
    # Use config from the last checkpoint
    final_cfg = all_cfgs[-1]
    
    # Average state dicts
    avg_sd = {}
    keys = all_sds[0].keys()
    for k in keys:
        tensors = [sd[k].float() for sd in all_sds]
        avg_sd[k] = sum(tensors) / len(tensors)
        # Restore original dtype
        avg_sd[k] = avg_sd[k].to(dtype=all_sds[0][k].dtype)
    
    # Build model and load averaged weights
    model = _build_model(final_cfg)
    model.load_state_dict(avg_sd)
    _save_checkpoint(output_path, model, final_cfg)
    print(f"[checkpoint_avg] Averaged {len(ckpt_paths)} checkpoints -> {output_path}")


class LogitCalibrator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        method: str,
        hidden: Tuple[int, ...],
        a_range: float,
        b_range: float,
        delta_range: float,
    ):
        super().__init__()
        method = str(method)
        if method not in {'affine', 'residual'}:
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.a_range = float(a_range)
        self.b_range = float(b_range)
        self.delta_range = float(delta_range)

        layers: List[nn.Module] = []
        prev = int(input_dim)
        for h in hidden:
            h = int(h)
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        out_dim = 2 if self.method == 'affine' else 1
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, lid: torch.Tensor, z: torch.Tensor, use_logits_input: bool) -> Dict[str, torch.Tensor]:
        if use_logits_input:
            x = torch.cat([lid, z], dim=1)
        else:
            x = lid

        raw = self.net(x)

        if self.method == 'affine':
            raw_a = raw[:, :1]
            raw_b = raw[:, 1:]
            a = 1.0 + self.a_range * torch.tanh(raw_a)
            b = self.b_range * torch.tanh(raw_b)
            z_corr = a * z + b
            return {'z_corr': z_corr, 'a': a, 'b': b}

        delta = self.delta_range * torch.tanh(raw)
        z_corr = z + delta
        return {'z_corr': z_corr, 'delta': delta}


class GateNet(nn.Module):
    def __init__(self, input_dim: int, hidden: Tuple[int, ...]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = int(input_dim)
        for h in hidden:
            h = int(h)
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.sigmoid(logits)


class GatedCalibrator(nn.Module):
    def __init__(
        self,
        calibrator: LogitCalibrator,
        gate: GateNet,
        epsilon_gate: float,
        gate_use_logits_input: bool,
    ):
        super().__init__()
        self.calibrator = calibrator
        self.gate = gate
        self.epsilon_gate = float(epsilon_gate)
        self.gate_use_logits_input = bool(gate_use_logits_input)

    def forward(self, lid: torch.Tensor, z: torch.Tensor, use_logits_input: bool) -> Dict[str, torch.Tensor]:
        base_out = self.calibrator(lid, z, use_logits_input=use_logits_input)
        z_cal = base_out['z_corr']
        gate_x = torch.cat([lid, z], dim=1) if self.gate_use_logits_input else lid
        w = self.gate(gate_x)
        alpha = self.epsilon_gate + (1.0 - self.epsilon_gate) * w
        z_corr = z + alpha * (z_cal - z)
        out = dict(base_out)
        out['z_corr'] = z_corr
        out['w'] = w
        out['alpha'] = alpha
        return out


class FlipOnlyCalibrator(nn.Module):
    def __init__(
        self,
        calibrator: LogitCalibrator,
        flip_expert: 'FlipAffineExpert',
        alpha_init: float,
    ):
        super().__init__()
        self.calibrator = calibrator
        self.flip_expert = flip_expert
        alpha_init = float(alpha_init)
        alpha_init = min(0.999, max(0.001, alpha_init))
        logit = math.log(alpha_init / (1.0 - alpha_init))
        self.alpha_logit = nn.Parameter(torch.tensor([logit], dtype=torch.float32))

    def forward(self, lid: torch.Tensor, z: torch.Tensor, use_logits_input: bool) -> Dict[str, torch.Tensor]:
        base_out = self.calibrator(lid, z, use_logits_input=use_logits_input)
        z_base = base_out['z_corr']
        z_flip = self.flip_expert(z)
        alpha = torch.sigmoid(self.alpha_logit).view(1, 1)
        z_corr = (1.0 - alpha) * z_base + alpha * z_flip
        out = dict(base_out)
        out['z_corr'] = z_corr
        out['alpha'] = alpha.expand_as(z_corr)
        return out


class FlipAffineExpert(nn.Module):
    def __init__(self, init_scale: float, init_bias: float):
        super().__init__()
        init_scale = float(init_scale)
        init_bias = float(init_bias)
        if init_scale <= 0:
            raise ValueError(f"init_scale must be > 0, got {init_scale}")
        self.log_scale = nn.Parameter(torch.log(torch.tensor([init_scale], dtype=torch.float32)))
        self.bias = nn.Parameter(torch.tensor([init_bias], dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        a = -torch.exp(self.log_scale)
        return a * z + self.bias


class FlipMoECalibrator(nn.Module):
    def __init__(
        self,
        calibrator: LogitCalibrator,
        gate: GateNet,
        flip_expert: FlipAffineExpert,
        epsilon_gate: float,
        gate_use_logits_input: bool,
    ):
        super().__init__()
        self.calibrator = calibrator
        self.gate = gate
        self.flip_expert = flip_expert
        self.epsilon_gate = float(epsilon_gate)
        self.gate_use_logits_input = bool(gate_use_logits_input)

    def forward(self, lid: torch.Tensor, z: torch.Tensor, use_logits_input: bool) -> Dict[str, torch.Tensor]:
        base_out = self.calibrator(lid, z, use_logits_input=use_logits_input)
        z_base = base_out['z_corr']

        gate_x = torch.cat([lid, z], dim=1) if self.gate_use_logits_input else lid
        w = self.gate(gate_x)
        alpha = self.epsilon_gate + (1.0 - self.epsilon_gate) * w

        z_flip = self.flip_expert(z)
        z_corr = (1.0 - alpha) * z_base + alpha * z_flip

        out = dict(base_out)
        out['z_corr'] = z_corr
        out['w'] = w
        out['alpha'] = alpha
        return out


@torch.no_grad()
def _eval_split(
    model: nn.Module,
    device: torch.device,
    lid_clean: np.ndarray,
    z_clean: np.ndarray,
    y_clean: np.ndarray,
    lid_adv: np.ndarray,
    z_adv: np.ndarray,
    y_adv: np.ndarray,
    attack_success_mask: np.ndarray,
    use_logits_input: bool,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()

    z_clean_t = torch.from_numpy(np.asarray(z_clean, dtype=np.float32).reshape(-1, 1)).to(device)
    y_clean_t = torch.from_numpy(np.asarray(y_clean, dtype=np.int64).reshape(-1, 1)).to(device)
    z_adv_t = torch.from_numpy(np.asarray(z_adv, dtype=np.float32).reshape(-1, 1)).to(device)
    y_adv_t = torch.from_numpy(np.asarray(y_adv, dtype=np.int64).reshape(-1, 1)).to(device)

    clean_pred_before = _sigmoid_pred_from_logit(z_clean_t)
    adv_pred_before = _sigmoid_pred_from_logit(z_adv_t)

    clean_acc_before = float((clean_pred_before == y_clean_t).float().mean().item())
    adv_acc_before = float((adv_pred_before == y_adv_t).float().mean().item())

    def _batched_corr(lid_np: np.ndarray, z_t: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        outs_z = []
        outs_w = []
        n = len(z_t)
        for i in range(0, n, batch_size):
            lid_b = torch.from_numpy(np.asarray(lid_np[i:i + batch_size], dtype=np.float32)).to(device)
            z_b = z_t[i:i + batch_size]
            out = model(lid_b, z_b, use_logits_input=use_logits_input)
            outs_z.append(out['z_corr'].detach().cpu())
            if 'w' in out:
                outs_w.append(out['w'].detach().cpu())
        z_corr = torch.cat(outs_z, dim=0)
        w = torch.cat(outs_w, dim=0) if len(outs_w) > 0 else None
        return z_corr, w

    z_clean_corr, w_clean = _batched_corr(lid_clean, z_clean_t)
    z_adv_corr, w_adv = _batched_corr(lid_adv, z_adv_t)

    clean_pred_after = _sigmoid_pred_from_logit(z_clean_corr)
    adv_pred_after = _sigmoid_pred_from_logit(z_adv_corr)

    clean_acc_after = float((clean_pred_after.to(device) == y_clean_t).float().mean().item())
    adv_acc_after = float((adv_pred_after.to(device) == y_adv_t).float().mean().item())

    attack_success_mask = np.asarray(attack_success_mask, dtype=bool).reshape(-1)
    if attack_success_mask.shape[0] != len(z_adv):
        attack_success_mask = np.zeros(len(z_adv), dtype=bool)

    if int(attack_success_mask.sum()) > 0:
        succ_idx = np.where(attack_success_mask)[0]
        succ_before = adv_pred_before[succ_idx]
        succ_after = adv_pred_after[succ_idx]
        succ_y = y_adv_t[succ_idx]
        succ_recover_rate = float((succ_after.to(device) == succ_y).float().mean().item())
        succ_before_acc = float((succ_before == succ_y).float().mean().item())
    else:
        succ_recover_rate = float('nan')
        succ_before_acc = float('nan')

    if w_clean is not None:
        mean_w_clean = float(w_clean.mean().item())
    else:
        mean_w_clean = float('nan')
    if w_adv is not None:
        mean_w_adv = float(w_adv.mean().item())
    else:
        mean_w_adv = float('nan')
    if w_adv is not None and int(attack_success_mask.sum()) > 0:
        mean_w_succ = float(w_adv[succ_idx].mean().item())
    else:
        mean_w_succ = float('nan')

    return {
        'clean_acc_before': clean_acc_before,
        'clean_acc_after': clean_acc_after,
        'adv_acc_before': adv_acc_before,
        'adv_acc_after': adv_acc_after,
        'succ_acc_before': succ_before_acc,
        'succ_acc_after': succ_recover_rate,
        'n_adv': float(len(z_adv)),
        'n_succ': float(int(attack_success_mask.sum())),
        'mean_w_clean': mean_w_clean,
        'mean_w_adv': mean_w_adv,
        'mean_w_succ': mean_w_succ,
    }


def _ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(b.data)


def _train_one(
    model: nn.Module,
    device: torch.device,
    train_adv: _CalibDataset,
    train_clean: _CalibDataset,
    epochs: int,
    lr: float,
    batch_size: int,
    lambda_id: float,
    lambda_kd: float,
    lambda_delta: float,
    lambda_fail_id: float,
    lambda_gate: float,
    gate_pos_weight: float,
    use_logits_input: bool,
    log_every_epochs: int,
    log_every_steps: int,
    ema_decay: float = 0.0,
    lr_schedule: str = 'constant',
    grad_clip: float = 0.0,
    weight_decay: float = 0.0,
    lr_warmup_ratio: float = 0.0,
    label_smoothing: float = 0.0,
) -> Optional[nn.Module]:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    adv_loader = DataLoader(train_adv, batch_size=batch_size, shuffle=True, drop_last=False)
    clean_loader = DataLoader(train_clean, batch_size=batch_size, shuffle=True, drop_last=False)

    total_steps = int(epochs) * len(adv_loader)
    warmup_steps = int(float(lr_warmup_ratio) * total_steps) if float(lr_warmup_ratio) > 0 else 0
    scheduler = None
    if str(lr_schedule) == 'cosine' and total_steps > 1:
        if warmup_steps > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps), eta_min=float(lr) * 0.01)
            scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=float(lr) * 0.01)

    ema_model: Optional[nn.Module] = None
    if float(ema_decay) > 0:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        print(f"[train] EMA enabled: decay={float(ema_decay):.4f}")
    if scheduler is not None:
        print(f"[train] LR schedule: {lr_schedule}, total_steps={total_steps}, warmup_steps={warmup_steps}")
    if float(grad_clip) > 0:
        print(f"[train] Gradient clipping: max_norm={float(grad_clip)}")
    if float(label_smoothing) > 0:
        print(f"[train] Label smoothing: {float(label_smoothing):.3f}")

    # Intra-round best epoch tracking
    best_epoch_metric: float = float('-inf')
    best_epoch_state: Optional[dict] = None
    best_epoch_idx: int = -1

    global_step = 0
    for epoch_idx in range(int(epochs)):
        model.train()
        it_clean = iter(clean_loader)

        epoch_loss_total = 0.0
        epoch_loss_adv = 0.0
        epoch_loss_id = 0.0
        epoch_loss_kd = 0.0
        epoch_loss_delta = 0.0
        epoch_loss_fail_id = 0.0
        epoch_loss_gate = 0.0
        epoch_adv_correct = 0
        epoch_adv_total = 0
        epoch_clean_correct = 0
        epoch_clean_total = 0

        epoch_w_adv_sum = 0.0
        epoch_w_adv_n = 0
        epoch_w_clean_sum = 0.0
        epoch_w_clean_n = 0
        epoch_alpha_adv_sum = 0.0
        epoch_alpha_adv_n = 0
        epoch_alpha_clean_sum = 0.0
        epoch_alpha_clean_n = 0

        for batch_adv in adv_loader:
            try:
                batch_clean = next(it_clean)
            except StopIteration:
                it_clean = iter(clean_loader)
                batch_clean = next(it_clean)

            lid_adv = batch_adv['lid'].to(device)
            z_adv = batch_adv['z'].to(device)
            y_adv = batch_adv['y'].to(device)
            teacher_z = batch_adv.get('teacher_z', None)
            w_adv = batch_adv.get('w', None)
            if teacher_z is not None:
                teacher_z = teacher_z.to(device)
            if w_adv is not None:
                w_adv = w_adv.to(device)

            lid_clean = batch_clean['lid'].to(device)
            z_clean = batch_clean['z'].to(device)
            y_clean = batch_clean.get('y', None)
            if y_clean is not None:
                y_clean = y_clean.to(device)

            y_gate_adv = batch_adv.get('gate_y', None)
            y_gate_clean = batch_clean.get('gate_y', None)
            if y_gate_adv is not None:
                y_gate_adv = y_gate_adv.to(device)
            if y_gate_clean is not None:
                y_gate_clean = y_gate_clean.to(device)

            gate_w_adv = batch_adv.get('gate_w', None)
            gate_w_clean = batch_clean.get('gate_w', None)
            if gate_w_adv is not None:
                gate_w_adv = gate_w_adv.to(device)
            if gate_w_clean is not None:
                gate_w_clean = gate_w_clean.to(device)

            out_adv = model(lid_adv, z_adv, use_logits_input=use_logits_input)
            z_corr_adv = out_adv['z_corr']

            y_adv_target = y_adv
            if float(label_smoothing) > 0:
                ls = float(label_smoothing)
                y_adv_target = y_adv * (1.0 - ls) + (1.0 - y_adv) * ls
            bce = F.binary_cross_entropy_with_logits(z_corr_adv, y_adv_target, reduction='none')
            if w_adv is not None:
                bce = bce * w_adv
            loss_adv = bce.mean()

            loss_kd = torch.tensor(0.0, device=device)
            if lambda_kd > 0 and teacher_z is not None:
                kd = F.mse_loss(z_corr_adv, teacher_z, reduction='none')
                if w_adv is not None:
                    kd = kd * w_adv
                loss_kd = kd.mean()

            loss_fail_id = torch.tensor(0.0, device=device)
            if float(lambda_fail_id) > 0:
                succ_adv = batch_adv.get('succ', None)
                valid_adv = batch_adv.get('valid', None)
                if succ_adv is not None:
                    succ_mask = (succ_adv.to(device) > 0.5).reshape(-1, 1)
                    fail_mask = (~succ_mask)
                    if valid_adv is not None:
                        fail_mask = fail_mask & (valid_adv.to(device) > 0.5).reshape(-1, 1)
                    if int(fail_mask.sum().item()) > 0:
                        diff = z_corr_adv - z_adv
                        loss_fail_id = (diff[fail_mask] ** 2).mean()

            out_clean = model(lid_clean, z_clean, use_logits_input=use_logits_input)
            z_corr_clean = out_clean['z_corr']
            loss_id = F.mse_loss(z_corr_clean, z_clean)

            loss_delta = torch.tensor(0.0, device=device)
            if lambda_delta > 0:
                loss_delta = F.mse_loss(z_corr_adv, z_adv)

            loss_gate = torch.tensor(0.0, device=device)
            if (
                lambda_gate > 0
                and ('w' in out_adv)
                and ('w' in out_clean)
                and (y_gate_adv is not None)
                and (y_gate_clean is not None)
            ):
                loss_gate_adv = F.binary_cross_entropy(out_adv['w'], y_gate_adv, reduction='none')
                loss_gate_clean = F.binary_cross_entropy(out_clean['w'], y_gate_clean, reduction='none')
                if float(gate_pos_weight) != 1.0:
                    loss_gate_adv = loss_gate_adv * (1.0 + (float(gate_pos_weight) - 1.0) * y_gate_adv)
                    loss_gate_clean = loss_gate_clean * (1.0 + (float(gate_pos_weight) - 1.0) * y_gate_clean)
                if gate_w_adv is not None:
                    loss_gate_adv = loss_gate_adv * gate_w_adv
                if gate_w_clean is not None:
                    loss_gate_clean = loss_gate_clean * gate_w_clean
                loss_gate = 0.5 * (loss_gate_adv.mean() + loss_gate_clean.mean())

            loss = (
                loss_adv
                + float(lambda_id) * loss_id
                + float(lambda_kd) * loss_kd
                + float(lambda_delta) * loss_delta
                + float(lambda_fail_id) * loss_fail_id
                + float(lambda_gate) * loss_gate
            )

            with torch.no_grad():
                pred_adv = (z_corr_adv > 0).long()
                y_adv_long = (y_adv > 0.5).long()
                epoch_adv_correct += int((pred_adv == y_adv_long).sum().item())
                epoch_adv_total += int(y_adv_long.numel())

                if y_clean is not None:
                    pred_clean = (z_corr_clean > 0).long()
                    y_clean_long = (y_clean > 0.5).long()
                    epoch_clean_correct += int((pred_clean == y_clean_long).sum().item())
                    epoch_clean_total += int(y_clean_long.numel())

                if 'w' in out_adv:
                    epoch_w_adv_sum += float(out_adv['w'].sum().item())
                    epoch_w_adv_n += int(out_adv['w'].numel())
                if 'w' in out_clean:
                    epoch_w_clean_sum += float(out_clean['w'].sum().item())
                    epoch_w_clean_n += int(out_clean['w'].numel())
                if 'alpha' in out_adv:
                    epoch_alpha_adv_sum += float(out_adv['alpha'].sum().item())
                    epoch_alpha_adv_n += int(out_adv['alpha'].numel())
                if 'alpha' in out_clean:
                    epoch_alpha_clean_sum += float(out_clean['alpha'].sum().item())
                    epoch_alpha_clean_n += int(out_clean['alpha'].numel())

                epoch_loss_total += float(loss.detach().item())
                epoch_loss_adv += float(loss_adv.detach().item())
                epoch_loss_id += float(loss_id.detach().item())
                epoch_loss_kd += float(loss_kd.detach().item())
                epoch_loss_delta += float(loss_delta.detach().item())
                epoch_loss_fail_id += float(loss_fail_id.detach().item())
                epoch_loss_gate += float(loss_gate.detach().item())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            opt.step()
            if scheduler is not None:
                scheduler.step()
            if ema_model is not None:
                _ema_update(ema_model, model, float(ema_decay))

            global_step += 1
            if int(log_every_steps) > 0 and (global_step % int(log_every_steps) == 0):
                adv_acc = (epoch_adv_correct / epoch_adv_total) if epoch_adv_total > 0 else float('nan')
                clean_acc = (epoch_clean_correct / epoch_clean_total) if epoch_clean_total > 0 else float('nan')
                w_adv = (epoch_w_adv_sum / epoch_w_adv_n) if epoch_w_adv_n > 0 else float('nan')
                alpha_adv = (epoch_alpha_adv_sum / epoch_alpha_adv_n) if epoch_alpha_adv_n > 0 else float('nan')
                print(
                    f"[train][step {global_step}] loss={epoch_loss_total / max(1, global_step):.4f} "
                    f"adv_acc={adv_acc:.3f} clean_acc={clean_acc:.3f} "
                    f"w_adv={w_adv:.3f} alpha_adv={alpha_adv:.3f}"
                )

        if int(log_every_epochs) > 0 and ((epoch_idx + 1) % int(log_every_epochs) == 0):
            n_batches = len(adv_loader)
            adv_acc = (epoch_adv_correct / epoch_adv_total) if epoch_adv_total > 0 else float('nan')
            clean_acc = (epoch_clean_correct / epoch_clean_total) if epoch_clean_total > 0 else float('nan')
            w_adv = (epoch_w_adv_sum / epoch_w_adv_n) if epoch_w_adv_n > 0 else float('nan')
            w_clean = (epoch_w_clean_sum / epoch_w_clean_n) if epoch_w_clean_n > 0 else float('nan')
            alpha_adv = (epoch_alpha_adv_sum / epoch_alpha_adv_n) if epoch_alpha_adv_n > 0 else float('nan')
            alpha_clean = (epoch_alpha_clean_sum / epoch_alpha_clean_n) if epoch_alpha_clean_n > 0 else float('nan')
            print(
                f"[train][epoch {epoch_idx + 1}/{int(epochs)}] "
                f"loss={epoch_loss_total / max(1, n_batches):.4f} "
                f"(adv={epoch_loss_adv / max(1, n_batches):.4f} "
                f"id={epoch_loss_id / max(1, n_batches):.4f} "
                f"kd={epoch_loss_kd / max(1, n_batches):.4f} "
                f"delta={epoch_loss_delta / max(1, n_batches):.4f} "
                f"fail_id={epoch_loss_fail_id / max(1, n_batches):.4f} "
                f"gate={epoch_loss_gate / max(1, n_batches):.4f}) "
                f"adv_acc={adv_acc:.3f} clean_acc={clean_acc:.3f} "
                f"w_adv={w_adv:.3f} w_clean={w_clean:.3f} "
                f"alpha_adv={alpha_adv:.3f} alpha_clean={alpha_clean:.3f}"
            )

        # Intra-round best epoch: track based on combined adv + clean accuracy
        epoch_adv_acc = (epoch_adv_correct / epoch_adv_total) if epoch_adv_total > 0 else 0.0
        epoch_clean_acc = (epoch_clean_correct / epoch_clean_total) if epoch_clean_total > 0 else 0.0
        epoch_metric = 0.7 * epoch_adv_acc + 0.3 * epoch_clean_acc
        if epoch_metric > best_epoch_metric:
            best_epoch_metric = epoch_metric
            best_epoch_idx = epoch_idx
            best_epoch_state = copy.deepcopy(model.state_dict())

    # Restore best epoch weights if we tracked them and training ran > 1 epoch
    if best_epoch_state is not None and int(epochs) > 1 and best_epoch_idx < int(epochs) - 1:
        model.load_state_dict(best_epoch_state)
        print(f"[train] Restored best epoch {best_epoch_idx + 1}/{int(epochs)} (metric={best_epoch_metric:.4f})")
        # Update EMA model to match restored weights
        if ema_model is not None:
            ema_model = copy.deepcopy(model)
            ema_model.eval()

    return ema_model


def _collect_npzs(feature_dir: Path) -> List[Path]:
    feature_dir = Path(os.path.expanduser(str(feature_dir)))
    if not feature_dir.exists() or not feature_dir.is_dir():
        cwd = Path.cwd()
        candidates = []
        try:
            for p in cwd.iterdir():
                if p.is_dir() and len(list(p.glob('*.npz'))) > 0:
                    candidates.append(p.name)
        except Exception:
            candidates = []
        msg = (
            f"feature_dir does not exist or is not a directory: {feature_dir}\n"
            f"cwd: {cwd}\n"
            "Tip: make sure to pass './your_dir' (Linux/macOS) or '.\\\\your_dir' (Windows), not '.your_dir'.\n"
        )
        if len(candidates) > 0:
            msg += f"Candidate directories under cwd containing .npz: {candidates}\n"
        raise FileNotFoundError(msg)

    npzs = sorted([p for p in feature_dir.glob('*.npz') if p.is_file()])
    if len(npzs) == 0:
        cwd = Path.cwd()
        msg = (
            f"No .npz found under {feature_dir}\n"
            f"cwd: {cwd}\n"
            "Tip: ensure you're pointing to the directory that directly contains '*.npz' files." 
        )
        raise FileNotFoundError(msg)
    return npzs

def _stack_train_data(
    items: List[NpzAttackData],
    success_weight: float,
    gate_target: str,
    gate_weight_valid_mask: bool,
) -> Tuple[_CalibDataset, _CalibDataset]:
    lid_clean = np.concatenate([it.lid_clean for it in items], axis=0)
    z_clean = np.concatenate([it.z_clean for it in items], axis=0)
    y_clean = np.concatenate([it.y_clean_eff for it in items], axis=0)

    lid_adv = np.concatenate([it.lid_adv for it in items], axis=0)
    z_adv = np.concatenate([it.z_adv for it in items], axis=0)
    y_adv = np.concatenate([it.y_adv_eff for it in items], axis=0)

    teacher_z = np.concatenate([it.z_clean for it in items], axis=0)
    succ = np.concatenate([it.attack_success_mask for it in items], axis=0).astype(bool)
    valid = np.concatenate([
        np.asarray(it.attack_valid_mask, dtype=bool) if it.attack_valid_mask is not None else np.ones_like(it.attack_success_mask, dtype=bool)
        for it in items
    ], axis=0).astype(bool)
    w = np.ones_like(y_adv, dtype=np.float32)
    if float(success_weight) != 1.0:
        w[succ] *= float(success_weight)

    gate_target = str(gate_target)
    if gate_target not in {'is_adv', 'is_success'}:
        raise ValueError(f"Unknown gate_target: {gate_target}")

    if gate_target == 'is_adv':
        gate_y_adv = np.ones_like(y_adv, dtype=np.float32)
    else:
        gate_y_adv = succ.astype(np.float32)

    if bool(gate_weight_valid_mask):
        gate_weight_adv = valid.astype(np.float32)
    else:
        gate_weight_adv = np.ones_like(y_adv, dtype=np.float32)

    gate_y_clean = np.zeros_like(y_clean, dtype=np.float32)
    gate_weight_clean = np.ones_like(y_clean, dtype=np.float32)

    ds_adv = _CalibDataset(
        lid=lid_adv,
        z=z_adv,
        y=y_adv,
        teacher_z=teacher_z,
        sample_weight=w,
        gate_y=gate_y_adv,
        gate_weight=gate_weight_adv,
        succ=succ.astype(np.float32),
        valid=valid.astype(np.float32),
    )
    ds_clean = _CalibDataset(
        lid=lid_clean,
        z=z_clean,
        y=y_clean,
        teacher_z=None,
        sample_weight=None,
        gate_y=gate_y_clean,
        gate_weight=gate_weight_clean,
        succ=None,
        valid=None,
    )
    return ds_adv, ds_clean


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--held_out_attack', type=str, default='')
    parser.add_argument('--loao', action='store_true')
    parser.add_argument('--eval_all_attacks', action='store_true')
    parser.add_argument('--train_frac', type=float, default=1.0)
    parser.add_argument('--split_seed', type=int, default=42)

    parser.add_argument('--method', type=str, default='affine', choices=['affine', 'residual'])
    parser.add_argument('--use_logits_input', action='store_true')
    parser.add_argument('--input_mode', type=str, default='auto', choices=['auto', 'lid_logit', 'lid_only', 'logit_only'])

    parser.add_argument('--use_gate', action='store_true')
    parser.add_argument('--struct_mode', type=str, default='auto', choices=['auto', 'base', 'gate', 'flip', 'full'])
    parser.add_argument('--epsilon_gate', type=float, default=0.2)
    parser.add_argument('--lambda_gate', type=float, default=0.5)
    parser.add_argument('--gate_hidden', type=str, default='64,64')
    parser.add_argument('--gate_use_logits_input', action='store_true')
    parser.add_argument('--gate_target', type=str, default='is_adv', choices=['is_adv', 'is_success'])
    parser.add_argument('--gate_weight_valid_mask', action='store_true')
    parser.add_argument('--gate_pos_weight', type=float, default=1.0)

    parser.add_argument('--use_flip_expert', action='store_true')
    parser.add_argument('--flip_init_scale', type=float, default=1.0)
    parser.add_argument('--flip_init_bias', type=float, default=0.0)

    parser.add_argument('--hidden', type=str, default='128,128')

    parser.add_argument('--a_range', type=float, default=0.5)
    parser.add_argument('--b_range', type=float, default=2.0)
    parser.add_argument('--delta_range', type=float, default=2.0)

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--log_every_epochs', type=int, default=1)
    parser.add_argument('--log_every_steps', type=int, default=0)

    parser.add_argument('--lambda_id', type=float, default=2.0)
    parser.add_argument('--lambda_kd', type=float, default=1.0)
    parser.add_argument('--lambda_delta', type=float, default=0.0)
    parser.add_argument('--lambda_fail_id', type=float, default=0.0)

    parser.add_argument('--success_weight', type=float, default=3.0)

    parser.add_argument('--ema_decay', type=float, default=0.0)
    parser.add_argument('--lr_schedule', type=str, default='constant', choices=['constant', 'cosine'])
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_warmup_ratio', type=float, default=0.0,
                         help='Fraction of total training steps for LR warmup (e.g. 0.1 = 10%%). Only used with cosine schedule.')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                         help='Label smoothing for adversarial BCE loss (e.g. 0.05). Reduces overconfidence.')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--output_dir', type=str, default='./logit_calibrators')

    parser.add_argument('--save_model_path', type=str, default='')
    parser.add_argument('--load_model_path', type=str, default='')
    parser.add_argument('--skip_train', action='store_true')

    parser.add_argument('--infer_npz', type=str, default='')
    parser.add_argument('--infer_split', type=str, default='adv', choices=['clean', 'adv'])
    parser.add_argument('--infer_from', type=str, default='test', choices=['train', 'test', 'all'])
    parser.add_argument('--infer_index', type=int, default=0)

    args = parser.parse_args()

    _seed_everything(args.seed)

    resolved_input_mode = _resolve_input_mode(getattr(args, 'input_mode', 'auto'), bool(args.use_logits_input))
    if resolved_input_mode == 'lid_only':
        args.use_logits_input = False
    else:
        args.use_logits_input = True

    resolved_struct_mode = _resolve_struct_mode(getattr(args, 'struct_mode', 'auto'), bool(args.use_gate), bool(args.use_flip_expert))
    if str(getattr(args, 'struct_mode', 'auto')).strip().lower() not in {'', 'auto'}:
        if resolved_struct_mode == 'base':
            args.use_gate = False
            args.use_flip_expert = False
            args.lambda_gate = 0.0
        elif resolved_struct_mode == 'gate':
            args.use_gate = True
            args.use_flip_expert = False
        elif resolved_struct_mode == 'flip':
            args.use_gate = False
            args.use_flip_expert = True
            args.lambda_gate = 0.0
        else:  # full
            args.use_gate = True
            args.use_flip_expert = True

        print(
            f"[info] struct_mode={resolved_struct_mode} overrides flags: "
            f"use_gate={bool(args.use_gate)} use_flip_expert={bool(args.use_flip_expert)} lambda_gate={float(args.lambda_gate)}"
        )

    if bool(getattr(args, 'use_gate', False)) and resolved_input_mode == 'logit_only' and (not bool(getattr(args, 'gate_use_logits_input', False))):
        args.gate_use_logits_input = True
        print("[info] input_mode=logit_only overrides gate_use_logits_input=True")

    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_npz_path(p: str) -> Path:
        pp = Path(os.path.expanduser(str(p)))
        if pp.exists():
            return pp
        cand = feature_dir / pp
        if cand.exists():
            return cand
        raise FileNotFoundError(f"npz not found: {p} (also tried {cand})")

    def _resolve_save_path(template: str, attack: str, mode: str) -> Path:
        template = str(template).strip()
        if template == '':
            raise ValueError('empty save_model_path')
        out = template.format(attack=str(attack), mode=str(mode))
        p = Path(os.path.expanduser(out))
        if str(p).lower().endswith(('.pt', '.pth')):
            return p
        name = f"calibrator_{mode}_{attack}.pt"
        return p / name

    def _resolve_load_path(template: str, attack: str, mode: str) -> Path:
        template = str(template).strip()
        if template == '':
            raise ValueError('empty load_model_path')
        p = Path(os.path.expanduser(template.format(attack=str(attack), mode=str(mode))))
        return p

    if str(args.infer_npz).strip() != '':
        if str(args.load_model_path).strip() == '':
            raise ValueError("--infer_npz requires --load_model_path")

        npz_path = _resolve_npz_path(args.infer_npz)
        item = _load_npz_for_calibration(npz_path, require_pairing=True)
        n = int(len(item.z_clean))
        idx_tr, idx_te = _split_attack_indices(item.attack, n=n, train_frac=float(args.train_frac), seed=int(args.split_seed))
        if str(args.infer_from) == 'train':
            pool = idx_tr
        elif str(args.infer_from) == 'test':
            pool = idx_te
        else:
            pool = np.arange(n, dtype=np.int64)

        if len(pool) == 0:
            raise ValueError(f"Empty infer pool for attack={item.attack} infer_from={args.infer_from}")

        infer_i = int(args.infer_index)
        if infer_i < 0:
            infer_i = len(pool) + infer_i
        if not (0 <= infer_i < len(pool)):
            raise IndexError(f"infer_index out of range: {infer_i} (pool size={len(pool)})")
        src_idx = int(pool[infer_i])

        cfg, sd = _load_checkpoint(_resolve_load_path(args.load_model_path, attack=item.attack, mode='infer'), map_location='cpu')
        cfg = dict(cfg)
        cfg.setdefault('method', args.method)
        cfg.setdefault('input_mode', resolved_input_mode)
        cfg.setdefault('use_logits_input', bool(args.use_logits_input))
        cfg.setdefault('use_gate', bool(args.use_gate))
        cfg.setdefault('epsilon_gate', float(args.epsilon_gate))
        cfg.setdefault('gate_use_logits_input', bool(args.gate_use_logits_input))
        cfg.setdefault('use_flip_expert', bool(args.use_flip_expert))
        cfg.setdefault('flip_init_scale', float(args.flip_init_scale))
        cfg.setdefault('flip_init_bias', float(args.flip_init_bias))
        cfg.setdefault('hidden', str(args.hidden))
        cfg.setdefault('gate_hidden', str(args.gate_hidden))
        cfg.setdefault('a_range', float(args.a_range))
        cfg.setdefault('b_range', float(args.b_range))
        cfg.setdefault('delta_range', float(args.delta_range))
        cfg['lid_dim'] = int(item.lid_adv.shape[1])

        infer_mode = str(cfg.get('input_mode', resolved_input_mode)).strip().lower()
        if infer_mode in {'lid_only', 'lid_logit', 'logit_only'}:
            cfg['input_mode'] = infer_mode
            if infer_mode == 'lid_only':
                cfg['use_logits_input'] = False
            else:
                cfg['use_logits_input'] = True

        if str(cfg.get('input_mode', '')).strip().lower() == 'logit_only':
            cfg['lid_dim'] = 0

        model = _build_model_from_config(cfg)
        model.load_state_dict(sd, strict=True)

        device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
        model = model.to(device)
        model.eval()

        if str(args.infer_split) == 'clean':
            lid_np = item.lid_clean[src_idx]
            z_np = float(item.z_clean[src_idx])
            y_np = int(item.y_clean_eff[src_idx])
        else:
            lid_np = item.lid_adv[src_idx]
            z_np = float(item.z_adv[src_idx])
            y_np = int(item.y_adv_eff[src_idx])

        if str(cfg.get('input_mode', '')).strip().lower() == 'logit_only':
            lid_np = np.asarray(lid_np, dtype=np.float32).reshape(-1)[:0]

        lid_t = torch.from_numpy(np.asarray(lid_np, dtype=np.float32).reshape(1, -1)).to(device)
        z_t = torch.tensor([[z_np]], dtype=torch.float32, device=device)
        out = model(lid_t, z_t, use_logits_input=bool(cfg.get('use_logits_input', False)))
        z_corr = float(out['z_corr'].detach().cpu().item())
        prob = float(torch.sigmoid(out['z_corr']).detach().cpu().item())
        pred = int((out['z_corr'] > 0).long().detach().cpu().item())

        w = float(out['w'].detach().cpu().item()) if ('w' in out) else float('nan')
        alpha = float(out['alpha'].detach().cpu().item()) if ('alpha' in out) else float('nan')
        print(
            f"[infer] npz={npz_path.name} attack={item.attack} split={args.infer_split} infer_from={args.infer_from} "
            f"pool_idx={infer_i} src_idx={src_idx} y={y_np}\n"
            f"[infer] z_in={z_np:.6f} z_corr={z_corr:.6f} prob={prob:.6f} pred={pred} w={w:.6f} alpha={alpha:.6f}"
        )
        return

    print(f"[info] feature_dir: {feature_dir}")
    print(f"[info] output_dir:  {output_dir}")
    npzs = _collect_npzs(feature_dir)
    print(f"[info] found npz files: {len(npzs)}")
    all_items_raw = [_load_npz_for_calibration(p, require_pairing=True) for p in npzs]
    all_items = [_apply_input_mode_to_attack_data(it, resolved_input_mode) for it in all_items_raw]

    attacks = sorted({it.attack for it in all_items})
    if len(attacks) == 0:
        raise RuntimeError('No attacks found')
    print(f"[info] attacks: {attacks}")

    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')

    hidden = _parse_int_tuple(args.hidden)

    items_by_attack: Dict[str, NpzAttackData] = {it.attack: it for it in all_items}
    split_by_attack: Dict[str, Tuple[NpzAttackData, NpzAttackData]] = {}
    for atk in attacks:
        it = items_by_attack[atk]
        n = int(len(it.z_clean))
        idx_tr, idx_te = _split_attack_indices(atk, n=n, train_frac=float(args.train_frac), seed=int(args.split_seed))
        split_by_attack[atk] = (_subset_attack_data(it, idx_tr), _subset_attack_data(it, idx_te))

    if args.eval_all_attacks and args.loao:
        raise ValueError("--eval_all_attacks and --loao are mutually exclusive")

    def _make_config(lid_dim: int) -> Dict[str, object]:
        return {
            'method': str(args.method),
            'use_logits_input': bool(args.use_logits_input),
            'input_mode': str(resolved_input_mode),
            'struct_mode': str(resolved_struct_mode),
            'hidden': tuple(int(x) for x in hidden),
            'a_range': float(args.a_range),
            'b_range': float(args.b_range),
            'delta_range': float(args.delta_range),
            'use_gate': bool(args.use_gate),
            'epsilon_gate': float(args.epsilon_gate),
            'gate_hidden': _parse_int_tuple(args.gate_hidden),
            'gate_use_logits_input': bool(args.gate_use_logits_input),
            'use_flip_expert': bool(args.use_flip_expert),
            'flip_init_scale': float(args.flip_init_scale),
            'flip_init_bias': float(args.flip_init_bias),
            'lid_dim': int(lid_dim),
        }

    def _build_model(input_dim: int, lid_dim: int) -> nn.Module:
        cfg = _make_config(lid_dim=lid_dim)
        return _build_model_from_config(cfg)

    def _load_model_for_lid_dim(load_path: Path, lid_dim: int) -> nn.Module:
        cfg_loaded, sd = _load_checkpoint(load_path, map_location='cpu')
        cfg = _config_for_loaded_checkpoint(
            cfg_loaded=cfg_loaded,
            sd=sd,
            fallback_cfg=_make_config(lid_dim=int(lid_dim)),
            data_lid_dim=int(lid_dim),
            load_path=load_path,
        )
        args.use_logits_input = bool(cfg.get('use_logits_input', args.use_logits_input))
        args.use_gate = bool(cfg.get('use_gate', args.use_gate))
        args.gate_use_logits_input = bool(cfg.get('gate_use_logits_input', args.gate_use_logits_input))
        args.use_flip_expert = bool(cfg.get('use_flip_expert', args.use_flip_expert))
        model_loaded = _build_model_from_config(cfg)
        model_loaded.load_state_dict(sd, strict=True)
        print(
            f"[load] {load_path} | use_logits_input={bool(args.use_logits_input)} "
            f"gate_use_logits_input={bool(args.gate_use_logits_input)}"
        )
        return model_loaded


    def _run_one(held_out: str) -> Dict[str, float]:
        held_out = str(held_out)
        print(
            f"\n[loao] held_out_attack={held_out} | method={args.method} | "
            f"input_mode={resolved_input_mode} | struct_mode={resolved_struct_mode} | use_logits_input={bool(args.use_logits_input)} | train_frac={float(args.train_frac)}"
        )
        train_items = [split_by_attack[a][0] for a in attacks if a != held_out]
        test = split_by_attack[held_out][1]

        train_adv, train_clean = _stack_train_data(
            train_items,
            success_weight=args.success_weight,
            gate_target=args.gate_target,
            gate_weight_valid_mask=bool(args.gate_weight_valid_mask),
        )

        input_dim = int(test.lid_adv.shape[1]) + (1 if args.use_logits_input else 0)
        model: nn.Module = _build_model(input_dim=input_dim, lid_dim=int(test.lid_adv.shape[1]))

        model = model.to(device)

        if str(args.load_model_path).strip() != '':
            if args.loao and ('{attack}' not in str(args.load_model_path)):
                raise ValueError("--loao with --load_model_path requires a template containing '{attack}'")
            load_path = _resolve_load_path(args.load_model_path, attack=held_out, mode='loao')
            model = _load_model_for_lid_dim(load_path, lid_dim=int(test.lid_adv.shape[1])).to(device)

        if not bool(args.skip_train) and int(args.epochs) > 0:
            ema_model = _train_one(
                model=model,
                device=device,
                train_adv=train_adv,
                train_clean=train_clean,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                lambda_id=args.lambda_id,
                lambda_kd=args.lambda_kd,
                lambda_delta=args.lambda_delta,
                lambda_fail_id=args.lambda_fail_id,
                lambda_gate=args.lambda_gate,
                gate_pos_weight=args.gate_pos_weight,
                use_logits_input=args.use_logits_input,
                log_every_epochs=args.log_every_epochs,
                log_every_steps=args.log_every_steps,
                ema_decay=args.ema_decay,
                lr_schedule=args.lr_schedule,
                grad_clip=args.grad_clip,
                weight_decay=args.weight_decay,
                lr_warmup_ratio=args.lr_warmup_ratio,
                label_smoothing=args.label_smoothing,
            )
            if ema_model is not None:
                model = ema_model
                print('[train] Using EMA model for save/eval')

        if str(args.save_model_path).strip() != '':
            if args.loao and ('{attack}' not in str(args.save_model_path)):
                save_path = _resolve_save_path(args.save_model_path, attack=held_out, mode='loao')
            else:
                save_path = _resolve_save_path(args.save_model_path, attack=held_out, mode='loao')
            _save_checkpoint(save_path, model=model, config=_make_config(lid_dim=int(test.lid_adv.shape[1])))
            print(f"[save] model: {save_path}")

        metrics = _eval_split(
            model=model,
            device=device,
            lid_clean=test.lid_clean,
            z_clean=test.z_clean,
            y_clean=test.y_clean_eff,
            lid_adv=test.lid_adv,
            z_adv=test.z_adv,
            y_adv=test.y_adv_eff,
            attack_success_mask=test.attack_success_mask,
            use_logits_input=args.use_logits_input,
            batch_size=args.batch_size,
        )

        model_out = {
            'held_out_attack': held_out,
            'eval_mode': 'loao',
            'train_frac': float(args.train_frac),
            'split_seed': float(args.split_seed),
            'method': args.method,
            'input_mode': str(resolved_input_mode),
            'struct_mode': str(resolved_struct_mode),
            'use_logits_input': float(bool(args.use_logits_input)),
            'use_gate': float(bool(args.use_gate)),
            'epsilon_gate': float(args.epsilon_gate) if args.use_gate else float('nan'),
            'lambda_gate': float(args.lambda_gate) if args.use_gate else float('nan'),
            'gate_use_logits_input': float(bool(args.gate_use_logits_input)) if args.use_gate else float('nan'),
            'gate_target': str(args.gate_target) if args.use_gate else '',
            'gate_weight_valid_mask': float(bool(args.gate_weight_valid_mask)) if args.use_gate else float('nan'),
            'gate_pos_weight': float(args.gate_pos_weight) if args.use_gate else float('nan'),
            'use_flip_expert': float(bool(args.use_flip_expert)),
            'flip_init_scale': float(args.flip_init_scale) if args.use_flip_expert else float('nan'),
            'flip_init_bias': float(args.flip_init_bias) if args.use_flip_expert else float('nan'),
            **metrics,
        }

        out_path = output_dir / (
            f"calib_{feature_dir.name}_{held_out}_{args.method}_in{resolved_input_mode}_struct{resolved_struct_mode}_logits{int(bool(args.use_logits_input))}"
            f"_gate{int(bool(args.use_gate))}_flip{int(bool(args.use_flip_expert))}"
            f"_tr{float(args.train_frac):.3f}_seed{int(args.split_seed)}.txt"
        )
        with open(out_path, 'w', encoding='utf-8') as f:
            for k, v in model_out.items():
                f.write(f"{k}: {v}\n")

        print(
            "[done] "
            f"clean {metrics['clean_acc_before']:.4f}->{metrics['clean_acc_after']:.4f} | "
            f"adv {metrics['adv_acc_before']:.4f}->{metrics['adv_acc_after']:.4f} | "
            f"succ {metrics['succ_acc_before']:.4f}->{metrics['succ_acc_after']:.4f} "
            f"(n_succ={int(metrics['n_succ'])}) | "
            f"mean_w_adv {metrics.get('mean_w_adv', float('nan')):.3f}\n"
            f"[save] {out_path}"
        )

        return model_out


    def _run_seen_all() -> List[Dict[str, float]]:
        print(
            f"\n[seen] method={args.method} | input_mode={resolved_input_mode} | struct_mode={resolved_struct_mode} | use_logits_input={bool(args.use_logits_input)} | "
            f"train_frac={float(args.train_frac)}"
        )
        train_items = [split_by_attack[a][0] for a in attacks]
        train_adv, train_clean = _stack_train_data(
            train_items,
            success_weight=args.success_weight,
            gate_target=args.gate_target,
            gate_weight_valid_mask=bool(args.gate_weight_valid_mask),
        )

        any_atk = attacks[0]
        lid_dim = int(split_by_attack[any_atk][0].lid_adv.shape[1])
        input_dim = lid_dim + (1 if args.use_logits_input else 0)
        model: nn.Module = _build_model(input_dim=input_dim, lid_dim=lid_dim)
        model = model.to(device)

        if str(args.load_model_path).strip() != '':
            if '{attack}' in str(args.load_model_path):
                raise ValueError("--eval_all_attacks with --load_model_path should not use '{attack}' template")
            load_path = _resolve_load_path(args.load_model_path, attack='seen', mode='seen')
            model = _load_model_for_lid_dim(load_path, lid_dim=lid_dim).to(device)

        if not bool(args.skip_train) and int(args.epochs) > 0:
            ema_model = _train_one(
                model=model,
                device=device,
                train_adv=train_adv,
                train_clean=train_clean,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                lambda_id=args.lambda_id,
                lambda_kd=args.lambda_kd,
                lambda_delta=args.lambda_delta,
                lambda_fail_id=args.lambda_fail_id,
                lambda_gate=args.lambda_gate,
                gate_pos_weight=args.gate_pos_weight,
                use_logits_input=args.use_logits_input,
                log_every_epochs=args.log_every_epochs,
                log_every_steps=args.log_every_steps,
                ema_decay=args.ema_decay,
                lr_schedule=args.lr_schedule,
                grad_clip=args.grad_clip,
                weight_decay=args.weight_decay,
                lr_warmup_ratio=args.lr_warmup_ratio,
                label_smoothing=args.label_smoothing,
            )
            if ema_model is not None:
                model = ema_model
                print('[train] Using EMA model for save/eval')

        if str(args.save_model_path).strip() != '':
            if '{attack}' in str(args.save_model_path):
                raise ValueError("--eval_all_attacks with --save_model_path should not use '{attack}' template")
            save_path = _resolve_save_path(args.save_model_path, attack='seen', mode='seen')
            _save_checkpoint(save_path, model=model, config=_make_config(lid_dim=int(lid_dim)))
            print(f"[save] model: {save_path}")

        rows: List[Dict[str, float]] = []
        for atk in attacks:
            test = split_by_attack[atk][1]
            metrics = _eval_split(
                model=model,
                device=device,
                lid_clean=test.lid_clean,
                z_clean=test.z_clean,
                y_clean=test.y_clean_eff,
                lid_adv=test.lid_adv,
                z_adv=test.z_adv,
                y_adv=test.y_adv_eff,
                attack_success_mask=test.attack_success_mask,
                use_logits_input=args.use_logits_input,
                batch_size=args.batch_size,
            )
            row = {
                'held_out_attack': atk,
                'eval_mode': 'seen',
                'train_frac': float(args.train_frac),
                'split_seed': float(args.split_seed),
                'method': args.method,
                'input_mode': str(resolved_input_mode),
                'struct_mode': str(resolved_struct_mode),
                'use_logits_input': float(bool(args.use_logits_input)),
                'use_gate': float(bool(args.use_gate)),
                'epsilon_gate': float(args.epsilon_gate) if args.use_gate else float('nan'),
                'lambda_gate': float(args.lambda_gate) if args.use_gate else float('nan'),
                'gate_use_logits_input': float(bool(args.gate_use_logits_input)) if args.use_gate else float('nan'),
                'gate_target': str(args.gate_target) if args.use_gate else '',
                'gate_weight_valid_mask': float(bool(args.gate_weight_valid_mask)) if args.use_gate else float('nan'),
                'gate_pos_weight': float(args.gate_pos_weight) if args.use_gate else float('nan'),
                'use_flip_expert': float(bool(args.use_flip_expert)),
                'flip_init_scale': float(args.flip_init_scale) if args.use_flip_expert else float('nan'),
                'flip_init_bias': float(args.flip_init_bias) if args.use_flip_expert else float('nan'),
                **metrics,
            }
            out_path = output_dir / (
                f"calib_{feature_dir.name}_{atk}_{args.method}_in{resolved_input_mode}_struct{resolved_struct_mode}_logits{int(bool(args.use_logits_input))}"
                f"_gate{int(bool(args.use_gate))}_flip{int(bool(args.use_flip_expert))}"
                f"_seen_tr{float(args.train_frac):.3f}_seed{int(args.split_seed)}.txt"
            )
            with open(out_path, 'w', encoding='utf-8') as f:
                for k, v in row.items():
                    f.write(f"{k}: {v}\n")
            print(
                "[seen-done] "
                f"attack={atk} | clean {metrics['clean_acc_before']:.4f}->{metrics['clean_acc_after']:.4f} | "
                f"adv {metrics['adv_acc_before']:.4f}->{metrics['adv_acc_after']:.4f} | "
                f"succ {metrics['succ_acc_before']:.4f}->{metrics['succ_acc_after']:.4f} "
                f"(n_succ={int(metrics['n_succ'])}) | mean_w_adv {metrics.get('mean_w_adv', float('nan')):.3f}\n"
                f"[save] {out_path}"
            )
            rows.append(row)
        return rows

    if args.eval_all_attacks:
        rows = _run_seen_all()
        csv_path = output_dir / (
            f"summary_{feature_dir.name}_{args.method}_in{resolved_input_mode}_struct{resolved_struct_mode}_logits{int(bool(args.use_logits_input))}"
            f"_gate{int(bool(args.use_gate))}_flip{int(bool(args.use_flip_expert))}"
            f"_seen_tr{float(args.train_frac):.3f}_seed{int(args.split_seed)}.csv"
        )
        keys = list(rows[0].keys())
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(','.join(keys) + '\n')
            for r in rows:
                f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
        print(f"\n[save] summary csv: {csv_path}")
        return

    if args.loao:
        rows = []
        for held_out in attacks:
            rows.append(_run_one(held_out))

        csv_path = output_dir / (
            f"summary_{feature_dir.name}_{args.method}_in{resolved_input_mode}_struct{resolved_struct_mode}_logits{int(bool(args.use_logits_input))}"
            f"_gate{int(bool(args.use_gate))}_flip{int(bool(args.use_flip_expert))}"
            f"_loao_tr{float(args.train_frac):.3f}_seed{int(args.split_seed)}.csv"
        )
        keys = list(rows[0].keys())
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(','.join(keys) + '\n')
            for r in rows:
                f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
        print(f"\n[save] summary csv: {csv_path}")
        return

    held_out = str(args.held_out_attack).strip()
    if held_out == '':
        held_out = attacks[0]
    _run_one(held_out)


if __name__ == '__main__':
    main()
