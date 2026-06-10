#!/usr/bin/env python3
"""
t-SNE visualization of CLEAN vs ADVERSARIAL samples on AIGI datasets.

For a given detector (UnivFD / ForgeLens) this script draws, for each attack,
two t-SNE panels using exactly the same set of clean + adversarial samples:

    1. "Original"   -> the detector's own penultimate feature (CLIP embedding for
                       UnivFD, the ln_post cls-token feature for ForgeLens), i.e.
                       the representation fed into the binary classification head.
    2. "Calibrated" -> the hidden representation of the appended logit-calibrator
                       network, computed from the per-layer LID feature vector and
                       the detector's raw logit (the calibrator's inputs).

Each panel is colored by 4 classes: Clean Real / Clean Fake / Adv Real / Adv Fake.

Model loading, attack generation and LID extraction are imported directly from
extract_lid_features.py so the behaviour matches feature extraction / training.

Example (UnivFD):
    python tsne_clean_adv.py \
        --model_type univfd \
        --data_dir /path/to/aigi_dataset \
        --model_path ./weights/univfd.pth \
        --calibrator_ckpt ./logit_calibrators/calib_univfd_seen.pt \
        --attacks pgd,apgd,cw,fab,square \
        --n_per_class 300 \
        --output_dir ./tsne_out

Example (ForgeLens):
    python tsne_clean_adv.py \
        --model_type forgelens \
        --data_dir /path/to/aigi_dataset \
        --model_path ./weights/forgelens_stage2.pth \
        --calibrator_ckpt ./logit_calibrators/calib_forgelens_seen.pt \
        --attacks pgd,apgd,cw,fab,square
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Re-use the exact building blocks from the extraction pipeline.
from extract_lid_features import (
    ImageDataset,
    CLIPImageTransform,
    ForgeLensImageTransform,
    CLIPFeatureExtractor,
    ForgeLensFeatureExtractor,
    compute_lid_features,
    generate_adversarial,
    _to_single_logit_tensor,
    _pred_labels_from_model_output,
    _load_trained_logit_calibrator,
)
from train_logit_calibrator import _load_npz_for_calibration


# ----------------------------- class / color setup -----------------------------
# 4 classes shown in the legend, matching the reference figure.
CLASS_NAMES = ['Clean Real', 'Clean Fake', 'Adv Real', 'Adv Fake']
CLASS_COLORS = {
    'Clean Real': '#1f77b4',  # blue
    'Clean Fake': '#ff7f0e',  # orange
    'Adv Real':   '#2ca02c',  # green
    'Adv Fake':   '#d62728',  # red
}


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _peek_calibrator_lid_dim(ckpt_path: str) -> Optional[int]:
    """Read lid_dim from a calibrator checkpoint without constructing the model."""
    ckpt_path = str(ckpt_path).strip()
    if ckpt_path == '':
        return None
    obj = torch.load(ckpt_path, map_location='cpu')
    if not isinstance(obj, dict):
        return None
    cfg = obj.get('config', None)
    if isinstance(cfg, dict) and 'lid_dim' in cfg:
        try:
            return int(cfg['lid_dim'])
        except Exception:
            return None
    sd = obj.get('state_dict', obj)
    if not isinstance(sd, dict):
        return None
    for key in ('calibrator.net.0.weight', 'net.0.weight'):
        w = sd.get(key, None)
        if torch.is_tensor(w) and w.dim() == 2:
            # This is only a fallback. For use_logits_input=True the real lid_dim
            # is input_dim - 1, but checkpoints saved by train_logit_calibrator.py
            # should carry config['lid_dim'] and use the branch above.
            return int(w.shape[1])
    return None


# ----------------------------- data collection -----------------------------
def _collect_images(roots: List[Path]) -> List[Path]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    out: List[Path] = []
    for root in roots:
        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in exts:
                out.append(p)
    return sorted(out)


def _discover_class_roots(data_dir: Path) -> Tuple[List[Path], List[Path]]:
    real_dir = data_dir / '0_real'
    fake_dir = data_dir / '1_fake'
    if real_dir.exists() and fake_dir.exists():
        return [real_dir], [fake_dir]
    real_roots = sorted([p for p in data_dir.rglob('0_real') if p.is_dir()])
    fake_roots = sorted([p for p in data_dir.rglob('1_fake') if p.is_dir()])
    if not real_roots or not fake_roots:
        raise ValueError(
            f"Could not find 0_real / 1_fake under {data_dir} "
            f"(found real={len(real_roots)} fake={len(fake_roots)})"
        )
    return real_roots, fake_roots


def _resolve_clean_data_dir(args) -> Path:
    external_clean_dir = str(getattr(args, 'external_clean_dir', '') or '').strip()
    if external_clean_dir != '':
        return Path(external_clean_dir)
    return Path(args.data_dir)


def _build_clean_split(
    data_dir: Path,
    n_per_class: int,
    ref_per_class: int,
    transform,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    """Returns (ref_loader, clean_loader, clean_true_labels).

    This follows the same balanced split logic as `extract_lid_features.py`:
    - `ref_per_class` corresponds to `ref_samples // 2`
    - `n_per_class` corresponds to `query_samples`
    - when data is insufficient, reduce the split the same way while preserving
      at least one reference sample per class and, when possible, one query.
    """
    real_roots, fake_roots = _discover_class_roots(data_dir)
    real_all = _collect_images(real_roots)
    fake_all = _collect_images(fake_roots)

    rng = np.random.default_rng(int(seed))
    rng.shuffle(real_all)
    rng.shuffle(fake_all)

    requested_ref_half = int(ref_per_class)
    requested_query = int(n_per_class)
    available_real = len(real_all)
    available_fake = len(fake_all)

    if available_real == 0:
        raise ValueError(f"Not enough real images under {data_dir}: found 0 (roots={len(real_roots)})")
    if available_fake == 0:
        raise ValueError(f"Not enough fake images under {data_dir}: found 0 (roots={len(fake_roots)})")

    total_balanced = min(available_real, available_fake)
    if total_balanced <= 0:
        raise ValueError(
            f"Not enough balanced images under {data_dir}: need at least 1 per class, "
            f"found real={available_real}, fake={available_fake}"
        )

    if total_balanced >= requested_ref_half + requested_query:
        usable_ref_half = requested_ref_half
        usable_query = requested_query
    else:
        usable_ref_half = min(requested_ref_half, max(1, total_balanced // 2))
        usable_query = total_balanced - usable_ref_half
        if usable_query <= 0 and total_balanced >= 2:
            usable_query = 1
            usable_ref_half = total_balanced - 1

    if usable_ref_half <= 0:
        usable_ref_half = 1
    if usable_query < 0:
        usable_query = 0

    if usable_ref_half < requested_ref_half or usable_query < requested_query:
        print(
            f"[warn] reducing sample request for {data_dir}: "
            f"requested_ref_half={requested_ref_half}, requested_query={requested_query}, "
            f"usable_ref_half={usable_ref_half}, usable_query={usable_query}, "
            f"available_real={available_real}, available_fake={available_fake}, total_balanced={total_balanced}"
        )

    real_images = real_all[:usable_ref_half + usable_query]
    fake_images = fake_all[:usable_ref_half + usable_query]

    ref_real = real_images[:usable_ref_half]
    ref_fake = fake_images[:usable_ref_half]
    q_real = real_images[usable_ref_half:usable_ref_half + usable_query]
    q_fake = fake_images[usable_ref_half:usable_ref_half + usable_query]

    print(
        f"[data] ref_real={len(ref_real)} ref_fake={len(ref_fake)} "
        f"query_real={len(q_real)} query_fake={len(q_fake)} "
        f"usable_ref_half={usable_ref_half} usable_query={usable_query}"
    )
    if len(ref_real) == 0 or len(ref_fake) == 0:
        raise ValueError(
            f"Reference split missing a class: ref_real={len(ref_real)} ref_fake={len(ref_fake)}. "
            f"Check data_dir layout and ref_samples={int(2 * ref_per_class)}."
        )
    if len(q_real) == 0 or len(q_fake) == 0:
        print(
            f"[warn] Query split missing samples: query_real={len(q_real)} query_fake={len(q_fake)}. "
            f"Proceeding with the reduced query subset for {data_dir}."
        )

    ref_ds = ImageDataset(ref_real + ref_fake,
                          [0] * len(ref_real) + [1] * len(ref_fake),
                          transform=transform)
    clean_ds = ImageDataset(q_real + q_fake,
                            [0] * len(q_real) + [1] * len(q_fake),
                            transform=transform)

    ref_loader = DataLoader(ref_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    clean_labels = np.asarray(clean_ds.labels, dtype=np.int64)
    return ref_loader, clean_loader, clean_labels


# ----------------------------- model building -----------------------------
def build_detector(args, device) -> Tuple[nn.Module, List[str], object]:
    mt = str(args.model_type).lower()
    include_input = bool(getattr(args, 'include_input', False))
    if mt in ('univfd', 'clip'):
        model = CLIPFeatureExtractor(
            model_name=str(args.clip_model),
            model_path=(args.model_path if args.model_path else None),
            fc_weights_path=(args.fc_weights if args.fc_weights else None),
            device=device,
            feat_mode=str(args.clip_feat_mode),
            include_input=include_input,
            apply_input_normalize=True,
        )
        layers = [f'layer{i}' for i in range(model.get_num_layers())]
        transform = CLIPImageTransform(
            load_size=int(args.clip_load_size),
            crop_size=int(args.clip_crop_size),
            resize_mode=str(args.clip_resize_mode),
        )
    elif mt == 'forgelens':
        if not args.model_path or not os.path.exists(str(args.model_path)):
            raise ValueError("forgelens requires a valid --model_path")
        model = ForgeLensFeatureExtractor(
            model_path=str(args.model_path),
            device=str(args.device),
            include_input=include_input,
            stage=int(args.forgelens_stage),
            feature_set=str(args.forgelens_feature_set),
            wsgm_count=int(args.forgelens_wsgm_count),
            wsgm_reduction_factor=int(args.forgelens_wsgm_reduction_factor),
            faformer_layers=int(args.forgelens_faformer_layers),
            faformer_head=int(args.forgelens_faformer_head),
            faformer_reduction_factor=int(args.forgelens_faformer_reduction_factor),
        )
        fa_n = int(getattr(model, 'num_layers', int(args.forgelens_faformer_layers)))
        clip_n = int(getattr(model, 'num_clip_layers', 24))
        fs = str(args.forgelens_feature_set)
        if fs == 'faformer':
            layers = [f'layer{i}' for i in range(fa_n)]
        elif fs == 'clip_proj':
            layers = [f'clip_layer{i}' for i in range(clip_n)]
        elif fs == 'clip_unproj':
            layers = [f'clipu_layer{i}' for i in range(clip_n)]
        elif fs == 'clip_both':
            layers = [f'clip_layer{i}' for i in range(clip_n)] + [f'clipu_layer{i}' for i in range(clip_n)]
        elif fs == 'all_proj':
            layers = [f'layer{i}' for i in range(fa_n)] + [f'clip_layer{i}' for i in range(clip_n)]
        elif fs == 'all_unproj':
            layers = [f'layer{i}' for i in range(fa_n)] + [f'clipu_layer{i}' for i in range(clip_n)]
        else:
            layers = ([f'layer{i}' for i in range(fa_n)]
                      + [f'clip_layer{i}' for i in range(clip_n)]
                      + [f'clipu_layer{i}' for i in range(clip_n)])
        transform = ForgeLensImageTransform(size=int(args.img_crop_size))
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    if include_input and 'input' not in layers:
        layers = ['input'] + layers

    if args.layers:
        raw = [s.strip() for s in str(args.layers).split(',') if s.strip()]
        sel = []
        for item in raw:
            if mt in ('univfd', 'clip') and item.isdigit():
                sel.append(f'layer{item}')
            else:
                sel.append(item)
        layers = sel

    model.eval()
    return model, layers, transform


def _get_head_linear(model: nn.Module, model_type: str) -> nn.Linear:
    """Return the final binary classification Linear whose INPUT is the
    penultimate detector feature we visualize."""
    mt = str(model_type).lower()
    if mt in ('univfd', 'clip'):
        fc = getattr(model, 'fc', None)
        if isinstance(fc, nn.Linear):
            return fc
        if fc is None:
            raise RuntimeError("UnivFD model has no fc head; provide --model_path / --fc_weights")
    # forgelens: model.model.fc is Sequential(Dropout, Linear)
    inner = getattr(model, 'model', None)
    head = getattr(inner, 'fc', None) if inner is not None else None
    if isinstance(head, nn.Linear):
        return head
    if isinstance(head, nn.Sequential):
        for m in reversed(list(head.modules())):
            if isinstance(m, nn.Linear):
                return m
    # generic fallback: last Linear with small output dim
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear) and int(m.out_features) <= 2:
            last = m
    if last is None:
        raise RuntimeError("Could not locate a binary classification head Linear")
    return last


class _PenultimateHook:
    """Captures the input tensor to the classification head Linear."""

    def __init__(self, model: nn.Module, model_type: str):
        self.layer = _get_head_linear(model, model_type)
        self.store: Optional[torch.Tensor] = None
        self._handle = self.layer.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        x = inputs[0]
        self.store = x.detach().reshape(x.shape[0], -1).float().cpu()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@torch.no_grad()
def extract_logits_and_feats(
    model: nn.Module,
    model_type: str,
    loader: DataLoader,
    device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the detector over a loader, returning (z, penultimate_feat).

    z: (N,) single logit ; feat: (N, D) penultimate feature.
    """
    hook = _PenultimateHook(model, model_type)
    z_list: List[np.ndarray] = []
    feat_list: List[np.ndarray] = []
    try:
        for images, _ in loader:
            images = images.to(device)
            out = model(images)
            z = _to_single_logit_tensor(out).reshape(-1).detach().cpu().numpy()
            z_list.append(z.astype(np.float32))
            if hook.store is None:
                raise RuntimeError("Penultimate hook did not capture features")
            feat_list.append(hook.store.numpy().astype(np.float32))
            hook.store = None
    finally:
        hook.remove()
    return np.concatenate(z_list, axis=0), np.concatenate(feat_list, axis=0)


# ----------------------------- attack -----------------------------
def detect_label_flip(model, model_type, ref_loader, true_labels_hint, device) -> bool:
    """Probe whether the detector uses an inverted (real/fake) label convention."""
    hook = _PenultimateHook(model, model_type)
    try:
        preds = []
        labs = []
        with torch.no_grad():
            for images, labels in ref_loader:
                images = images.to(device)
                out = model(images)
                preds.append(_pred_labels_from_model_output(out).detach().cpu().numpy())
                labs.append(np.asarray(labels).reshape(-1))
                hook.store = None
        preds = np.concatenate(preds)
        labs = np.concatenate(labs)
    finally:
        hook.remove()
    acc = float(np.mean(preds == labs))
    acc_flip = float(np.mean(preds == (1 - labs)))
    flip = acc_flip > acc
    print(f"[flip-probe] acc={acc:.4f} acc_flip={acc_flip:.4f} -> flip={flip}")
    return flip


def generate_adv_loader(
    model,
    clean_loader: DataLoader,
    clean_labels: np.ndarray,
    flip: bool,
    attack: str,
    args,
    device,
) -> DataLoader:
    """Generate adversarial images for every clean sample (attacking the base
    detector). Returns a loader aligned 1:1 with clean_loader (same order)."""
    adv_chunks = []
    offset = 0
    for images, labels in clean_loader:
        images = images.to(device)
        labels = labels.to(device)
        eff = (1 - labels) if flip else labels
        adv = generate_adversarial(
            model,
            images,
            eff,
            attack_type=str(attack),
            eps=float(args.epsilon),
            alpha=float(args.alpha),
            steps=int(args.steps),
            apgd_restarts=int(args.apgd_restarts),
            fab_restarts=int(args.fab_restarts),
            square_queries=int(args.square_queries),
            square_restarts=int(args.square_restarts),
            cw_c=float(args.cw_c),
            cw_kappa=float(args.cw_kappa),
            cw_lr=float(args.cw_lr),
            cw_binary_search_steps=args.cw_binary_search_steps,
            device=device,
        )
        adv_chunks.append(adv.detach().cpu())
        offset += int(images.shape[0])
    adv_images = torch.cat(adv_chunks, dim=0)
    adv_ds = TensorDataset(adv_images, torch.from_numpy(np.asarray(clean_labels, dtype=np.int64)))
    return DataLoader(adv_ds, batch_size=int(args.batch_size), shuffle=False)


# ----------------------------- LID -----------------------------
def compute_lid(
    model,
    layers,
    ref_loader,
    query_loader,
    query_type,
    args,
    device,
    batch_ref_loader=None,
) -> np.ndarray:
    spatial = 'flatten'
    if str(args.ref_mode) == 'batch_clean':
        return compute_lid_features(
            model, ref_loader, query_loader, query_type, layers,
            k=int(args.lid_k), chunk_size=int(args.chunk_size), device=device,
            feat_norm=('l2' if args.feat_norm == 'l2' else 'none'),
            distance_metric=str(args.lid_distance_metric),
            ref_mode='batch_clean',
            batch_ref_loader=(batch_ref_loader if batch_ref_loader is not None else query_loader),
            batch_ref_size=int(args.batch_ref_size),
            spatial_feat_mode=spatial,
        )
    return compute_lid_features(
        model, ref_loader, query_loader, query_type, layers,
        k=int(args.lid_k), chunk_size=int(args.chunk_size), device=device,
        feat_norm=('l2' if args.feat_norm == 'l2' else 'none'),
        distance_metric=str(args.lid_distance_metric),
        ref_mode='global',
        spatial_feat_mode=spatial,
    )


# ----------------------------- calibrator hidden features -----------------------------
def _base_calibrator(calibrator: nn.Module) -> nn.Module:
    # GatedCalibrator / FlipMoECalibrator / FlipOnlyCalibrator expose `.calibrator`
    inner = getattr(calibrator, 'calibrator', None)
    return inner if inner is not None else calibrator


@torch.no_grad()
def calibrator_hidden_feats(
    calibrator: nn.Module,
    lid: np.ndarray,
    z: np.ndarray,
    use_logits_input: bool,
    device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Hidden (penultimate) representation of the calibrator network.

    The calibrator's `net` is Linear -> ReLU -> ... -> Linear(.,1). We take the
    activation right before the final Linear as the learned calibrator feature.
    """
    base = _base_calibrator(calibrator)
    net = base.net
    feat_net = nn.Sequential(*list(net.children())[:-1])  # drop final Linear

    lid = np.asarray(lid, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32).reshape(-1, 1)
    n = lid.shape[0]
    out_list = []
    for i in range(0, n, batch_size):
        lid_b = torch.from_numpy(lid[i:i + batch_size]).to(device)
        z_b = torch.from_numpy(z[i:i + batch_size]).to(device)
        if use_logits_input:
            x = torch.cat([lid_b, z_b], dim=1)
        else:
            x = lid_b
        h = feat_net(x)
        out_list.append(h.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out_list, axis=0)


@torch.no_grad()
def calibrator_corrected_logits(
    calibrator: nn.Module,
    lid: np.ndarray,
    z: np.ndarray,
    use_logits_input: bool,
    device,
    batch_size: int = 1024,
) -> np.ndarray:
    lid = np.asarray(lid, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32).reshape(-1, 1)
    out_list = []
    for i in range(0, int(lid.shape[0]), int(batch_size)):
        lid_b = torch.from_numpy(lid[i:i + batch_size]).to(device)
        z_b = torch.from_numpy(z[i:i + batch_size]).to(device)
        out = calibrator(lid_b, z_b, use_logits_input=use_logits_input)
        out_list.append(out['z_corr'].detach().cpu().numpy().reshape(-1))
    return np.concatenate(out_list, axis=0)


# ----------------------------- t-SNE + plotting -----------------------------
def run_tsne(feats: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    from sklearn.manifold import TSNE
    feats = np.asarray(feats, dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    n = feats.shape[0]
    perp = float(min(float(perplexity), max(5.0, (n - 1) / 3.0)))
    common = dict(
        n_components=2,
        perplexity=perp,
        init='pca',
        learning_rate='auto',
        random_state=int(seed),
    )
    # sklearn renamed `n_iter` -> `max_iter` in 1.5; support both.
    try:
        tsne = TSNE(max_iter=1000, **common)
    except TypeError:
        tsne = TSNE(n_iter=1000, **common)
    return tsne.fit_transform(feats)


def make_panel_labels(clean_labels: np.ndarray) -> np.ndarray:
    """Build the 4-class label vector for [clean(real,fake), adv(real,fake)].

    Returns an array of class names with length 2*N (clean block then adv block).
    """
    clean_labels = np.asarray(clean_labels).reshape(-1)
    clean_cls = np.where(clean_labels == 0, 'Clean Real', 'Clean Fake')
    adv_cls = np.where(clean_labels == 0, 'Adv Real', 'Adv Fake')
    return np.concatenate([clean_cls, adv_cls], axis=0)


def scatter_panel(ax, emb: np.ndarray, panel_labels: np.ndarray, point_size: float = 8.0):
    for cls in CLASS_NAMES:
        mask = (panel_labels == cls)
        if not np.any(mask):
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   s=point_size, c=CLASS_COLORS[cls], label=cls,
                   edgecolors='none', alpha=0.75)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _resolve_npz_paths(npz_paths: str, feature_dir: str) -> List[Path]:
    paths: List[Path] = []
    raw = str(npz_paths or '').strip()
    if raw:
        for item in raw.split(','):
            item = item.strip()
            if item:
                paths.append(Path(item))
    feature_dir_raw = str(feature_dir or '').strip()
    if feature_dir_raw:
        root = Path(feature_dir_raw)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"feature_dir does not exist or is not a directory: {root}")
        paths.extend(sorted([p for p in root.glob('*.npz') if p.is_file()]))
    uniq = []
    seen = set()
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            uniq.append(p)
            seen.add(key)
    if not uniq:
        raise ValueError("NPZ mode requires --npz_paths or --feature_dir")
    return uniq


def _npz_base_features(lid: np.ndarray, z: np.ndarray, mode: str) -> np.ndarray:
    lid = np.asarray(lid, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32).reshape(-1, 1)
    mode = str(mode).lower()
    if mode == 'lid':
        return lid
    if mode == 'logit':
        return z
    if mode == 'lid_logit':
        return np.concatenate([lid, z], axis=1)
    raise ValueError(f"Unknown npz_base_feature: {mode}")


def run_npz_mode(args, device, output_dir: Path, exp_name: str) -> None:
    npz_paths = _resolve_npz_paths(args.npz_paths, args.feature_dir)
    print(f"[npz] using {len(npz_paths)} npz file(s): {[p.name for p in npz_paths]}")

    first = _load_npz_for_calibration(npz_paths[0], require_pairing=True)
    lid_dim = int(first.lid_clean.shape[1])
    calibrator, calib_cfg = _load_trained_logit_calibrator(
        str(args.calibrator_ckpt), lid_dim=lid_dim, device=device)
    calib_use_logits = bool(calib_cfg.get('use_logits_input', False))
    print(
        f"[npz] loaded calibrator (lid_dim={lid_dim}, use_logits_input={calib_use_logits}, "
        f"struct={calib_cfg.get('struct_mode', calib_cfg.get('method'))})"
    )

    n_attacks = len(npz_paths)
    fig, axes = plt.subplots(2, n_attacks, figsize=(2.6 * n_attacks, 5.4), squeeze=False)
    saved = {}

    for col, npz_path in enumerate(npz_paths):
        item = _load_npz_for_calibration(npz_path, require_pairing=True)
        if int(item.lid_clean.shape[1]) != lid_dim or int(item.lid_adv.shape[1]) != lid_dim:
            raise ValueError(
                f"LID dim mismatch in {npz_path}: clean={item.lid_clean.shape} adv={item.lid_adv.shape}, "
                f"expected lid_dim={lid_dim}"
            )

        clean_labels = np.asarray(item.y_clean_eff, dtype=np.int64).reshape(-1)
        adv_labels = np.asarray(item.y_adv_eff, dtype=np.int64).reshape(-1)
        if len(clean_labels) != len(adv_labels):
            raise ValueError(f"Expected paired clean/adv labels in {npz_path}")
        panel_labels = np.concatenate([
            np.where(clean_labels == 0, 'Clean Real', 'Clean Fake'),
            np.where(adv_labels == 0, 'Adv Real', 'Adv Fake'),
        ], axis=0)

        z_clean_corr = calibrator_corrected_logits(
            calibrator, item.lid_clean, item.z_clean, calib_use_logits, device,
            batch_size=int(args.batch_size),
        )
        z_adv_corr = calibrator_corrected_logits(
            calibrator, item.lid_adv, item.z_adv, calib_use_logits, device,
            batch_size=int(args.batch_size),
        )
        clean_acc = float(((z_clean_corr > 0).astype(np.int64) == clean_labels).mean())
        adv_acc = float(((z_adv_corr > 0).astype(np.int64) == adv_labels).mean())
        print(f"[npz][{item.attack}] clean_acc={clean_acc:.4f} adv_acc={adv_acc:.4f} n={len(clean_labels)}")

        base_feats = np.concatenate([
            _npz_base_features(item.lid_clean, item.z_clean, args.npz_base_feature),
            _npz_base_features(item.lid_adv, item.z_adv, args.npz_base_feature),
        ], axis=0)
        h_clean = calibrator_hidden_feats(
            calibrator, item.lid_clean, item.z_clean, calib_use_logits, device,
            batch_size=int(args.batch_size),
        )
        h_adv = calibrator_hidden_feats(
            calibrator, item.lid_adv, item.z_adv, calib_use_logits, device,
            batch_size=int(args.batch_size),
        )
        calib_feats = np.concatenate([h_clean, h_adv], axis=0)

        emb_base = run_tsne(base_feats, args.seed, args.tsne_perplexity)
        emb_calib = run_tsne(calib_feats, args.seed, args.tsne_perplexity)

        scatter_panel(axes[0][col], emb_base, panel_labels, point_size=args.point_size)
        scatter_panel(axes[1][col], emb_calib, panel_labels, point_size=args.point_size)
        axes[0][col].set_title(str(item.attack).upper(), fontsize=12)

        if args.save_embeddings:
            saved[str(item.attack)] = {
                'emb_base': emb_base,
                'emb_calib': emb_calib,
                'panel_labels': panel_labels,
            }

    axes[0][0].set_ylabel(f"NPZ {args.npz_base_feature}", fontsize=12)
    axes[1][0].set_ylabel('Calibrated', fontsize=12)

    handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=8,
                          markerfacecolor=CLASS_COLORS[c], markeredgecolor='none', label=c)
               for c in CLASS_NAMES]
    fig.legend(handles=handles, labels=CLASS_NAMES, loc='lower center',
               ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02))

    title_map = {
        'clip': 'UnivFD',
        'univfd': 'UnivFD',
        'cnnspot': 'CNNSpot',
        'gram': 'Gram',
        'forgelens': 'ForgeLens',
        'csf': 'CSF',
    }
    title_model = title_map.get(str(args.model_type).lower(), str(args.model_type))
    fig.suptitle(f"{title_model} (from NPZ)", fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    out_png = output_dir / f"{exp_name}.png"
    out_pdf = output_dir / f"{exp_name}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"\n[save] {out_png}")
    print(f"[save] {out_pdf}")

    if args.save_embeddings:
        npz_out = output_dir / f"{exp_name}_embeddings.npz"
        flat = {}
        for atk, d in saved.items():
            flat[f'{atk}_emb_base'] = d['emb_base']
            flat[f'{atk}_emb_calib'] = d['emb_calib']
            flat[f'{atk}_panel_labels'] = d['panel_labels']
        np.savez(npz_out, **flat)
        print(f"[save] {npz_out}")


# ----------------------------- main -----------------------------
def main():
    parser = argparse.ArgumentParser(description='t-SNE of clean vs adversarial samples (base vs calibrator features)')

    parser.add_argument(
        '--model_type',
        type=str,
        default='univfd',
        choices=['univfd', 'clip', 'cnnspot', 'gram', 'forgelens', 'csf'],
    )
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--fc_weights', type=str, default='')
    parser.add_argument('--clip_model', type=str, default='ViT-L/14')
    parser.add_argument('--clip_feat_mode', type=str, default='flatten_tokens',
                        choices=['mean_patch', 'cls', 'flatten', 'flatten_tokens'])
    parser.add_argument('--clip_load_size', type=int, default=256)
    parser.add_argument('--clip_crop_size', type=int, default=224)
    parser.add_argument('--clip_resize_mode', type=str, default='short_side', choices=['short_side', 'square'])

    parser.add_argument('--forgelens_stage', type=int, default=2, choices=[1, 2])
    parser.add_argument('--forgelens_feature_set', type=str, default='all_proj',
                        choices=['faformer', 'clip_proj', 'clip_unproj', 'clip_both', 'all_proj', 'all_unproj', 'all_both'])
    parser.add_argument('--forgelens_wsgm_count', type=int, default=4)
    parser.add_argument('--forgelens_wsgm_reduction_factor', type=int, default=4)
    parser.add_argument('--forgelens_faformer_layers', type=int, default=2)
    parser.add_argument('--forgelens_faformer_reduction_factor', type=int, default=1)
    parser.add_argument('--forgelens_faformer_head', type=int, default=2)
    parser.add_argument('--img_crop_size', type=int, default=224)

    parser.add_argument('--include_input', action='store_true',
                        help='Prepend the raw input as an extra LID layer (adds 1 to lid_dim). '
                             'Set this if the calibrator was trained with --include_input.')
    parser.add_argument('--no_include_input', action='store_false', dest='include_input',
                        help='Disable the raw input LID layer.')
    parser.add_argument('--layers', type=str, default=None,
                        help='Override LID layers (must match the calibrator checkpoint lid_dim)')

    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--external_clean_dir', type=str, default='',
                        help='Use this directory as the clean/ref source, matching extract_lid_features.py.')
    parser.add_argument('--n_per_class', type=int, default=300, help='clean real / clean fake query samples each')
    parser.add_argument('--query_samples', type=int, default=0,
                        help='extract_lid_features.py-style balanced query samples per class. If set, overrides n_per_class.')
    parser.add_argument('--ref_per_class', type=int, default=500, help='LID reference images per class')
    parser.add_argument('--ref_samples', type=int, default=1000,
                        help='extract_lid_features.py-style total reference samples. If set, ref_per_class = ref_samples // 2.')

    parser.add_argument('--calibrator_ckpt', type=str, required=True)

    parser.add_argument('--attacks', type=str, default='pgd,apgd,cw,fab,square')
    parser.add_argument('--feature_dir', type=str, default='',
                        help='Directory containing extract_lid_features.py generated *.npz files. '
                             'If set, plot directly from npz and skip model/attack/LID extraction.')
    parser.add_argument('--npz_paths', type=str, default='',
                        help='Comma-separated extract_lid_features.py generated npz paths. '
                             'If set, plot directly from npz and skip model/attack/LID extraction.')
    parser.add_argument('--npz_base_feature', type=str, default='lid_logit',
                        choices=['lid', 'logit', 'lid_logit'],
                        help='Feature used for the top row in NPZ mode. The original detector '
                             'penultimate feature is not stored in extract_lid_features.py npz files.')

    # attack hyper-params (mirror extract_lid_features.py defaults)
    parser.add_argument('--epsilon', type=float, default=8 / 255)
    parser.add_argument('--alpha', type=float, default=2 / 255)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--apgd_restarts', type=int, default=1)
    parser.add_argument('--fab_restarts', type=int, default=1)
    parser.add_argument('--square_queries', type=int, default=500)
    parser.add_argument('--square_restarts', type=int, default=1)
    parser.add_argument('--cw_c', type=float, default=1.0)
    parser.add_argument('--cw_kappa', type=float, default=0.0)
    parser.add_argument('--cw_lr', type=float, default=0.01)
    parser.add_argument('--cw_binary_search_steps', type=int, default=None)

    # LID settings (must match how the calibrator was trained)
    parser.add_argument('--lid_k', type=int, default=20)
    parser.add_argument('--feat_norm', type=str, default='none', choices=['l2', 'none'])
    parser.add_argument('--lid_distance_metric', type=str, default='euclidean',
                        choices=['euclidean', 'cosine', 'manhattan', 'chebyshev'])
    parser.add_argument('--ref_mode', type=str, default='batch_clean', choices=['global', 'batch_clean'])
    parser.add_argument('--batch_ref_size', type=int, default=100)
    parser.add_argument('--chunk_size', type=int, default=512)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tsne_perplexity', type=float, default=30.0)
    parser.add_argument('--point_size', type=float, default=8.0)

    parser.add_argument('--output_dir', type=str, default='./tsne_out')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--save_embeddings', action='store_true', help='also dump per-panel embeddings to .npz')

    parser.set_defaults(include_input=True)
    args = parser.parse_args()
    _seed_everything(args.seed)

    if int(getattr(args, 'query_samples', 0)) > 0:
        args.n_per_class = int(args.query_samples)
    if int(getattr(args, 'ref_samples', 0)) > 0:
        if int(args.ref_samples) % 2 != 0:
            raise ValueError(f"ref_samples must be even, got {int(args.ref_samples)}")
        args.ref_per_class = int(args.ref_samples) // 2

    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attacks = [a.strip().lower() for a in str(args.attacks).split(',') if a.strip()]
    npz_mode = bool(str(args.feature_dir).strip() or str(args.npz_paths).strip())
    if len(attacks) == 0 and not npz_mode:
        raise ValueError("No attacks specified")

    exp_name = args.exp_name or f"tsne_{args.model_type}"

    if npz_mode:
        run_npz_mode(args, device, output_dir, exp_name)
        return

    if str(args.data_dir).strip() == '':
        raise ValueError("--data_dir is required unless --feature_dir or --npz_paths is used")

    # ---- 1. build detector + data ----
    print(f"[info] building {args.model_type} detector ...")
    model, layers, transform = build_detector(args, device)

    expected_lid_dim = _peek_calibrator_lid_dim(str(args.calibrator_ckpt))
    if expected_lid_dim is not None:
        if (
            not bool(getattr(args, 'include_input', False))
            and not args.layers
            and int(expected_lid_dim) == int(len(layers)) + 1
        ):
            args.include_input = True
            if hasattr(model, 'include_input'):
                model.include_input = True
            layers = ['input'] + layers
            print(
                f"[info] calibrator checkpoint expects lid_dim={int(expected_lid_dim)}; "
                "auto-enabled --include_input."
            )
        elif int(expected_lid_dim) != int(len(layers)):
            raise ValueError(
                f"Calibrator checkpoint expects lid_dim={int(expected_lid_dim)}, "
                f"but the current LID layer set has {int(len(layers))} layers: {layers}. "
                "Please pass --include_input or --layers to match the feature extraction "
                "settings used when training the calibrator."
            )
    print(f"[info] LID layers ({len(layers)}): {layers[:4]}{' ...' if len(layers) > 4 else ''}")

    clean_data_dir = _resolve_clean_data_dir(args)
    if str(clean_data_dir) != str(Path(args.data_dir)):
        print(f"[info] using external_clean_dir for clean/ref data: {clean_data_dir}")

    ref_loader, clean_loader, clean_labels = _build_clean_split(
        clean_data_dir, int(args.n_per_class), int(args.ref_per_class),
        transform, int(args.batch_size), int(args.seed),
    )
    n_clean = int(len(clean_labels))
    print(f"[info] clean query samples: {n_clean} (real={int(np.sum(clean_labels == 0))}, fake={int(np.sum(clean_labels == 1))})")

    # ---- 2. clean detector features / logits / LID (shared across attacks) ----
    print("[info] extracting clean logits + penultimate features ...")
    z_clean, feat_clean = extract_logits_and_feats(model, args.model_type, clean_loader, device)
    print("[info] computing clean LID features ...")
    lid_clean = compute_lid(
        model, layers, ref_loader, clean_loader, 'clean', args, device,
        batch_ref_loader=clean_loader,
    )
    print(f"[info] lid_clean shape: {lid_clean.shape}")

    # ---- 3. load calibrator (lid_dim validated against clean LID) ----
    calibrator, calib_cfg = _load_trained_logit_calibrator(
        str(args.calibrator_ckpt), lid_dim=int(lid_clean.shape[1]), device=device)
    calib_use_logits = bool(calib_cfg.get('use_logits_input', False))
    print(f"[info] loaded calibrator (use_logits_input={calib_use_logits}, "
          f"struct={calib_cfg.get('struct_mode', calib_cfg.get('method'))})")

    h_clean = calibrator_hidden_feats(calibrator, lid_clean, z_clean, calib_use_logits, device)
    # ---- 4. label flip probe (attack direction) ----
    flip = detect_label_flip(model, args.model_type, ref_loader, clean_labels, device)
    z_clean_corr = calibrator_corrected_logits(
        calibrator, lid_clean, z_clean, calib_use_logits, device,
        batch_size=int(args.batch_size),
    )
    clean_effective_labels = (1 - clean_labels) if flip else clean_labels
    clean_calib_acc = float(((z_clean_corr > 0).astype(np.int64) == clean_effective_labels).mean())
    print(f"[info] calibrator clean accuracy on visualization subset: {clean_calib_acc:.4f}")

    panel_labels = make_panel_labels(clean_labels)

    # ---- 5. per-attack: generate adv, extract feats, build panels ----
    n_attacks = len(attacks)
    fig, axes = plt.subplots(2, n_attacks, figsize=(2.6 * n_attacks, 5.4), squeeze=False)

    saved = {}
    for col, attack in enumerate(attacks):
        print(f"\n========== attack: {attack} ==========")
        adv_loader = generate_adv_loader(model, clean_loader, clean_labels, flip, attack, args, device)

        print("[info] extracting adv logits + penultimate features ...")
        z_adv, feat_adv = extract_logits_and_feats(model, args.model_type, adv_loader, device)
        print("[info] computing adv LID features ...")
        lid_adv = compute_lid(
            model, layers, ref_loader, adv_loader, f'adv_{attack}', args, device,
            batch_ref_loader=clean_loader,
        )
        h_adv = calibrator_hidden_feats(calibrator, lid_adv, z_adv, calib_use_logits, device)

        # Panel A: original detector penultimate feature
        base_feats = np.concatenate([feat_clean, feat_adv], axis=0)
        emb_base = run_tsne(base_feats, args.seed, args.tsne_perplexity)

        # Panel B: calibrator hidden feature
        calib_feats = np.concatenate([h_clean, h_adv], axis=0)
        emb_calib = run_tsne(calib_feats, args.seed, args.tsne_perplexity)

        scatter_panel(axes[0][col], emb_base, panel_labels, point_size=args.point_size)
        scatter_panel(axes[1][col], emb_calib, panel_labels, point_size=args.point_size)
        axes[0][col].set_title(attack.upper(), fontsize=12)

        if args.save_embeddings:
            saved[attack] = dict(emb_base=emb_base, emb_calib=emb_calib)

    axes[0][0].set_ylabel('Original', fontsize=12)
    axes[1][0].set_ylabel('Calibrated', fontsize=12)

    # shared legend at the bottom
    handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=8,
                          markerfacecolor=CLASS_COLORS[c], markeredgecolor='none', label=c)
               for c in CLASS_NAMES]
    fig.legend(handles=handles, labels=CLASS_NAMES, loc='lower center',
               ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02))

    title_model = 'UnivFD' if str(args.model_type).lower() in ('univfd', 'clip') else 'ForgeLens'
    fig.suptitle(title_model, fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    out_png = output_dir / f"{exp_name}.png"
    out_pdf = output_dir / f"{exp_name}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f"\n[save] {out_png}")
    print(f"[save] {out_pdf}")

    if args.save_embeddings:
        npz_path = output_dir / f"{exp_name}_embeddings.npz"
        flat = {'panel_labels': panel_labels, 'clean_labels': clean_labels}
        for atk, d in saved.items():
            flat[f'{atk}_emb_base'] = d['emb_base']
            flat[f'{atk}_emb_calib'] = d['emb_calib']
        np.savez(npz_path, **flat)
        print(f"[save] {npz_path}")


if __name__ == '__main__':
    main()
