#!/usr/bin/env python3
"""
Extract per-layer LID features for training classifier (similar to LID/ project)
Supports both ResNet and CLIP models
"""

import os
import sys
import argparse
import csv
import contextlib
import inspect
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from networks import resnet_gram as ResnetGram

try:
    from robust_lid_detector.lid_calculator_gpu import mle_batch_gpu
except Exception:
    def mle_batch_gpu(ref: torch.Tensor, query: torch.Tensor, k: int = 20, distance_metric: str = 'euclidean') -> torch.Tensor:
        eps = 1e-10
        if not torch.is_tensor(ref):
            ref = torch.as_tensor(ref)
        if not torch.is_tensor(query):
            query = torch.as_tensor(query)

        ref = ref.to(device=query.device, dtype=torch.float32)
        query = query.to(device=query.device, dtype=torch.float32)

        if distance_metric == 'euclidean':
            d = torch.cdist(query, ref, p=2)
        elif distance_metric == 'cosine':
            qn = F.normalize(query, p=2, dim=1)
            rn = F.normalize(ref, p=2, dim=1)
            d = 1.0 - (qn @ rn.t())
        elif distance_metric == 'manhattan':
            d = torch.cdist(query, ref, p=1)
        elif distance_metric == 'chebyshev':
            d = torch.cdist(query, ref, p=float('inf'))
        else:
            raise ValueError(f"Unknown distance_metric: {distance_metric}")

        if int(d.shape[1]) == 0:
            return torch.full((int(d.shape[0]),), float('nan'), device=d.device, dtype=d.dtype)

        k_eff = int(min(int(k), int(d.shape[1])))
        if k_eff <= 0:
            return torch.full((int(d.shape[0]),), float('nan'), device=d.device, dtype=d.dtype)

        d2 = d + (d <= eps).to(d.dtype) * 1e6
        knn, _ = torch.topk(d2, k=k_eff, dim=1, largest=False, sorted=True)
        r_k = knn[:, -1:] + eps
        ratios = knn / r_k
        lid = -float(k_eff) / (torch.sum(torch.log(ratios + eps), dim=1) + eps)
        lid = torch.where(torch.isfinite(lid), lid, torch.zeros_like(lid))
        return torch.clamp(lid, 0.0, 10000.0)

try:
    import torchattacks as ta
    _HAS_TA = True
except ImportError:
    _HAS_TA = False

try:
    _evade_root = Path(__file__).parent / 'evadingfakedetector-main'
    if _evade_root.exists():
        sys.path.insert(0, str(_evade_root))
    from attack.stat_attack import StatAttack as _StatAttack
    _HAS_STAT = True
except Exception:
    _StatAttack = None
    _HAS_STAT = False

try:
    from autoattack import AutoAttack as OfficialAutoAttack
    _HAS_OFFICIAL_AA = True
except ImportError:
    OfficialAutoAttack = None
    _HAS_OFFICIAL_AA = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    _HAS_SK = True
except ImportError:
    _HAS_SK = False


# ============= Data utilities =============
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class ImageTransform:
    def __init__(self, size=224, load_size: Optional[int] = None, crop_size: Optional[int] = None, resize_mode: str = 'short_side'):
        base = int(size)
        self.load_size = int(base if load_size is None else load_size)
        self.crop_size = int(base if crop_size is None else crop_size)
        self.resize_mode = str(resize_mode)
        
    def __call__(self, img):
        img = img.convert('RGB')
        if int(self.load_size) > 0:
            if self.resize_mode == 'square':
                img = img.resize((self.load_size, self.load_size), Image.BICUBIC)
            else:
                w, h = img.size
                if w < h:
                    new_w = self.load_size
                    new_h = int(self.load_size * h / max(1, w))
                else:
                    new_h = self.load_size
                    new_w = int(self.load_size * w / max(1, h))
                img = img.resize((new_w, new_h), Image.BICUBIC)

        img = _translate_duplicate(img, crop_size=self.crop_size)
        w2, h2 = img.size
        left = max(0, (w2 - self.crop_size) // 2)
        top = max(0, (h2 - self.crop_size) // 2)
        img = img.crop((left, top, left + self.crop_size, top + self.crop_size))

        img = np.array(img, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)


def _translate_duplicate(img: Image.Image, crop_size: int = 224) -> Image.Image:
    if min(img.size) < int(crop_size):
        width, height = img.size
        new_width = width * int(np.ceil(int(crop_size) / max(1, width)))
        new_height = height * int(np.ceil(int(crop_size) / max(1, height)))
        new_img = Image.new('RGB', (new_width, new_height))
        for i in range(0, new_width, width):
            for j in range(0, new_height, height):
                new_img.paste(img, (i, j))
        return new_img
    return img


class ForgeLensImageTransform:
    def __init__(self, size: int = 224):
        self.size = int(size)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert('RGB')
        img = _translate_duplicate(img, crop_size=self.size)
        w, h = img.size
        left = max(0, (w - self.size) // 2)
        top = max(0, (h - self.size) // 2)
        img = img.crop((left, top, left + self.size, top + self.size))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)


class CLIPImageTransform:
    def __init__(self, load_size: int = 256, crop_size: int = 224, resize_mode: str = 'short_side'):
        self.load_size = int(load_size)
        self.crop_size = int(crop_size)
        self.resize_mode = str(resize_mode)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert('RGB')
        if int(self.load_size) > 0:
            if self.resize_mode == 'square':
                img = img.resize((self.load_size, self.load_size), Image.BICUBIC)
            else:
                w, h = img.size
                if w < h:
                    new_w = self.load_size
                    new_h = int(self.load_size * h / max(1, w))
                else:
                    new_h = self.load_size
                    new_w = int(self.load_size * w / max(1, h))
                img = img.resize((new_w, new_h), Image.BICUBIC)
        img = _translate_duplicate(img, crop_size=self.crop_size)
        w2, h2 = img.size
        left = max(0, (w2 - self.crop_size) // 2)
        top = max(0, (h2 - self.crop_size) // 2)
        img = img.crop((left, top, left + self.crop_size, top + self.crop_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)


class ImageDataset(Dataset):
    def __init__(self, image_list, label_list, transform=None):
        self.images = image_list
        self.labels = label_list
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label


def _parse_boolish(value) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'y', 't'}:
        return True
    if text in {'0', 'false', 'no', 'n', 'f', ''}:
        return False
    try:
        return bool(int(float(text)))
    except Exception:
        return False


def _parse_intish(value, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return int(default)


def _load_external_attack_metadata_csv(csv_path: str) -> Dict[str, Dict[str, object]]:
    path = Path(csv_path)
    if not path.exists():
        raise ValueError(f"external_metadata does not exist: {path}")

    rows: Dict[str, Dict[str, object]] = {}
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            adv_name = Path(str(row.get('adv_image_name', '')).strip()).name
            if adv_name == '':
                continue

            clean_correct = _parse_boolish(row.get('clean_correct_gen', 0))
            adv_correct = _parse_boolish(row.get('adv_correct_gen', 0))
            attack_valid = _parse_boolish(row.get('attack_valid_gen', clean_correct))
            attack_success = _parse_boolish(row.get('attack_success_gen', clean_correct and (not adv_correct)))
            attack_failure = _parse_boolish(row.get('attack_failure_gen', clean_correct and adv_correct))
            attack_ignored = _parse_boolish(row.get('attack_ignored_gen', not clean_correct))
            rows[adv_name] = {
                'primary_label': _parse_intish(row.get('primary_label', row.get('label', 1)), default=1),
                'clean_correct_gen': clean_correct,
                'adv_correct_gen': adv_correct,
                'attack_valid_gen': attack_valid,
                'attack_success_gen': attack_success,
                'attack_failure_gen': attack_failure,
                'attack_ignored_gen': attack_ignored,
            }

    if len(rows) == 0:
        raise ValueError(f"No usable rows found in external_metadata={path}")
    return rows


def _label_counts_from_pairs(pairs: List[Tuple[Path, int]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for _, y in pairs:
        yy = int(y)
        counts[yy] = counts.get(yy, 0) + 1
    return counts


def _select_balanced_labeled_pairs(
    pairs: List[Tuple[Path, int]],
    sample_limit: int,
    seed: int,
    shuffle_paths: bool,
    subset_name: str,
) -> List[Tuple[Path, int]]:
    if len(pairs) == 0:
        return []

    real_pairs = [(p, int(y)) for (p, y) in pairs if int(y) == 0]
    fake_pairs = [(p, int(y)) for (p, y) in pairs if int(y) == 1]

    if len(real_pairs) == 0 or len(fake_pairs) == 0:
        take = len(pairs) if int(sample_limit) <= 0 else min(len(pairs), int(sample_limit))
        if int(sample_limit) > 0 and take < int(sample_limit):
            print(
                f"[external-attack][warn] {subset_name} requested {int(sample_limit)} samples, "
                f"but only {int(len(pairs))} labeled samples are available."
            )
        return list(pairs[:take])

    if bool(shuffle_paths):
        rng = random.Random(int(seed))
        rng.shuffle(real_pairs)
        rng.shuffle(fake_pairs)

    max_balanced = int(2 * min(len(real_pairs), len(fake_pairs)))
    requested_total = int(sample_limit)
    take_total = max_balanced if requested_total <= 0 else min(requested_total, max_balanced)
    if take_total <= 0:
        return []

    if take_total % 2 != 0:
        adjusted_total = take_total - 1
        if adjusted_total <= 0:
            raise ValueError(f"{subset_name} requires at least two labeled samples to build a balanced subset")
        print(
            f"[external-attack][warn] {subset_name} balanced selection requires an even sample count; "
            f"reducing {int(take_total)} -> {int(adjusted_total)}."
        )
        take_total = adjusted_total

    if requested_total > 0 and take_total < requested_total:
        print(
            f"[external-attack][warn] {subset_name} requested {int(requested_total)} samples, but balanced selection "
            f"can provide at most {int(take_total)} (real={int(len(real_pairs))}, fake={int(len(fake_pairs))})."
        )

    take_each = int(take_total // 2)
    selected: List[Tuple[Path, int]] = []
    for idx in range(take_each):
        selected.append(real_pairs[idx])
        selected.append(fake_pairs[idx])
    return selected


# ============= Model wrappers =============
class ResNetFeatureExtractor(nn.Module):
    """Extract intermediate features from ResNet"""
    def __init__(self, model_path=None, device='cuda', include_input=False):
        super().__init__()
        self.model_type = 'resnet'
        import torchvision.models as models
        self.model = models.resnet50(pretrained=True)
        # Use a single-logit binary head (Real/Fake) to match checkpoints trained for detection.
        # If a checkpoint is provided, it typically contains this head and will fail to load into
        # the default ImageNet 1000-way head.
        self.model.fc = nn.Linear(2048, 1)
        self.include_input = bool(include_input)
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Compatibility with checkpoints saved from wrappers / DataParallel / Sequential:
            # - DataParallel: "module." prefix
            # - nn.Sequential(normalize, backbone): keys like "0.*" (normalize) and "1.*" (backbone)
            cleaned_state = {}
            for k, v in state_dict.items():
                k = str(k)
                if k.startswith('module.'):
                    k = k[len('module.'):]
                if k.startswith('model.'):
                    k = k[len('model.'):]
                if k.startswith('backbone.'):
                    k = k[len('backbone.'):]

                # Drop normalize branch if present, keep backbone branch.
                if k.startswith('0.'):
                    continue
                if k.startswith('1.'):
                    k = k[2:]

                cleaned_state[k] = v

            fc_w = cleaned_state.get('fc.weight', None)
            if torch.is_tensor(fc_w) and fc_w.dim() == 2:
                out_features = int(fc_w.shape[0])
                in_features = int(fc_w.shape[1])
                if (
                    not isinstance(self.model.fc, nn.Linear)
                    or int(getattr(self.model.fc, 'in_features', -1)) != int(in_features)
                    or int(getattr(self.model.fc, 'out_features', -1)) != int(out_features)
                ):
                    self.model.fc = nn.Linear(in_features, out_features)

            load_result = self.model.load_state_dict(cleaned_state, strict=False)
            missing = getattr(load_result, 'missing_keys', [])
            unexpected = getattr(load_result, 'unexpected_keys', [])
            print(f"Loaded ResNet weights from {model_path}")
            print(f"ResNet load_state_dict(strict=False): missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
            if len(missing) > 0:
                print(f"First missing keys: {missing[:10]}")
            if len(unexpected) > 0:
                print(f"First unexpected keys: {unexpected[:10]}")
        elif model_path:
            print(f"Warning: model_path not found, skipping weight load: {model_path}")
        
        # Normalization for ImageNet
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.model.eval()
        self.to(device)
        
    def extract_layer_features(self, x):
        """Extract features from multiple layers"""
        features = {}
        x = self.normalize(x)

        if self.include_input:
            features['input'] = x
        
        # Conv1
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        features['conv1'] = x.clone()
        
        x = self.model.maxpool(x)
        
        # Layer1-4
        x = self.model.layer1(x)
        features['layer1'] = x.clone()
        
        x = self.model.layer2(x)
        features['layer2'] = x.clone()
        
        x = self.model.layer3(x)
        features['layer3'] = x.clone()
        
        x = self.model.layer4(x)
        features['layer4'] = x.clone()
        
        # Avgpool
        x = self.model.avgpool(x)
        features['avgpool'] = x.clone()
        
        # Flatten and FC
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        features['fc'] = x.clone()
        
        return features
    
    def forward(self, x):
        x = self.normalize(x)
        return self.model(x)


class GramFeatureExtractor(nn.Module):
    def __init__(self, model_path=None, device='cuda', include_input=False):
        super().__init__()
        self.model_type = 'gram'
        self.include_input = bool(include_input)

        num_classes = 1
        cleaned_state = None

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            cleaned_state = {}
            for k, v in state_dict.items():
                k = str(k)
                if k.startswith('module.'):
                    k = k[len('module.'):]
                if k.startswith('model.'):
                    k = k[len('model.'):]
                if k.startswith('backbone.'):
                    k = k[len('backbone.'):]

                # Drop normalize branch if present, keep backbone branch.
                if k.startswith('0.'):
                    continue
                if k.startswith('1.'):
                    k = k[2:]

                cleaned_state[k] = v

            # Infer num_classes from checkpoint's final FC layer shape.
            fc_key = 'fcnewr.3.weight'
            if fc_key in cleaned_state:
                num_classes = int(cleaned_state[fc_key].shape[0])
                print(f"Gram-Net: inferred num_classes={num_classes} from checkpoint {fc_key} shape")
            else:
                fc_candidates = [
                    k for k, v in cleaned_state.items()
                    if 'fcnewr' in k and str(k).endswith('.weight') and torch.is_tensor(v) and v.dim() == 2
                ]
                if len(fc_candidates) > 0:
                    pick = fc_candidates[0]
                    num_classes = int(cleaned_state[pick].shape[0])
                    print(f"Gram-Net: inferred num_classes={num_classes} from checkpoint {pick} shape")

        self.model = ResnetGram.resnet18(num_classes=num_classes)

        if model_path and os.path.exists(model_path):
            try:
                load_result = self.model.load_state_dict(cleaned_state, strict=False)
            except RuntimeError as e:
                if 'fcnewr' in str(e) and cleaned_state is not None:
                    retry_classes = num_classes
                    if retry_classes != int(self.model.fcnewr[-1].out_features):
                        self.model = ResnetGram.resnet18(num_classes=retry_classes)
                    load_result = self.model.load_state_dict(cleaned_state, strict=False)
                else:
                    raise
            missing = getattr(load_result, 'missing_keys', [])
            unexpected = getattr(load_result, 'unexpected_keys', [])
            print(f"Loaded Gram-Net weights from {model_path}")
            print(f"Gram-Net load_state_dict(strict=False): missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
            if len(missing) > 0:
                print(f"First missing keys: {missing[:10]}")
            if len(unexpected) > 0:
                print(f"First unexpected keys: {unexpected[:10]}")
        elif model_path:
            print(f"Warning: model_path not found, skipping weight load: {model_path}")

        self.model.eval()
        self.to(device)

    def extract_layer_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: Dict[str, torch.Tensor] = {}

        if self.include_input:
            feats['input'] = x

        m = self.model

        x3 = x
        t = m.conv1(x3)
        t = m.bn1(t)
        x4 = m.relu(t)
        x5 = m.maxpool(x4)
        x6 = m.layer1(x5)
        x7 = m.layer2(x6)
        x8 = m.layer3(x7)
        x9 = m.layer4(x8)

        feats['x8'] = x8.clone()

        x_pooled = m.avgpool(x9)
        x_vec = x_pooled.view(x_pooled.size(0), -1)
        feats['x'] = x_vec.clone()

        g2 = m.conv_inter2_0(x6)
        g2 = m.gram(g2)
        g2 = m.g2_fc1(g2)
        g2 = m.g2_fc2(g2)
        g2 = m.avgpool(g2)
        g2 = g2.view(g2.size(0), -1)

        g3 = m.conv_inter3_0(x7)
        g3 = m.gram(g3)
        g3 = m.g3_fc1(g3)
        g3 = m.g3_fc2(g3)
        g3 = m.avgpool(g3)
        g3 = g3.view(g3.size(0), -1)

        g4 = m.conv_inter4_0(x8)
        g4 = m.gram(g4)
        g4 = m.g4_fc1(g4)
        g4 = m.g4_fc2(g4)
        g4 = m.avgpool(g4)
        g4 = g4.view(g4.size(0), -1)

        g2 = m.scale(g2)
        g3 = m.scale(g3)
        g4 = m.scale(g4)

        feats['g2'] = g2.clone()
        feats['g3'] = g3.clone()
        feats['g4'] = g4.clone()
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _import_csf_modules():
    csf_root = Path(__file__).parent / 'Frequency-Collaborative-main'
    if csf_root.exists() and str(csf_root) not in sys.path:
        sys.path.insert(0, str(csf_root))
    from CSF import FCACoAttentionNet
    from Data_Augmentation import ConditionalPadding, Extract_texture
    return FCACoAttentionNet, ConditionalPadding, Extract_texture


class CSFImageTransform:
    def __init__(self, size: int = 256, patch_size: int = 32):
        _, ConditionalPadding, Extract_texture = _import_csf_modules()
        import torchvision.transforms as tvt
        self.pad = ConditionalPadding(int(size))
        self.extract = Extract_texture(patch_size=int(patch_size))
        self.to_tensor = tvt.ToTensor()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert('RGB')
        img = self.pad(img)
        img = self.extract(img)
        return self.to_tensor(img)


class CSFFeatureExtractor(nn.Module):
    def __init__(self, model_path=None, device='cuda', include_input=False):
        super().__init__()
        self.model_type = 'csf'
        self.include_input = bool(include_input)
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        FCACoAttentionNet, _, _ = _import_csf_modules()
        self.model = FCACoAttentionNet(num_classes=2)

        model_path = None if model_path is None else str(model_path).strip()
        if model_path == '':
            model_path = None
        if model_path is not None and not os.path.exists(model_path):
            raise ValueError(f"--model_path not found: {model_path}")
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                for key in ['state_dict', 'model_state_dict', 'model', 'net', 'network']:
                    if key in checkpoint and isinstance(checkpoint[key], dict):
                        checkpoint = checkpoint[key]
                        break

            if isinstance(checkpoint, dict) and any(str(k).startswith('module.') for k in checkpoint.keys()):
                checkpoint = {str(k).replace('module.', '', 1): v for k, v in checkpoint.items()}

            load_result = self.model.load_state_dict(checkpoint, strict=False)
            missing = getattr(load_result, 'missing_keys', [])
            unexpected = getattr(load_result, 'unexpected_keys', [])
            print(f"Loaded CSF weights from {model_path} (missing_keys={len(missing)}, unexpected_keys={len(unexpected)})")

        self.model.eval()
        self.to(device)

    def extract_layer_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: Dict[str, torch.Tensor] = {}

        x = self.normalize(x)
        if self.include_input:
            feats['input'] = x

        m = self.model
        x = m.ExtractNet(x)
        feats['extract_out'] = x.clone()

        h = m.head
        x = h.layer1(x)
        feats['head_layer1'] = x.clone()
        x = h.layer2(x)
        feats['head_layer2'] = x.clone()
        x = h.layer3(x)
        feats['head_layer3'] = x.clone()
        x = h.layer4(x)
        feats['head_layer4'] = x.clone()
        x = h.avg_pool1(x)
        feats['avg_pool1'] = x.clone()

        x = h.layer5(x)
        feats['head_layer5'] = x.clone()
        x = h.layer6(x)
        feats['head_layer6'] = x.clone()
        x = h.avg_pool2(x)
        feats['avg_pool2'] = x.clone()

        x = h.layer7(x)
        feats['head_layer7'] = x.clone()
        x = h.layer8(x)
        feats['head_layer8'] = x.clone()
        x = h.avg_pool3(x)
        feats['avg_pool3'] = x.clone()

        x = h.layer9(x)
        feats['head_layer9'] = x.clone()
        x = h.layer10(x)
        feats['head_layer10'] = x.clone()

        x = h.adaptive_avg_pool(x)
        x = h.flatten(x)
        feats['flatten'] = x.clone()
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        return self.model(x)


class CLIPFeatureExtractor(nn.Module):
    """Extract intermediate features from CLIP ViT"""
    def __init__(
        self,
        model_name='ViT-L/14',
        model_path=None,
        fc_weights_path=None,
        device='cuda',
        feat_mode='flatten',
        include_input=False,
        apply_input_normalize: bool = True,
    ):
        super().__init__()
        self.model_type = 'clip'
        self.feat_mode = str(feat_mode)
        self.include_input = bool(include_input)
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        self.normalize = Normalize(mean, std)
        self.apply_input_normalize = bool(apply_input_normalize)
        
        # Load CLIP model
        try:
            from networks.univfd_models.clip import clip as univfd_clip
            self.clip_model, _ = univfd_clip.load(model_name, device='cpu')
        except Exception:
            import clip
            self.clip_model, _ = clip.load(model_name, device='cpu')

        model_path = None if model_path is None else str(model_path).strip()
        if model_path == '':
            model_path = None

        if model_path is not None and not os.path.exists(model_path):
            raise ValueError(f"--model_path not found: {model_path}")

        if model_path is not None:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            if isinstance(state_dict, dict) and any(str(k).startswith('module.') for k in state_dict.keys()):
                state_dict = {str(k).replace('module.', '', 1): v for k, v in state_dict.items()}

            clip_state = {}
            head_state = {}
            for k, v in (state_dict.items() if isinstance(state_dict, dict) else []):
                ks = str(k)
                if ks.startswith('model.'):
                    clip_state[ks[len('model.'):]] = v
                    continue
                if ks.startswith('fc.'):
                    head_state[ks] = v
                    continue
                clip_state[ks] = v

            fc_state = None
            if 'fc.weight' in state_dict and 'fc.bias' in state_dict:
                w = state_dict['fc.weight']
                b = state_dict['fc.bias']
                if torch.is_tensor(w) and torch.is_tensor(b) and w.dim() == 2 and b.dim() == 1:
                    fc_state = {'weight': w, 'bias': b}
            if fc_state is None and 'fc.weight' in head_state and 'fc.bias' in head_state:
                w = head_state['fc.weight']
                b = head_state['fc.bias']
                if torch.is_tensor(w) and torch.is_tensor(b) and w.dim() == 2 and b.dim() == 1:
                    fc_state = {'weight': w, 'bias': b}
            if fc_state is None:
                for prefix in ['fc', 'classifier', 'head']:
                    w_key = prefix + '.weight'
                    b_key = prefix + '.bias'
                    if w_key in state_dict and b_key in state_dict:
                        w = state_dict[w_key]
                        b = state_dict[b_key]
                        if torch.is_tensor(w) and torch.is_tensor(b) and w.dim() == 2 and b.dim() == 1:
                            fc_state = {'weight': w, 'bias': b}
                            break

            if fc_state is None:
                raise RuntimeError(f"Could not find fc weights in checkpoint: {model_path}")

            in_features = int(fc_state['weight'].shape[1])
            out_features = int(fc_state['weight'].shape[0])
            self.fc = nn.Linear(in_features, out_features)
            self.fc.load_state_dict(fc_state, strict=True)

            clip_state_clean = {}
            for k, v in clip_state.items():
                ks = str(k)
                if ks.startswith('fc.') or ks.startswith('classifier.') or ks.startswith('head.'):
                    continue
                clip_state_clean[ks] = v

            load_result = self.clip_model.load_state_dict(clip_state_clean, strict=False)
            missing = getattr(load_result, 'missing_keys', [])
            unexpected = getattr(load_result, 'unexpected_keys', [])
            print(f"Loaded UnivFD model from {model_path} (missing_keys={len(missing)}, unexpected_keys={len(unexpected)})")
        else:
            self.fc = None
        
        # Load FC weights if provided (only used when --model_path is not provided)
        fc_weights_path = None if fc_weights_path is None else str(fc_weights_path).strip()
        if fc_weights_path == '':
            fc_weights_path = None

        if model_path is None and fc_weights_path is not None and not os.path.exists(fc_weights_path):
            raise ValueError(f"--fc_weights path not found: {fc_weights_path}")

        if model_path is None and fc_weights_path is not None:
            checkpoint = torch.load(fc_weights_path, map_location='cpu')
            if isinstance(checkpoint, nn.Module):
                checkpoint = checkpoint.state_dict()

            if isinstance(checkpoint, dict):
                for key in ['state_dict', 'model_state_dict', 'model', 'net', 'network']:
                    if key in checkpoint:
                        inner = checkpoint[key]
                        if isinstance(inner, nn.Module):
                            inner = inner.state_dict()
                        if isinstance(inner, dict):
                            checkpoint = inner
                            break

            if isinstance(checkpoint, dict) and any(str(k).startswith('module.') for k in checkpoint.keys()):
                checkpoint = {str(k).replace('module.', '', 1): v for k, v in checkpoint.items()}

            fc_state = None
            if isinstance(checkpoint, dict):
                if (
                    'weight' in checkpoint
                    and 'bias' in checkpoint
                    and torch.is_tensor(checkpoint['weight'])
                    and checkpoint['weight'].dim() == 2
                ):
                    fc_state = {'weight': checkpoint['weight'], 'bias': checkpoint['bias']}
                else:
                    head_names = {'fc', 'classifier', 'head'}
                    candidates = []
                    for k in checkpoint.keys():
                        ks = str(k)
                        if not ks.endswith('.weight'):
                            continue
                        parts = ks.split('.')
                        if len(parts) < 2:
                            continue
                        if parts[-2] not in head_names:
                            continue
                        prefix = '.'.join(parts[:-1])
                        bias_key = prefix + '.bias'
                        if bias_key in checkpoint:
                            w = checkpoint[ks]
                            b = checkpoint[bias_key]
                            if torch.is_tensor(w) and torch.is_tensor(b) and w.dim() == 2 and b.dim() == 1:
                                candidates.append(prefix)

                    if len(candidates) > 0:
                        feat_dim = {'RN50': 1024, 'ViT-L/14': 768, 'ViT-B/32': 512, 'ViT-B/16': 512}.get(model_name, 768)
                        chosen = None
                        for prefix in candidates:
                            w = checkpoint[prefix + '.weight']
                            if int(w.shape[1]) == int(feat_dim):
                                chosen = prefix
                                break
                        if chosen is None:
                            chosen = candidates[0]

                        fc_state = {
                            'weight': checkpoint[chosen + '.weight'],
                            'bias': checkpoint[chosen + '.bias'],
                        }

            if fc_state is None:
                raise RuntimeError(
                    f"Could not find a linear head (fc/classifier/head) in --fc_weights checkpoint: {fc_weights_path}"
                )

            clip_state = {}
            if isinstance(checkpoint, dict):
                for k, v in checkpoint.items():
                    ks = str(k)
                    if ks.startswith('fc.') or ks.startswith('classifier.') or ks.startswith('head.'):
                        continue
                    if ks.startswith('model.'):
                        clip_state[ks[len('model.'):]] = v
                    elif ks.startswith('clip_model.'):
                        clip_state[ks[len('clip_model.'):]] = v
                    else:
                        clip_state[ks] = v

            if isinstance(clip_state, dict) and any(s in clip_state for s in ['visual.conv1.weight', 'transformer.resblocks.0.attn.in_proj_weight']):
                load_result = self.clip_model.load_state_dict(clip_state, strict=False)
                missing = getattr(load_result, 'missing_keys', [])
                unexpected = getattr(load_result, 'unexpected_keys', [])
                print(
                    f"Loaded UnivFD CLIP backbone from {fc_weights_path} (missing_keys={len(missing)}, unexpected_keys={len(unexpected)})"
                )

            in_features = int(fc_state['weight'].shape[1])
            out_features = int(fc_state['weight'].shape[0])
            self.fc = nn.Linear(in_features, out_features)
            self.fc.load_state_dict(fc_state, strict=True)
            print(f"Loaded UnivFD FC head from {fc_weights_path} (in_features={in_features}, out_features={out_features})")
        elif model_path is None:
            self.fc = None
            print("Warning: --fc_weights is empty; running CLIP without FC head (embeddings only).")
            
        self.clip_model.eval()
        self.to(device)

    def get_num_layers(self) -> int:
        try:
            return int(len(self.clip_model.visual.transformer.resblocks))
        except Exception:
            try:
                return int(len(list(self.clip_model.visual.transformer.resblocks.children())))
            except Exception:
                return 0
        
    def extract_layer_features(self, x):
        """Extract features from all transformer layers"""
        features = {}
        handles = []
        outputs = {}
        
        def make_hook(layer_idx):
            def hook_fn(module, inp, out):
                outputs[layer_idx] = out
            return hook_fn
        
        # Register hooks for all transformer blocks
        for idx, block in enumerate(self.clip_model.visual.transformer.resblocks):
            h = block.register_forward_hook(make_hook(idx))
            handles.append(h)
        
        # Forward pass
        if bool(self.apply_input_normalize):
            x = self.normalize(x)
        if self.include_input:
            features['input'] = x
        _ = self.clip_model.visual(x)
        
        # Remove hooks
        for h in handles:
            h.remove()
        
        # Process outputs
        for idx, out in outputs.items():
            if isinstance(out, tuple):
                out = out[0]
            if out.dim() == 3:
                # Normalize to [B, T, D]
                if out.shape[0] == x.shape[0]:
                    toks = out
                elif out.shape[1] == x.shape[0]:
                    toks = out.permute(1, 0, 2)
                else:
                    toks = out

                if self.feat_mode == 'cls':
                    features[f'layer{idx}'] = toks[:, 0, :]
                elif self.feat_mode in {'flatten', 'flatten_tokens'}:
                    features[f'layer{idx}'] = toks.reshape(toks.shape[0], -1)
                else:  # mean_patch
                    # Average pool patch tokens (excluding CLS)
                    patch = toks[:, 1:, :] if toks.shape[1] > 1 else toks
                    features[f'layer{idx}'] = patch.mean(dim=1)
            else:
                features[f'layer{idx}'] = out
                
        return features
    
    def forward(self, x):
        if bool(self.apply_input_normalize):
            x = self.normalize(x)
        feat = self.clip_model.encode_image(x).float()
        if self.fc is not None:
            return self.fc(feat)
        return feat


class ForgeLensFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        include_input: bool = False,
        stage: int = 2,
        feature_set: str = 'faformer',
        wsgm_count: int = 4,
        wsgm_reduction_factor: int = 4,
        faformer_layers: int = 2,
        faformer_head: int = 2,
        faformer_reduction_factor: int = 1,
        clip_layers: int = 24,
    ):
        super().__init__()
        self.model_type = 'forgelens'
        self.include_input = bool(include_input)
        self.feature_set = str(feature_set)
        self.num_layers = int(faformer_layers)
        self.num_clip_layers = int(clip_layers)

        wsgm_count = int(wsgm_count)

        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        self.normalize = Normalize(mean, std)

        model_path = str(model_path).strip()
        if model_path == '' or not os.path.exists(model_path):
            raise ValueError(f"Invalid --model_path for ForgeLens: {model_path}")

        ckpt = torch.load(model_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt
        if isinstance(sd, dict) and any(str(k).startswith('module.') for k in sd.keys()):
            sd = {str(k).replace('module.', '', 1): v for k, v in sd.items()}

        if int(stage) == 2 and isinstance(sd, dict):
            faformer_idxs = []
            for k in sd.keys():
                m = re.search(r'^transformer\.resblocks\.(\d+)\.', str(k))
                if m is not None:
                    faformer_idxs.append(int(m.group(1)))
            if len(faformer_idxs) > 0:
                inferred = max(faformer_idxs) + 1
                self.num_layers = int(inferred)

            wsgm_idxs = []
            for k in sd.keys():
                m = re.search(r'WSGM_modules\.(\d+)\.', str(k))
                if m is not None:
                    wsgm_idxs.append(int(m.group(1)))
            if len(wsgm_idxs) > 0:
                inferred_wsgm = max(wsgm_idxs) + 1
                if int(inferred_wsgm) != int(wsgm_count):
                    print(
                        f"[forgelens][info] inferred WSGM_count={int(inferred_wsgm)} from checkpoint; "
                        f"overriding provided wsgm_count={int(wsgm_count)}"
                    )
                    wsgm_count = int(inferred_wsgm)

            clip_idxs = []
            for k in sd.keys():
                ks = str(k)
                m = re.search(r'backbone\.(?:backbone\.)?visual\.transformer\.resblocks\.(\d+)\.', ks)
                if m is not None:
                    clip_idxs.append(int(m.group(1)))
            if len(clip_idxs) > 0:
                self.num_clip_layers = int(max(clip_idxs) + 1)

        forge_root = Path(__file__).parent / 'ForgeLens-main'
        if not forge_root.exists():
            raise FileNotFoundError(f"ForgeLens-main not found at: {forge_root}")

        old_argv = list(sys.argv)
        sys.path.insert(0, str(forge_root))
        sys.argv = [
            'forgelens',
            '--WSGM_count', str(int(wsgm_count)),
            '--WSGM_reduction_factor', str(int(wsgm_reduction_factor)),
            '--FAFormer_layers', str(int(self.num_layers)),
            '--FAFormer_reduction_factor', str(int(faformer_reduction_factor)),
            '--FAFormer_head', str(int(faformer_head)),
            '--eval_stage', str(int(stage)),
            '--weights', str(model_path),
        ]
        try:
            if int(stage) == 1:
                from models.network.net_stage1 import net_stage1
                self.model = net_stage1()
            else:
                from options.options import Options
                opt = Options().parse()
                from models.network.net_stage2 import net_stage2
                self.model = net_stage2(opt, train=False)
        finally:
            sys.argv = old_argv

        try:
            self.model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            if int(stage) != 2 or not isinstance(sd, dict):
                raise

            faformer_idxs = []
            for k in sd.keys():
                m = re.search(r'^transformer\.resblocks\.(\d+)\.', str(k))
                if m is not None:
                    faformer_idxs.append(int(m.group(1)))
            inferred = (max(faformer_idxs) + 1) if len(faformer_idxs) > 0 else 0
            if inferred > 0 and int(inferred) != int(self.num_layers):
                self.num_layers = int(inferred)
                old_argv = list(sys.argv)
                sys.argv = [
                    'forgelens',
                    '--WSGM_count', str(int(wsgm_count)),
                    '--WSGM_reduction_factor', str(int(wsgm_reduction_factor)),
                    '--FAFormer_layers', str(int(self.num_layers)),
                    '--FAFormer_reduction_factor', str(int(faformer_reduction_factor)),
                    '--FAFormer_head', str(int(faformer_head)),
                    '--eval_stage', str(int(stage)),
                    '--weights', str(model_path),
                ]
                try:
                    from options.options import Options
                    opt = Options().parse()
                    from models.network.net_stage2 import net_stage2
                    self.model = net_stage2(opt, train=False)
                finally:
                    sys.argv = old_argv
                self.model.load_state_dict(sd, strict=True)
            else:
                raise e
        self.model = self.model.float()
        self.model.eval()
        self.to(device)

    def _safe_attention_context(self):
        if not torch.cuda.is_available():
            return contextlib.nullcontext()
        cuda_backends = getattr(torch.backends, 'cuda', None)
        sdp_kernel = getattr(cuda_backends, 'sdp_kernel', None) if cuda_backends is not None else None
        if sdp_kernel is None:
            return contextlib.nullcontext()
        try:
            return sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except TypeError:
            return contextlib.nullcontext()

    def _get_clip_visual(self):
        m = self.model
        if hasattr(m, 'backbone') and hasattr(m.backbone, 'visual'):
            return m.backbone.visual
        if hasattr(m, 'backbone') and hasattr(m.backbone, 'backbone') and hasattr(m.backbone.backbone, 'visual'):
            return m.backbone.backbone.visual
        return None

    def _get_clip_model(self):
        m = self.model
        if hasattr(m, 'backbone') and hasattr(m.backbone, 'encode_image'):
            return m.backbone
        if hasattr(m, 'backbone') and hasattr(m.backbone, 'backbone') and hasattr(m.backbone.backbone, 'encode_image'):
            return m.backbone.backbone
        return None

    def _clip_layer_tokens(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        clip_model = self._get_clip_model()
        visual = self._get_clip_visual()
        if clip_model is None or visual is None:
            return {}

        resblocks = getattr(getattr(getattr(visual, 'transformer', None), 'resblocks', None), 'children', None)
        if resblocks is None:
            return {}

        outputs: Dict[int, torch.Tensor] = {}
        handles = []
        try:
            for idx, block in enumerate(list(visual.transformer.resblocks.children())):
                def _make_hook(i: int):
                    def _hook(_m, _inp, out):
                        try:
                            if isinstance(out, (tuple, list)):
                                out_t = out[0]
                            else:
                                out_t = out
                            if torch.is_tensor(out_t) and out_t.dim() == 3:
                                outputs[i] = out_t.detach()
                        except Exception:
                            pass
                    return _hook
                handles.append(block.register_forward_hook(_make_hook(idx)))

            _ = clip_model.encode_image(x)

            out_feats: Dict[str, torch.Tensor] = {}
            n = int(self.num_clip_layers)
            if len(outputs) > 0:
                n = max(outputs.keys()) + 1
            for i in range(int(n)):
                out_t = outputs.get(i, None)
                if out_t is None:
                    continue
                cls_tok = out_t[0]
                try:
                    cls_ln = visual.ln_post(cls_tok)
                except Exception:
                    cls_ln = cls_tok
                out_feats[f'clipu_layer{i}'] = cls_ln
                try:
                    proj = getattr(visual, 'proj', None)
                    if proj is not None:
                        out_feats[f'clip_layer{i}'] = cls_ln @ proj
                except Exception:
                    pass
            return out_feats
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

    def extract_layer_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: Dict[str, torch.Tensor] = {}
        x = self.normalize(x)
        if self.include_input:
            feats['input'] = x

        clip_feats = self._clip_layer_tokens(x)
        if self.feature_set in ['clip_proj', 'clip_both', 'all_proj', 'all_both']:
            for k, v in clip_feats.items():
                if k.startswith('clip_layer'):
                    feats[k] = v
        if self.feature_set in ['clip_unproj', 'clip_both', 'all_unproj', 'all_both']:
            for k, v in clip_feats.items():
                if k.startswith('clipu_layer'):
                    feats[k] = v

        if self.feature_set in ['faformer', 'all_proj', 'all_unproj', 'all_both'] and hasattr(self.model, 'transformer'):
            def _parse_cls_tokens(obj):
                if obj is None:
                    return None
                if isinstance(obj, dict):
                    for kk in ['cls_tokens', 'cls', 'tokens']:
                        if kk in obj:
                            return obj[kk]
                    return None
                if isinstance(obj, (tuple, list)):
                    # Heuristic: cls_tokens in ForgeLens is typically a list/tuple of tensors
                    for item in obj:
                        if isinstance(item, (tuple, list)) and len(item) > 0 and all(torch.is_tensor(t) for t in item):
                            return item
                    # Next: a 3D tensor [B, N, D]
                    for item in obj:
                        if torch.is_tensor(item) and item.dim() == 3:
                            return item
                    # Fallback: last element
                    return obj[-1] if len(obj) > 0 else None
                return obj

            def _get_backbone_cls_tokens(x_in: torch.Tensor):
                # Prefer stage1 wrapper if present; otherwise fallback to raw CLIP encode_image
                if hasattr(self.model, 'backbone') and callable(self.model.backbone):
                    out = self.model.backbone(x_in)
                    return _parse_cls_tokens(out)
                clip_m = self._get_clip_model()
                if clip_m is not None and hasattr(clip_m, 'encode_image'):
                    out = clip_m.encode_image(x_in)
                    return _parse_cls_tokens(out)
                return None

            B = int(x.shape[0])
            with torch.no_grad():
                cls_tokens = _get_backbone_cls_tokens(x)
                cls_tokens_t = None
                if isinstance(cls_tokens, (tuple, list)) and len(cls_tokens) > 0:
                    cls_tokens_t = torch.stack(list(cls_tokens), dim=1)
                elif torch.is_tensor(cls_tokens):
                    if cls_tokens.dim() == 3:
                        cls_tokens_t = cls_tokens
                    elif cls_tokens.dim() == 2:
                        cls_tokens_t = cls_tokens.unsqueeze(1)

                if torch.is_tensor(cls_tokens_t) and cls_tokens_t.numel() > 0:
                    cls = self.model.cls_token.view(1, 1, -1).repeat(B, 1, 1)
                    x_cat = torch.cat([cls, cls_tokens_t], dim=1)
                    x_lnd = x_cat.permute(1, 0, 2)
                    out, _ = self.model.transformer(x_lnd)
                    for i in range(int(self.num_layers)):
                        k = f'layer{i}'
                        if k in out:
                            feats[k] = out[k]

        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        x = x.float()
        with self._safe_attention_context():
            out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


class StatAttackModelWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, model_type: str):
        super().__init__()
        self.base_model = base_model
        self.model_type = str(model_type)
        self.features: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type == 'resnet':
            m = self.base_model.model
            x = self.base_model.normalize(x)
            x = m.conv1(x)
            x = m.bn1(x)
            x = m.relu(x)
            x = m.maxpool(x)
            x = m.layer1(x)
            x = m.layer2(x)
            x = m.layer3(x)
            x = m.layer4(x)
            gp = m.avgpool(x)
            self.features['global_pool'] = gp
            out = torch.flatten(gp, 1)
            out = m.fc(out)
            return out

        if self.model_type == 'clip':
            x = self.base_model.normalize(x)
            feat = self.base_model.clip_model.encode_image(x).float()
            self.features['global_pool'] = feat
            if getattr(self.base_model, 'fc', None) is not None:
                return self.base_model.fc(feat)
            return feat

        if self.model_type == 'gram':
            m = self.base_model.model
            # GramFeatureExtractor does not apply Normalize; keep consistent.
            x3 = x
            t = m.conv1(x3)
            t = m.bn1(t)
            x4 = m.relu(t)
            x5 = m.maxpool(x4)
            x6 = m.layer1(x5)
            x7 = m.layer2(x6)
            x8 = m.layer3(x7)
            x9 = m.layer4(x8)
            gp = m.avgpool(x9)
            self.features['global_pool'] = gp
            # Return a differentiable tensor; StatAttack only requires global_pool.
            return gp.flatten(1)

        if self.model_type == 'forgelens':
            x = self.base_model.normalize(x)
            clip_m = self.base_model._get_clip_model()
            if clip_m is None or not hasattr(clip_m, 'encode_image'):
                raise RuntimeError('ForgeLens model does not expose a CLIP encode_image for StatAttack')
            feat = clip_m.encode_image(x)
            self.features['global_pool'] = feat
            out = self.base_model.model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out

        raise ValueError(f"Unknown model_type for StatAttackModelWrapper: {self.model_type}")


class AAWrapper(nn.Module):
    """Wrapper for adversarial attacks"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        logits = self.model(x)
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)
        # If the model already outputs multi-class logits, keep as-is.
        if logits.dim() == 2 and logits.shape[1] > 1:
            return logits
        # For binary classification with a single logit, expand to 2-class logits.
        logit_neg = torch.zeros_like(logits)
        return torch.cat([logit_neg, logits], dim=1)


def _to_single_logit_tensor(out: torch.Tensor) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.dim() == 1:
        return out.view(-1, 1)
    if out.dim() == 2 and out.shape[1] == 1:
        return out
    if out.dim() == 2 and out.shape[1] == 2:
        return (out[:, 1:2] - out[:, 0:1])
    raise ValueError(f"Unsupported logits shape (expected [B], [B,1], or [B,2]): {tuple(out.shape)}")


def _numpy_logits_to_z(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim == 1:
        return logits.reshape(-1, 1).astype(np.float32, copy=False)
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits.astype(np.float32, copy=False)
    if logits.ndim == 2 and logits.shape[1] == 2:
        return (logits[:, 1:2] - logits[:, 0:1]).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported numpy logits shape for z: {tuple(logits.shape)}")


def _load_trained_logit_calibrator(ckpt_path: str, lid_dim: int, device: torch.device):
    ckpt_path = str(ckpt_path).strip()
    if ckpt_path == '':
        raise ValueError("Empty --calibrator_ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"calibrator_ckpt not found: {ckpt_path}")

    from train_logit_calibrator import _build_model_from_config, _load_checkpoint

    cfg, sd = _load_checkpoint(Path(ckpt_path), map_location='cpu')
    if not isinstance(cfg, dict) or len(cfg) == 0:
        raise ValueError(
            "Calibrator checkpoint is missing config. "
            "Please save it via train_logit_calibrator.py --save_model_path (checkpoint with config + state_dict)."
        )

    cfg = dict(cfg)
    ckpt_lid_dim = cfg.get('lid_dim', None)
    if ckpt_lid_dim is not None:
        try:
            ckpt_lid_dim_i = int(ckpt_lid_dim)
        except Exception:
            ckpt_lid_dim_i = None
        if ckpt_lid_dim_i is not None and int(ckpt_lid_dim_i) != int(lid_dim):
            raise ValueError(
                "Calibrator checkpoint lid_dim mismatch. "
                f"ckpt lid_dim={int(ckpt_lid_dim_i)} but extracted lid_dim={int(lid_dim)}. "
                "This usually means you changed feature extraction settings (model_type/clip_model/layers/include_input). "
                "Please use a calibrator trained with the same LID feature dimension, or pass --layers to match the checkpoint."
            )
    else:
        cfg['lid_dim'] = int(lid_dim)
    model = _build_model_from_config(cfg)
    model.load_state_dict(sd, strict=True)
    model = model.to(device)
    model.eval()
    return model, cfg


class CalibratedAAWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        calibrator: nn.Module,
        lid: torch.Tensor,
        use_logits_input: bool,
        layers: List[str],
        ref_feats_by_layer: Dict[str, torch.Tensor],
        k: int,
        distance_metric: str,
        feat_norm: str,
        spatial_feat_mode: str,
        adaptive_lid_recompute_every: int = 0,
    ):
        super().__init__()
        self.base_model = base_model
        self.calibrator = calibrator
        self.register_buffer('lid', lid)
        self.use_logits_input = bool(use_logits_input)
        self.layers = list(layers)
        self.ref_feats_by_layer = dict(ref_feats_by_layer)
        self.k = int(k)
        self.distance_metric = str(distance_metric)
        self.feat_norm = str(feat_norm)
        self.spatial_feat_mode = str(spatial_feat_mode)
        self._last_x_full: Optional[torch.Tensor] = None
        self._active_idx: Optional[torch.Tensor] = None
        self._warned_mismatch: bool = False
        self._ref_feats_device: Dict[str, torch.Tensor] = {}
        self._ref_feats_device_id: Optional[Tuple[str, int]] = None
        self.adaptive_lid_recompute_every = int(adaptive_lid_recompute_every)
        self._fwd_calls: int = 0
        self._subbatch_cursor: int = 0
        self._subbatch_full: int = int(lid.shape[0]) if torch.is_tensor(lid) else 0
        self._recompute_lid_warned_fallback: bool = False
        self._recompute_lid_warned_nonfinite: bool = False
        self._recompute_lid_last_fallback: bool = False
        self._recompute_lid_last_nonfinite: bool = False

    def _base_logits(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]

        if torch.is_tensor(out) and out.dim() == 2 and int(out.shape[1]) > 2:
            head = None
            for name in ['fc', 'classifier', 'head']:
                if hasattr(self.base_model, name):
                    cand = getattr(self.base_model, name)
                    if isinstance(cand, nn.Module):
                        head = cand
                        break
            if head is not None:
                try:
                    projected = head(out)
                    if torch.is_tensor(projected) and projected.dim() == 2 and int(projected.shape[0]) == int(out.shape[0]):
                        return projected
                except Exception:
                    pass
        return out

    def _get_ref_feat_on_device(self, layer: str, device: torch.device) -> Optional[torch.Tensor]:
        ref_feat = self.ref_feats_by_layer.get(layer, None)
        if ref_feat is None:
            return None

        if not torch.is_tensor(ref_feat):
            ref_feat = torch.as_tensor(ref_feat)

        dev_id = (str(device.type), int(device.index) if device.index is not None else -1)
        if self._ref_feats_device_id != dev_id:
            self._ref_feats_device = {}
            self._ref_feats_device_id = dev_id

        cached = self._ref_feats_device.get(layer, None)
        if cached is not None:
            return cached

        cached = ref_feat.detach().to(device=device, non_blocking=True)
        if cached.dtype != torch.float32:
            cached = cached.float()
        cached = cached.contiguous()
        self._ref_feats_device[layer] = cached
        return cached

    def _agg(self, feat: torch.Tensor) -> torch.Tensor:
        if not isinstance(feat, torch.Tensor):
            feat = torch.as_tensor(feat)
        if feat.dim() > 2:
            if str(self.spatial_feat_mode) == 'avgpool':
                feat = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)
            else:
                feat = feat.reshape(feat.shape[0], -1)
        else:
            feat = feat.reshape(feat.shape[0], -1)
        return feat

    def _recompute_lid(self, x: torch.Tensor) -> torch.Tensor:
        self._recompute_lid_last_fallback = False
        self._recompute_lid_last_nonfinite = False
        try:
            with torch.no_grad():
                layer_feats = self.base_model.extract_layer_features(x)

            lids_out = []
            for layer in self.layers:
                if layer not in layer_feats:
                    lids_out.append(torch.full((int(x.shape[0]),), float('nan'), device=x.device))
                    continue

                ref_feat_d = self._get_ref_feat_on_device(str(layer), x.device)
                if ref_feat_d is None:
                    lids_out.append(torch.full((int(x.shape[0]),), float('nan'), device=x.device))
                    continue

                if int(ref_feat_d.shape[0]) <= int(self.k):
                    lids_out.append(torch.full((int(x.shape[0]),), float('nan'), device=x.device))
                    continue

                feat = self._agg(layer_feats[layer])
                if self.feat_norm == 'l2':
                    feat = F.normalize(feat, p=2, dim=1)
                lids = mle_batch_gpu(ref_feat_d, feat, k=int(self.k), distance_metric=str(self.distance_metric))
                lids_out.append(lids)

            lid = torch.stack(lids_out, dim=1)
            try:
                if not bool(torch.isfinite(lid).all()):
                    self._recompute_lid_last_nonfinite = True
                    if not self._recompute_lid_warned_nonfinite:
                        self._recompute_lid_warned_nonfinite = True
                        n_bad = int((~torch.isfinite(lid)).sum().item())
                        print(f"[calibrator][warn] recompute_lid produced non-finite values: n_nonfinite={n_bad}")
            except Exception:
                pass
            return lid
        except Exception as e:
            self._recompute_lid_last_fallback = True
            if not self._recompute_lid_warned_fallback:
                self._recompute_lid_warned_fallback = True
                print(f"[calibrator][warn] recompute_lid failed; falling back to fixed clean LID. err={type(e).__name__}: {e}")
            return self.lid[: int(x.shape[0])]

    def _align_lid(self, x: torch.Tensor) -> torch.Tensor:
        lid = self.lid
        if self._active_idx is None:
            self._active_idx = torch.arange(int(lid.shape[0]), device=lid.device)

        if int(x.shape[0]) == int(lid.shape[0]) and int(self._active_idx.numel()) != int(lid.shape[0]):
            self._active_idx = torch.arange(int(lid.shape[0]), device=lid.device)

        if int(x.shape[0]) == int(lid.shape[0]):
            self._subbatch_cursor = 0
            self._subbatch_full = int(lid.shape[0])

        if int(self._active_idx.numel()) == int(x.shape[0]):
            self._last_x_full = x.detach()
            return lid.index_select(0, self._active_idx)

        if self._last_x_full is not None and int(self._last_x_full.shape[0]) == int(self._active_idx.numel()):
            try:
                x_full = self._last_x_full
                b_sub = int(x.shape[0])
                b_full = int(x_full.shape[0])

                x_sub_s = x[:, :, ::16, ::16].contiguous().view(int(x.shape[0]), -1)
                x_full_s = x_full[:, :, ::16, ::16].contiguous().view(int(x_full.shape[0]), -1)

                diff = (x_sub_s.view(b_sub, 1, -1) - x_full_s.view(1, b_full, -1)).abs().sum(dim=2)
                k = 2 if int(diff.shape[1]) >= 2 else 1
                topv, topi = diff.topk(k=int(k), largest=False, dim=1)
                bestv = topv[:, 0]
                besti = topi[:, 0]
                if int(k) >= 2:
                    secondv = topv[:, 1]
                else:
                    secondv = torch.full_like(bestv, float('inf'))

                denom = float(max(1, int(x_sub_s.shape[1])))
                best_mean = bestv / denom
                second_mean = secondv / denom

                if int(b_sub) == 1:
                    ok = (best_mean <= 0.06)
                else:
                    ok = (best_mean <= 0.04) & ((second_mean - best_mean) >= 0.01)

                if bool(torch.all(ok)) and (int(b_sub) == 1 or int(torch.unique(besti).numel()) == int(besti.numel())):
                    new_active = self._active_idx.index_select(0, besti.to(device=self._active_idx.device))
                    self._active_idx = new_active
                    self._last_x_full = x.detach()
                    return lid.index_select(0, self._active_idx)
            except Exception:
                pass

        if not self._warned_mismatch:
            print(
                f"[attack][warn] CalibratedAAWrapper lid batch mismatch: lid={int(lid.shape[0])} x={int(x.shape[0])}. "
                f"Falling back to slicing fixed LID for sub-batch (index match failed)."
            )
            self._warned_mismatch = True

        b_sub = int(x.shape[0])
        b_full = int(lid.shape[0])
        if b_sub <= 0:
            return lid[:0]

        base_idx = self._active_idx
        if base_idx is None or int(base_idx.numel()) != b_full:
            base_idx = torch.arange(b_full, device=lid.device)

        if int(self._subbatch_full) != b_full:
            self._subbatch_full = b_full
            self._subbatch_cursor = 0

        if int(self._subbatch_cursor) + b_sub > b_full:
            self._subbatch_cursor = 0

        start = int(self._subbatch_cursor)
        end = int(self._subbatch_cursor) + b_sub
        self._subbatch_cursor = end
        sub_idx = base_idx[start:end]
        if int(sub_idx.numel()) != b_sub:
            return lid[:b_sub]
        return lid.index_select(0, sub_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._fwd_calls += 1
        logits = self._base_logits(x)
        z = _to_single_logit_tensor(logits)

        if int(self.adaptive_lid_recompute_every) > 0 and (int(self._fwd_calls) % int(self.adaptive_lid_recompute_every) == 0):
            lid = self._recompute_lid(x)
        else:
            lid = self._align_lid(x)

        out = self.calibrator(lid, z, use_logits_input=self.use_logits_input)
        z_corr = out['z_corr']
        if z_corr.dim() == 1:
            z_corr = z_corr.view(-1, 1)
        logit_neg = torch.zeros_like(z_corr)
        return torch.cat([logit_neg, z_corr], dim=1)


def _pred_labels_from_model_output(out: torch.Tensor) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.dim() == 1:
        return (out > 0).long()
    if out.dim() == 2 and out.shape[1] == 1:
        return (out[:, 0] > 0).long()
    return torch.argmax(out, dim=1).long()


def _model_output_to_numpy_logits(out) -> np.ndarray:
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        out = torch.as_tensor(out)
    if out.dim() == 0:
        out = out.view(1, 1)
    elif out.dim() == 1:
        out = out.view(-1, 1)
    elif out.dim() >= 3:
        out = out.view(out.shape[0], -1)
    return out.detach().float().cpu().numpy()


def _find_detector_head_linear(model: nn.Module, model_type: str) -> nn.Linear:
    """Find the final detector head; its input is the penultimate feature."""
    model_type = str(model_type).lower()
    if model_type == 'clip':
        fc = getattr(model, 'fc', None)
        if isinstance(fc, nn.Linear):
            return fc
    if model_type == 'resnet' and hasattr(model, 'model'):
        fc = getattr(model.model, 'fc', None)
        if isinstance(fc, nn.Linear):
            return fc
    if model_type == 'forgelens' and hasattr(model, 'model'):
        head = getattr(model.model, 'fc', None)
        if isinstance(head, nn.Linear):
            return head
        if isinstance(head, nn.Sequential):
            for m in reversed(list(head.modules())):
                if isinstance(m, nn.Linear):
                    return m
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear) and int(m.out_features) <= 2:
            last = m
    if last is None:
        raise RuntimeError(f"Could not locate final detector Linear head for model_type={model_type}")
    return last


class DetectorPenultimateFeatureHook:
    """Capture the input to the detector's final Linear classification head."""
    def __init__(self, model: nn.Module, model_type: str):
        self.layer = _find_detector_head_linear(model, model_type)
        self.store: Optional[torch.Tensor] = None
        self.handle = self.layer.register_forward_pre_hook(self._hook)

    def _hook(self, _module, inputs):
        x = inputs[0]
        if torch.is_tensor(x):
            self.store = x.detach().reshape(int(x.shape[0]), -1).float().cpu()

    def pop_numpy(self, expected_bs: int, tag: str) -> np.ndarray:
        if self.store is None:
            raise RuntimeError(f"Detector penultimate feature hook did not capture {tag} features")
        arr = self.store.numpy().astype(np.float32, copy=False)
        self.store = None
        if int(arr.shape[0]) != int(expected_bs):
            raise RuntimeError(
                f"Detector penultimate feature batch mismatch for {tag}: "
                f"expected {int(expected_bs)}, got {int(arr.shape[0])}"
            )
        return arr

    def reset(self) -> None:
        self.store = None

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _print_logits_stats(name: str, logits: np.ndarray):
    if logits is None:
        return
    try:
        flat = np.asarray(logits).reshape(-1).astype(np.float32, copy=False)
        if flat.size == 0:
            return
        print(
            f"{name} logits: shape={np.asarray(logits).shape}, "
            f"mean={float(np.nanmean(flat)):.6f}, std={float(np.nanstd(flat)):.6f}, "
            f"min={float(np.nanmin(flat)):.6f}, max={float(np.nanmax(flat)):.6f}"
        )
    except Exception:
        print(f"{name} logits: shape={np.asarray(logits).shape}")


# ============= Attack generation =============


def _parse_number_str(v: str) -> float:
    v = str(v).strip()
    if v == '':
        return 0.0
    if '/' in v:
        a, b = v.split('/', 1)
        return float(a) / float(b)
    return float(v)


def _parse_float_list(value: str) -> List[float]:
    raw = str(value).strip()
    if raw == '':
        return []
    out: List[float] = []
    for item in raw.split(','):
        item = str(item).strip()
        if item == '':
            continue
        out.append(float(_parse_number_str(item)))
    return out


def _parse_int_range(value: str, default: tuple) -> tuple:
    raw = str(value).strip()
    if raw == '':
        return default
    a, b = [s.strip() for s in raw.split(',', 1)]
    return (int(a), int(b))


def _parse_float_range(value: str, default: tuple) -> tuple:
    raw = str(value).strip()
    if raw == '':
        return default
    a, b = [s.strip() for s in raw.split(',', 1)]
    return (float(_parse_number_str(a)), float(_parse_number_str(b)))


def _parse_steps_range_by_attack(value: str) -> Dict[str, tuple]:
    raw = str(value).strip()
    if raw == '':
        return {}
    out: Dict[str, tuple] = {}
    chunks = [c.strip() for c in raw.split(';') if c.strip()]
    for chunk in chunks:
        if ':' not in chunk:
            raise ValueError(f"Invalid steps_range_by_attack chunk (missing ':'): {chunk}")
        atk, rng = chunk.split(':', 1)
        atk = str(atk).strip().lower()
        a, b = [s.strip() for s in str(rng).split(',', 1)]
        out[atk] = (int(a), int(b))
    return out


def _schedule_eps(progress: float, eps_values: List[float], breaks: List[float]) -> float:
    if len(eps_values) == 0:
        raise ValueError('eps_schedule cannot be empty')
    if len(breaks) == 0 and len(eps_values) > 1:
        breaks = [float(i) / float(len(eps_values)) for i in range(1, len(eps_values))]
    if len(breaks) != max(0, len(eps_values) - 1):
        raise ValueError('eps_schedule_breaks length must be len(eps_schedule)-1')
    for idx, brk in enumerate(breaks):
        if float(progress) < float(brk):
            return float(eps_values[idx])
    return float(eps_values[-1])

def _instantiate_torchattack(attack_cls, model, **candidate_kwargs):
    sig = inspect.signature(attack_cls.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard('self')
    allowed.discard('model')
    kwargs = {k: v for k, v in candidate_kwargs.items() if k in allowed}
    return attack_cls(model, **kwargs)


def generate_adversarial(
    model,
    images,
    labels,
    guided_images=None,
    wrapped_model=None,
    aa_stage_eval_wrapped_model=None,
    attack_type='pgd',
    eps=8/255,
    alpha=2/255,
    steps=20,
    device='cuda',
    apgd_restarts=1,
    aa_norm='Linf',
    aa_version='standard',
    aa_seed=0,
    aa_backend='torchattacks',
    aa_stage_eval_recompute_lid: bool = False,
    aa_stage_log_recompute_lid: bool = False,
    fab_restarts=1,
    fab_beta=None,
    fab_eta=None,
    fab_alpha_max=None,
    pixle_pixels=1,
    pixle_restarts=1,
    square_queries=500,
    square_restarts=1,
    square_p_init=None,
    cw_c=1.0,
    cw_kappa=0.0,
    cw_lr=0.01,
    cw_binary_search_steps=None,
):
    """Generate adversarial examples"""
    if wrapped_model is None:
        wrapped = AAWrapper(model).to(device)
    else:
        wrapped = wrapped_model.to(device)
    wrapped.eval()

    eval_wrapped = None
    if aa_stage_eval_wrapped_model is not None:
        eval_wrapped = aa_stage_eval_wrapped_model.to(device)
        eval_wrapped.eval()
    
    attack_type = str(attack_type).lower()

    if attack_type in {'stat', 'statattack'}:
        if not _HAS_TA:
            raise RuntimeError("torchattacks not installed (required for StatAttack base class)")
        if not _HAS_STAT or _StatAttack is None:
            raise RuntimeError("evadingfakedetector StatAttack not available (missing evadingfakedetector-main or import failed)")
        if guided_images is None:
            raise ValueError("StatAttack requires guided_images (real batch) to be provided")

        # StatAttack is designed to transform fake images to look statistically like real images.
        # Only perturb fake samples (label==1). Keep real samples unchanged.
        labels_t = labels
        if not torch.is_tensor(labels_t):
            labels_t = torch.as_tensor(labels_t, device=images.device)
        labels_t = labels_t.to(device)
        labels_t = labels_t.view(-1)
        if labels_t.numel() != int(images.shape[0]):
            raise ValueError(
                f"StatAttack labels batch mismatch: got {int(labels_t.numel())} labels for {int(images.shape[0])} images"
            )
        fake_mask = labels_t.eq(1)
        if not bool(fake_mask.any().item()):
            return images

        base = getattr(model, 'model_type', None)
        if base is None:
            raise RuntimeError('model is missing model_type attribute for StatAttack integration')

        stat_model = StatAttackModelWrapper(model, model_type=str(base)).to(device)
        stat_model.eval()

        attacker = _StatAttack(
            stat_model,
            step=int(steps),
            epsilon=float(eps),
            epsilon_n=float(getattr(model, '_stat_epsilon_n', 4.0 / 255.0)),
            noise_lr=float(getattr(model, '_stat_noise_lr', 1.0 / 255.0)),
            lambda_mmd=float(getattr(model, '_stat_lambda_mmd', 10.0)),
            lambda_n=float(getattr(model, '_stat_lambda_n', 0.0)),
            lambda_b=float(getattr(model, '_stat_lambda_b', 10.0)),
            lambda_s=float(getattr(model, '_stat_lambda_s', 10.0)),
            bias_lr=float(getattr(model, '_stat_bias_lr', 1e-1)),
            spatial_lr=float(getattr(model, '_stat_spatial_lr', 1e-3)),
            degree=int(getattr(model, '_stat_degree', 11)),
            tune_scale=int(getattr(model, '_stat_tune_scale', 8)),
            blur_lr=float(getattr(model, '_stat_blur_lr', 1e-1)),
            radius=int(getattr(model, '_stat_radius', 3)),
            attack_mode=str(getattr(model, '_stat_attack_mode', 'all')),
            noise_mode=str(getattr(model, '_stat_noise_mode', 'add')),
            bias_mode=str(getattr(model, '_stat_bias_mode', 'same')),
            blur_attack=str(getattr(model, '_stat_blur_attack', 'add')),
        )

        guided_images = guided_images.to(device)
        images_fake = images[fake_mask]
        # Match guided batch size to the number of fake samples.
        guided_fake = guided_images
        if int(guided_fake.shape[0]) != int(images_fake.shape[0]):
            guided_fake = guided_fake[: int(images_fake.shape[0])]

        adv_fake = attacker(guided_fake, images_fake)
        if not torch.is_tensor(adv_fake):
            adv_fake = torch.as_tensor(adv_fake, device=device)
        adv_fake = torch.clamp(adv_fake, 0.0, 1.0)

        adv_all = images.clone()
        adv_all[fake_mask] = adv_fake
        return adv_all

    if attack_type != 'autoattack' and not _HAS_TA:
        raise RuntimeError("torchattacks not installed")

    if attack_type == 'fgsm':
        attacker = ta.FGSM(wrapped, eps=eps)
    elif attack_type == 'pgd':
        attacker = ta.PGD(wrapped, eps=eps, alpha=alpha, steps=steps, random_start=True)
    elif attack_type == 'apgd':
        apgd_cls = getattr(ta, 'APGD', None)
        if apgd_cls is None:
            raise RuntimeError(
                "APGD is not available in your installed torchattacks. "
                "Please upgrade torchattacks or choose another attack."
            )
        candidate = {
            'eps': eps,
            'steps': int(steps),
            'iters': int(steps),
            'n_restarts': int(apgd_restarts),
            'restarts': int(apgd_restarts),
        }
        attacker = _instantiate_torchattack(apgd_cls, wrapped, **candidate)
    elif attack_type == 'bim':
        attacker = _instantiate_torchattack(
            ta.BIM,
            wrapped,
            eps=eps,
            alpha=alpha,
            iters=steps,
            steps=steps,
        )
    elif attack_type == 'cw':
        candidate = {
            'c': float(cw_c),
            'kappa': float(cw_kappa),
            'steps': int(steps),
            'iters': int(steps),
            'lr': float(cw_lr),
        }
        if cw_binary_search_steps is not None:
            candidate['binary_search_steps'] = int(cw_binary_search_steps)
            candidate['binary_search_step'] = int(cw_binary_search_steps)

        attacker = _instantiate_torchattack(ta.CW, wrapped, **candidate)
    elif attack_type == 'square':
        candidate = {
            'eps': eps,
            'n_queries': int(square_queries),
            'n_restarts': int(square_restarts),
        }
        if square_p_init is not None:
            candidate['p_init'] = float(square_p_init)
            candidate['p'] = float(square_p_init)
        attacker = _instantiate_torchattack(ta.Square, wrapped, **candidate)
    elif attack_type == 'fab':
        fab_cls = getattr(ta, 'FAB', None)
        if fab_cls is None:
            raise RuntimeError(
                "FAB is not available in your installed torchattacks. "
                "Please upgrade torchattacks (FAB was added in newer versions) "
                "or choose another attack."
            )
        candidate = {
            'eps': eps,
            'steps': steps,
            'iters': steps,
            'n_classes': 2,
            'n_restarts': int(fab_restarts),
            'restarts': int(fab_restarts),
        }
        if fab_beta is not None:
            candidate['beta'] = float(fab_beta)
        if fab_eta is not None:
            candidate['eta'] = float(fab_eta)
        if fab_alpha_max is not None:
            candidate['alpha_max'] = float(fab_alpha_max)

        attacker = _instantiate_torchattack(
            fab_cls,
            wrapped,
            **candidate,
        )
    elif attack_type == 'pixle':
        pixle_cls = getattr(ta, 'Pixle', None)
        if pixle_cls is None:
            raise RuntimeError(
                "Pixle attack is not available in your installed torchattacks. "
                "Please upgrade torchattacks to a version that includes Pixle, "
                "or implement Pixle via an additional dependency."
            )
        attacker = _instantiate_torchattack(
            pixle_cls,
            wrapped,
            eps=eps,
            steps=steps,
            iters=steps,
            n_restarts=int(pixle_restarts),
            restarts=int(pixle_restarts),
            pixels=int(pixle_pixels),
            n_pixels=int(pixle_pixels),
            max_iterations=steps,
            max_iters=steps,
        )
    elif attack_type == 'autoattack':
        aa_backend = str(aa_backend).strip().lower() or 'torchattacks'

        if aa_backend == 'official':
            if not _HAS_OFFICIAL_AA:
                raise RuntimeError(
                    "Official AutoAttack not installed. Install via either:\n"
                    "  pip install autoattack\n"
                    "or\n"
                    "  pip install git+https://github.com/fra31/auto-attack.git"
                )

            # Determine number of classes from wrapped model output.
            n_classes = None
            try:
                with torch.no_grad():
                    out_probe = wrapped(images[:1])
                    if isinstance(out_probe, (tuple, list)):
                        out_probe = out_probe[0]
                    if torch.is_tensor(out_probe) and out_probe.dim() == 2:
                        n_classes = int(out_probe.shape[1])
            except Exception:
                n_classes = None

            # Official AutoAttack standard uses APGD-DLR variants; for binary logits this may be ill-defined.
            # Use a binary-safe custom ensemble: APGD-CE + FAB + Square.
            version = str(aa_version)
            attacks_to_run = []
            if n_classes is not None and int(n_classes) <= 2:
                version = 'custom'
                attacks_to_run = ['apgd-ce', 'fab', 'square']

            candidate = {
                'model': wrapped,
                'norm': str(aa_norm),
                'eps': float(eps),
                'version': str(version),
                'seed': int(aa_seed),
                'verbose': False,
                'attacks_to_run': list(attacks_to_run),
            }
            sig = inspect.signature(OfficialAutoAttack.__init__)
            allowed = set(sig.parameters.keys())
            allowed.discard('self')
            kwargs = {k: v for k, v in candidate.items() if k in allowed}
            adversary = OfficialAutoAttack(**kwargs)

            try:
                adv = adversary.run_standard_evaluation(images, labels, bs=int(images.shape[0]))
            except Exception:
                # Last-resort fallback for binary: enforce custom attacks_to_run.
                if (n_classes is not None and int(n_classes) <= 2) and hasattr(adversary, 'attacks_to_run'):
                    adversary.attacks_to_run = ['apgd-ce', 'fab', 'square']
                    adv = adversary.run_standard_evaluation(images, labels, bs=int(images.shape[0]))
                else:
                    raise

            if not torch.is_tensor(adv):
                adv = torch.as_tensor(adv, device=images.device)
            return torch.clamp(adv, 0.0, 1.0)

        if not _HAS_TA:
            raise RuntimeError("torchattacks not installed (required for --aa_backend torchattacks)")

        aa_cls = getattr(ta, 'AutoAttack', None)
        if aa_cls is None:
            raise RuntimeError(
                "AutoAttack is not available in your installed torchattacks. "
                "If you intended to use the 'autoattack' package by Croce & Hein, "
                "please install it and add an integration here, or upgrade torchattacks."
            )
        n_classes = None
        try:
            with torch.no_grad():
                out_probe = wrapped(images[:1])
                if isinstance(out_probe, (tuple, list)):
                    out_probe = out_probe[0]
                if torch.is_tensor(out_probe) and out_probe.dim() == 2:
                    n_classes = int(out_probe.shape[1])
        except Exception:
            n_classes = None

        if n_classes is not None and int(n_classes) <= 2:
            multi_cls = getattr(ta, 'MultiAttack', None)
            apgd_cls = getattr(ta, 'APGD', None)
            fab_cls = getattr(ta, 'FAB', None)
            square_cls = getattr(ta, 'Square', None)
            if multi_cls is None or apgd_cls is None or fab_cls is None or square_cls is None:
                raise RuntimeError(
                    "Binary AutoAttack fallback requires torchattacks MultiAttack/APGD/FAB/Square. "
                    "Please upgrade torchattacks or choose another attack."
                )
            print(
                "[attack][warn] torchattacks AutoAttack uses targeted DLR components that are incompatible with binary logits; "
                "falling back to MultiAttack(APGD+FAB+Square)."
            )

            apgd_steps = int(steps) if steps is not None else 100
            apgd_r = int(apgd_restarts) if apgd_restarts is not None else 1
            fab_steps = int(steps) if steps is not None else 101
            fab_r = int(fab_restarts) if fab_restarts is not None else 1
            sq_q = int(square_queries) if square_queries is not None else 10000
            sq_r = int(square_restarts) if square_restarts is not None else 1

            apgd = _instantiate_torchattack(
                apgd_cls,
                wrapped,
                eps=float(eps),
                norm=str(aa_norm),
                seed=int(aa_seed),
                verbose=False,
                loss='ce',
                steps=int(apgd_steps),
                iters=int(apgd_steps),
                n_restarts=int(apgd_r),
                restarts=int(apgd_r),
            )
            fab = _instantiate_torchattack(
                fab_cls,
                wrapped,
                eps=float(eps),
                norm=str(aa_norm),
                seed=int(aa_seed),
                verbose=False,
                steps=int(fab_steps),
                iters=int(fab_steps),
                n_classes=2,
                multi_targeted=True,
                n_restarts=int(fab_r),
                restarts=int(fab_r),
                beta=(0.9 if fab_beta is None else float(fab_beta)),
                eta=(1.3 if fab_eta is None else float(fab_eta)),
                alpha_max=(0.05 if fab_alpha_max is None else float(fab_alpha_max)),
            )
            square = _instantiate_torchattack(
                square_cls,
                wrapped,
                eps=float(eps),
                norm=str(aa_norm),
                seed=int(aa_seed),
                verbose=False,
                loss='margin',
                resc_schedule=True,
                n_queries=int(sq_q),
                n_restarts=int(sq_r),
                p_init=(0.05 if square_p_init is None else float(square_p_init)),
                p=(0.05 if square_p_init is None else float(square_p_init)),
            )

            def _pred_ok(x: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    out = wrapped(x)
                    pred = _pred_labels_from_model_output(out)
                return (pred == labels)

            def _pred_ok_true_calibrated_recompute_lid(x: torch.Tensor, cal_wrap: nn.Module) -> torch.Tensor:
                try:
                    if not isinstance(cal_wrap, CalibratedAAWrapper):
                        raise RuntimeError('wrapped is not CalibratedAAWrapper')
                    with torch.no_grad():
                        if hasattr(cal_wrap, '_base_logits') and callable(getattr(cal_wrap, '_base_logits')):
                            logits = cal_wrap._base_logits(x)
                        else:
                            logits = cal_wrap.base_model(x)
                        z = _to_single_logit_tensor(logits)
                        lid = cal_wrap._recompute_lid(x)
                        if bool(getattr(cal_wrap, '_recompute_lid_last_fallback', False)):
                            print(
                                "[attack][autoattack][warn] stage-eval recompute_lid failed; used fixed clean LID. "
                                "remaining may be unreliable for this call."
                            )
                        if bool(getattr(cal_wrap, '_recompute_lid_last_nonfinite', False)):
                            print(
                                "[attack][autoattack][warn] stage-eval recompute_lid produced non-finite values; "
                                "remaining may be unreliable for this call."
                            )
                        out = cal_wrap.calibrator(lid, z, use_logits_input=cal_wrap.use_logits_input)
                        z_corr = out['z_corr']
                        if z_corr.dim() == 1:
                            z_corr = z_corr.view(-1, 1)
                        logit_neg = torch.zeros_like(z_corr)
                        cal_logits = torch.cat([logit_neg, z_corr], dim=1)
                        pred = _pred_labels_from_model_output(cal_logits)
                    return (pred == labels)
                except Exception:
                    return _pred_ok(x)

            stage_eval_model = eval_wrapped if eval_wrapped is not None else wrapped
            stage_eval_true_calibrated = bool(aa_stage_eval_recompute_lid) and isinstance(stage_eval_model, CalibratedAAWrapper)
            if bool(aa_stage_eval_recompute_lid) and not stage_eval_true_calibrated:
                print(
                    "[attack][autoattack][warn] aa_stage_eval_recompute_lid was requested, but stage eval model is not CalibratedAAWrapper; "
                    "falling back to surrogate-fixed-lid stage evaluation. To evaluate true calibrated (recompute-LID) success per stage while "
                    "attacking the backbone, provide --calibrator_ckpt and keep --aa_stage_eval_recompute_lid enabled."
                )

            def pred_ok_stage(x: torch.Tensor) -> torch.Tensor:
                if bool(stage_eval_true_calibrated):
                    return _pred_ok_true_calibrated_recompute_lid(x, stage_eval_model)
                return _pred_ok(x)

            clean_ok_log = None
            if bool(aa_stage_log_recompute_lid):
                try:
                    clean_ok_log = _pred_ok_true_calibrated_recompute_lid(images, stage_eval_model)
                except Exception:
                    clean_ok_log = None

            clean_ok = pred_ok_stage(images)
            remaining = clean_ok.clone()
            adv = images.detach().clone()

            def _run_stage(name: str, atk_obj) -> None:
                nonlocal adv, remaining
                n_rem = int(remaining.sum().item())
                if n_rem == 0:
                    print(f"[attack][autoattack] stage={name} remaining=0 (skip)")
                    return
                idx = remaining.nonzero(as_tuple=False).view(-1)
                x_in = images.index_select(0, idx)
                y_in = labels.index_select(0, idx)
                adv_sub = atk_obj(x_in, y_in)
                adv_stage = adv.detach().clone()
                adv_stage.index_copy_(0, idx, adv_sub.detach())
                adv_ok = pred_ok_stage(adv_stage)
                new_remaining = clean_ok & adv_ok
                gained = int((remaining & (~new_remaining)).sum().item())
                try:
                    succ_sub = (~adv_ok).index_select(0, idx)
                    if int(succ_sub.sum().item()) > 0:
                        succ_idx = idx.index_select(0, succ_sub.nonzero(as_tuple=False).view(-1))
                        adv_succ = adv_sub.detach().index_select(0, succ_sub.nonzero(as_tuple=False).view(-1))
                        adv.index_copy_(0, succ_idx, adv_succ)
                except Exception:
                    adv.index_copy_(0, idx, adv_sub.detach())
                remaining = new_remaining
                base = int(clean_ok.sum().item())
                succ = int((clean_ok & (~adv_ok)).sum().item())
                print(
                    f"[attack][autoattack] stage={name} rem_before={n_rem} gained={gained} succ_total={succ}/{base} "
                    f"succ_rate={float(succ / max(1, base)):.4f} "
                    f"({'true-calibrated-recompute-lid' if bool(stage_eval_true_calibrated) else 'surrogate-fixed-lid'}, batch-local)"
                )

                if bool(aa_stage_log_recompute_lid) and clean_ok_log is not None:
                    try:
                        adv_ok_log = _pred_ok_true_calibrated_recompute_lid(adv, stage_eval_model)
                        base_l = int(clean_ok_log.sum().item())
                        succ_l = int((clean_ok_log & (~adv_ok_log)).sum().item())
                        print(
                            f"[attack][autoattack][log] stage={name} succ_total={succ_l}/{base_l} "
                            f"succ_rate={float(succ_l / max(1, base_l)):.4f} (true-calibrated-recompute-lid, log-only)"
                        )
                    except Exception as e:
                        print(f"[attack][autoattack][log][warn] stage={name} recompute-lid metric failed: {e}")

            print(
                f"[attack][autoattack] binary fallback params: eps={float(eps):.6f} "
                f"apgd_steps={int(apgd_steps)} apgd_restarts={int(apgd_r)} "
                f"fab_steps={int(fab_steps)} fab_restarts={int(fab_r)} "
                f"square_queries={int(sq_q)} square_restarts={int(sq_r)}"
            )
            print(
                "[attack][autoattack] note: final success metrics are reported later as 'Base/Calibrated attack success rate' "
                "(calibrated uses recomputed LID + calibrator-defined masks)."
            )

            _run_stage('apgd', apgd)
            _run_stage('fab', fab)
            _run_stage('square', square)
            return torch.clamp(adv, 0.0, 1.0)
        else:
            candidate = {
                'norm': str(aa_norm),
                'eps': float(eps),
                'version': str(aa_version),
                'seed': int(aa_seed),
                'n_classes': 2,
            }
            attacker = _instantiate_torchattack(aa_cls, wrapped, **candidate)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")

    # IMPORTANT: gradient-based attacks (e.g. PGD/CW) need autograd enabled.
    adv = attacker(images, labels)
    return torch.clamp(adv, 0.0, 1.0)


def generate_noisy(images, eps=8/255, match_l2=None, device='cuda'):
    """Generate noisy samples with controlled L2 norm"""
    noise = torch.randn_like(images, device=device)
    
    if match_l2 is not None:
        # Match specific L2 norm
        noise = noise / torch.norm(noise.view(noise.shape[0], -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
        noise = noise * match_l2
    else:
        # Use epsilon for uniform noise
        noise = noise * eps
    
    noisy = images + noise
    return torch.clamp(noisy, 0.0, 1.0)


# ============= LID computation =============
def compute_lid_features(
    model,
    ref_loader,
    query_loader,
    query_type,
    layers,
    k=20,
    chunk_size=512,
    device='cuda',
    feat_norm='l2',
    distance_metric='euclidean',
    ref_mode='global',
    batch_ref_loader=None,
    batch_ref_size=0,
    spatial_feat_mode='flatten',
    pca_dim=0,
    pca_whiten=False,
    pca_standardize=False,
    pca_seed=42,
):
    """
    Compute per-layer LID features
    Returns: (n_samples, n_layers) array of LID values
    """

    spatial_feat_mode = str(spatial_feat_mode)
    if spatial_feat_mode not in ['avgpool', 'flatten']:
        raise ValueError(f"Unknown spatial_feat_mode: {spatial_feat_mode}")

    def _agg(feat):
        if not isinstance(feat, torch.Tensor):
            feat = torch.as_tensor(feat)
        if feat.dim() > 2:
            if spatial_feat_mode == 'avgpool':
                feat = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)
            else:
                feat = feat.reshape(feat.shape[0], -1)
        else:
            feat = feat.reshape(feat.shape[0], -1)
        return feat

    if ref_mode == 'batch_clean':
        if batch_ref_loader is None:
            raise ValueError("ref_mode='batch_clean' requires batch_ref_loader")
        if pca_dim and int(pca_dim) > 0:
            raise ValueError("PCA is only supported with ref_mode='global'")

        ref_size = int(batch_ref_size) if batch_ref_size is not None else 0
        if ref_size <= 0:
            ref_size = 0

        print(f"Computing LID for {query_type} samples...")
        lid_features = []

        ref_buf = {layer: None for layer in layers}

        with torch.no_grad():
            for (ref_images, _), (images, _) in tqdm(
                zip(batch_ref_loader, query_loader),
                desc=query_type,
                total=min(len(batch_ref_loader), len(query_loader)),
            ):
                ref_images = ref_images.to(device)
                images = images.to(device)

                ref_layer_feats = model.extract_layer_features(ref_images)
                layer_feats = model.extract_layer_features(images)

                for layer in layers:
                    if layer not in ref_layer_feats:
                        continue
                    ref_feat = ref_layer_feats[layer]
                    ref_feat = _agg(ref_feat)
                    if feat_norm == 'l2':
                        ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    ref_feat = ref_feat.detach().cpu()

                    if ref_buf[layer] is None:
                        buf = ref_feat
                    else:
                        buf = torch.cat([ref_buf[layer], ref_feat], dim=0)

                    if ref_size and buf.shape[0] > ref_size:
                        buf = buf[-ref_size:]
                    ref_buf[layer] = buf

                batch_lids = []
                for layer in layers:
                    if layer in layer_feats and ref_buf[layer] is not None:
                        feat = layer_feats[layer]
                        feat = _agg(feat)
                        if feat_norm == 'l2':
                            feat = F.normalize(feat, p=2, dim=1)

                        ref_feat = ref_buf[layer].to(device, non_blocking=True)
                        lids = mle_batch_gpu(ref_feat, feat, k=k, distance_metric=distance_metric)
                        batch_lids.append(lids.cpu().numpy())
                    else:
                        batch_lids.append(np.full(images.shape[0], np.nan))

                batch_lids = np.stack(batch_lids, axis=1)
                lid_features.append(batch_lids)

        return np.vstack(lid_features)

    # Extract reference features for each layer
    print(f"Extracting reference features...")
    ref_feats = {layer: [] for layer in layers}
    
    with torch.no_grad():
        for images, _ in tqdm(ref_loader, desc='Reference'):
            images = images.to(device)
            layer_feats = model.extract_layer_features(images)
            for layer in layers:
                if layer in layer_feats:
                    feat = layer_feats[layer]
                    feat = _agg(feat)
                    if feat_norm == 'l2':
                        feat = F.normalize(feat, p=2, dim=1)
                    ref_feats[layer].append(feat.cpu())
    
    # Concatenate reference features
    for layer in layers:
        ref_feats[layer] = torch.cat(ref_feats[layer], dim=0)

    pca_models = None
    if pca_dim and int(pca_dim) > 0:
        if not _HAS_SK:
            raise RuntimeError("scikit-learn is required for PCA but is not installed")
        pca_models = {}
        for layer in layers:
            ref_np = ref_feats[layer].numpy().astype(np.float32, copy=False)
            scaler = None
            if bool(pca_standardize):
                scaler = StandardScaler(with_mean=True, with_std=True)
                ref_np = scaler.fit_transform(ref_np)

            max_comp = min(ref_np.shape[0] - 1, ref_np.shape[1])
            n_comp = min(int(pca_dim), int(max_comp))
            if n_comp <= 0:
                raise ValueError(
                    f"Invalid pca_dim={pca_dim} for layer {layer}: ref shape={ref_np.shape}"
                )

            pca = PCA(n_components=int(n_comp), whiten=bool(pca_whiten), random_state=int(pca_seed))
            ref_proj = pca.fit_transform(ref_np).astype(np.float32, copy=False)
            ref_feats[layer] = torch.from_numpy(ref_proj)
            pca_models[layer] = (scaler, pca)
    
    # Extract query features and compute LID
    print(f"Computing LID for {query_type} samples...")
    lid_features = []
    
    with torch.no_grad():
        for images, _ in tqdm(query_loader, desc=query_type):
            images = images.to(device)
            layer_feats = model.extract_layer_features(images)
            bs = int(images.shape[0])
            
            batch_lids = []
            for layer in layers:
                if layer in layer_feats:
                    feat = layer_feats[layer]
                    feat = _agg(feat)
                    if feat_norm == 'l2':
                        feat = F.normalize(feat, p=2, dim=1)

                    if pca_models is not None:
                        feat_np = feat.cpu().numpy().astype(np.float32, copy=False)
                        scaler, pca = pca_models[layer]
                        if scaler is not None:
                            feat_np = scaler.transform(feat_np)
                        feat_np = pca.transform(feat_np).astype(np.float32, copy=False)
                        feat = torch.from_numpy(feat_np).to(device)

                    ref = ref_feats[layer].to(device)
                    lids = mle_batch_gpu(ref, feat, k=k, distance_metric=distance_metric)
                    batch_lids.append(lids.cpu().numpy())
                else:
                    # If layer not found, use NaN
                    batch_lids.append(np.full(bs, np.nan))
            
            # Stack layer LIDs: (batch_size, n_layers)
            batch_lids = np.stack(batch_lids, axis=1)
            lid_features.append(batch_lids)
    
    return np.vstack(lid_features)


# ============= Main =============
def main():
    parser = argparse.ArgumentParser(description='Extract per-layer LID features for classifier training')
    
    # Model settings
    parser.add_argument('--model_type', type=str, default='clip', choices=['resnet', 'clip', 'forgelens', 'gram', 'csf'])
    parser.add_argument('--model_path', type=str, default=None, help='Path to model weights')
    parser.add_argument('--clip_model', type=str, default='ViT-L/14')
    parser.add_argument('--fc_weights', type=str, default='./UniversalFakeDetect-main/pretrained_weights/fc_weights.pth')

    parser.add_argument('--forgelens_stage', type=int, default=2, choices=[1, 2])
    parser.add_argument(
        '--forgelens_feature_set',
        type=str,
        default='faformer',
        choices=['faformer', 'clip_proj', 'clip_unproj', 'clip_both', 'all_proj', 'all_unproj', 'all_both'],
    )
    parser.add_argument('--forgelens_wsgm_count', type=int, default=4)
    parser.add_argument('--forgelens_wsgm_reduction_factor', type=int, default=4)
    parser.add_argument('--forgelens_faformer_layers', type=int, default=2)
    parser.add_argument('--forgelens_faformer_reduction_factor', type=int, default=1)
    parser.add_argument('--forgelens_faformer_head', type=int, default=2)

    parser.add_argument('--include_input', action='store_true')
    parser.add_argument('--layers', type=str, default=None)

    parser.add_argument('--resnet_feat_mode', type=str, default='flatten', choices=['avgpool', 'flatten'])
    parser.add_argument('--clip_feat_mode', type=str, default='flatten', choices=['mean_patch', 'cls', 'flatten', 'flatten_tokens'])

    parser.add_argument('--clip_load_size', type=int, default=256)
    parser.add_argument('--clip_crop_size', type=int, default=224)
    parser.add_argument('--clip_no_resize', action='store_true')
    parser.add_argument('--clip_resize_mode', type=str, default='short_side', choices=['short_side', 'square'])
    parser.add_argument('--clip_disable_normalize', action='store_true')
    parser.add_argument('--clip_normalize_mode', type=str, default='auto', choices=['auto', 'on', 'off'])

    parser.add_argument('--img_load_size', type=int, default=224)
    parser.add_argument('--img_crop_size', type=int, default=224)
    parser.add_argument('--img_resize_mode', type=str, default='short_side', choices=['short_side', 'square'])

    parser.add_argument('--sanity_probe_n_each', type=int, default=None)
    parser.add_argument('--sanity_eval_batches', type=int, default=0)
    
    # Data settings
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--ref_samples', type=int, default=1000, help='Number of reference samples')
    parser.add_argument('--query_samples', type=int, default=1000, help='Number of query samples')

    parser.add_argument('--query_dir', type=str, default='', help='Optional: directory containing external query images (e.g., adversarial images). If set, the script will extract logits/LID for these images and save an npz, without generating attacks.')
    parser.add_argument('--query_glob', type=str, default='*_adv_image.png', help='Glob pattern for external query images under --query_dir')
    parser.add_argument('--query_label', type=int, default=1, help='Label assigned to external query images (default: 1=fake)')
    parser.add_argument('--query_limit', type=int, default=0, help='Limit number of external query images (0 means all)')
    parser.add_argument('--external_metadata', type=str, default='', help='Optional CSV metadata exported by StealthDiffusion main.py for external adversarial samples')
    parser.add_argument('--external_clean_dir', type=str, default='', help='Optional clean image directory used instead of paired origin images when external_metadata is provided')
    parser.add_argument('--external_clean_samples', type=int, default=0, help='Limit number of clean images drawn from --external_clean_dir (0 means all)')
    parser.add_argument('--external_ref_dir', type=str, default='', help='Optional reference image directory used instead of the default reference split when external_metadata is provided')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_shuffle_paths', action='store_false', dest='shuffle_paths')
    parser.set_defaults(shuffle_paths=True)

    parser.add_argument('--no_validate_images', action='store_false', dest='validate_images')
    parser.set_defaults(validate_images=True)

    parser.add_argument('--ref_mode', type=str, default='batch_clean', choices=['global', 'batch_clean'])
    parser.add_argument('--batch_ref_size', type=int, default=0)
    
    # Attack settings
    parser.add_argument(
        '--attack_type',
        type=str,
        default='pgd',
        choices=['fgsm', 'pgd', 'apgd', 'bim', 'fab', 'pixle', 'cw', 'square', 'autoattack'],
    )
    parser.add_argument('--attack_target', type=str, default='base', choices=['base', 'calibrated'])
    parser.add_argument('--calibrator_ckpt', type=str, default='')
    parser.add_argument('--epsilon', type=float, default=8/255)
    parser.add_argument('--alpha', type=float, default=2/255)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--apgd_restarts', type=int, default=1)
    parser.add_argument('--aa_norm', type=str, default='Linf')
    parser.add_argument('--aa_version', type=str, default='standard')
    parser.add_argument('--aa_seed', type=int, default=0)
    parser.add_argument('--aa_backend', type=str, default='torchattacks', choices=['torchattacks', 'official'])
    parser.add_argument('--aa_stage_eval_recompute_lid', action='store_true')
    parser.add_argument('--aa_stage_log_recompute_lid', action='store_true')
    parser.add_argument('--autoattack_success_target', type=str, default='base', choices=['base', 'calibrated'])
    parser.add_argument('--adaptive_lid_recompute_every', type=int, default=0)
    parser.add_argument('--debug_attack_calibrated_postcheck', action='store_true')
    parser.add_argument('--fab_restarts', type=int, default=1)
    parser.add_argument('--fab_beta', type=float, default=None)
    parser.add_argument('--fab_eta', type=float, default=None)
    parser.add_argument('--fab_alpha_max', type=float, default=None)
    parser.add_argument('--pixle_pixels', type=int, default=1)
    parser.add_argument('--pixle_restarts', type=int, default=1)
    parser.add_argument('--square_queries', type=int, default=500)
    parser.add_argument('--square_restarts', type=int, default=1)
    parser.add_argument('--square_p_init', type=float, default=None)
    parser.add_argument('--square_queries_range', type=str, default='')
    parser.add_argument('--cw_c', type=float, default=1.0)
    parser.add_argument('--cw_kappa', type=float, default=0.0)
    parser.add_argument('--cw_lr', type=float, default=0.01)
    parser.add_argument('--cw_binary_search_steps', type=int, default=None)
    parser.add_argument('--add_noisy', action='store_true', help='Add noisy samples as negative class')

    parser.add_argument('--dynamic_attack', action='store_true')
    parser.add_argument('--randomize_attack_params_per_batch', action='store_true')
    parser.add_argument('--eps_schedule', type=str, default='')
    parser.add_argument('--eps_schedule_breaks', type=str, default='')
    parser.add_argument('--steps_range', type=str, default='')
    parser.add_argument('--steps_range_by_attack', type=str, default='')
    parser.add_argument('--alpha_divisor', type=float, default=5.0)
    parser.add_argument('--cw_kappa_range', type=str, default='')
    
    # LID settings
    parser.add_argument('--lid_k', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--chunk_size', type=int, default=512)

    parser.add_argument('--feat_norm', type=str, default='l2', choices=['l2', 'none'])

    parser.add_argument(
        '--lid_distance_metric',
        type=str,
        default='euclidean',
        choices=['euclidean', 'cosine', 'manhattan', 'chebyshev'],
    )

    parser.add_argument('--pca_dim', type=int, default=0)
    parser.add_argument('--pca_whiten', action='store_true')
    parser.add_argument('--pca_standardize', action='store_true')
    parser.add_argument('--pca_seed', type=int, default=42)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./lid_features')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument(
        '--save_detector_features',
        action='store_true',
        help='Also save clean_detector_features and adv_detector_features: the input to the detector final head.',
    )
    
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment name
    if args.exp_name is None:
        args.exp_name = f"{args.model_type}_{args.attack_type}_k{args.lid_k}"
    
    # Load model
    print(f"Loading {args.model_type} model...")
    if args.model_type == 'resnet':
        model = ResNetFeatureExtractor(args.model_path, device=device, include_input=bool(args.include_input))
        layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
    elif args.model_type == 'gram':
        if str(args.model_path).strip() == '' or not os.path.exists(str(args.model_path)):
            raise ValueError("model_type=gram requires a valid --model_path")
        model = GramFeatureExtractor(args.model_path, device=device, include_input=bool(args.include_input))
        layers = ['x', 'x8', 'g2', 'g3', 'g4']
    elif args.model_type == 'csf':
        if str(args.model_path).strip() == '' or not os.path.exists(str(args.model_path)):
            raise ValueError("model_type=csf requires a valid --model_path")
        model = CSFFeatureExtractor(args.model_path, device=device, include_input=bool(args.include_input))
        layers = [
            'extract_out',
            'head_layer1', 'head_layer2', 'head_layer3', 'head_layer4',
            'head_layer5', 'head_layer6', 'head_layer7', 'head_layer8', 'head_layer9', 'head_layer10',
            'avg_pool1', 'avg_pool2', 'avg_pool3',
            'flatten',
        ]
    elif args.model_type == 'clip':
        _norm_mode = 'auto'
        if bool(getattr(args, 'clip_disable_normalize', False)):
            _norm_mode = 'off'
        else:
            _norm_mode = str(getattr(args, 'clip_normalize_mode', 'auto')).lower().strip()
            if _norm_mode not in {'auto', 'on', 'off'}:
                _norm_mode = 'auto'
        model = CLIPFeatureExtractor(
            model_name=args.clip_model,
            model_path=args.model_path,
            fc_weights_path=args.fc_weights,
            device=device,
            feat_mode=str(args.clip_feat_mode),
            include_input=bool(args.include_input),
            apply_input_normalize=(False if _norm_mode == 'off' else True),
        )
        layers = [f'layer{i}' for i in range(model.get_num_layers())]
    else:
        if str(args.model_path).strip() == '' or not os.path.exists(str(args.model_path)):
            raise ValueError("model_type=forgelens requires a valid --model_path")
        model = ForgeLensFeatureExtractor(
            model_path=str(args.model_path),
            device=str(args.device),
            include_input=bool(args.include_input),
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
            layers = [f'layer{i}' for i in range(int(fa_n))]
        elif fs == 'clip_proj':
            layers = [f'clip_layer{i}' for i in range(int(clip_n))]
        elif fs == 'clip_unproj':
            layers = [f'clipu_layer{i}' for i in range(int(clip_n))]
        elif fs == 'clip_both':
            layers = [f'clip_layer{i}' for i in range(int(clip_n))] + [f'clipu_layer{i}' for i in range(int(clip_n))]
        elif fs == 'all_proj':
            layers = [f'layer{i}' for i in range(int(fa_n))] + [f'clip_layer{i}' for i in range(int(clip_n))]
        elif fs == 'all_unproj':
            layers = [f'layer{i}' for i in range(int(fa_n))] + [f'clipu_layer{i}' for i in range(int(clip_n))]
        else:
            layers = (
                [f'layer{i}' for i in range(int(fa_n))]
                + [f'clip_layer{i}' for i in range(int(clip_n))]
                + [f'clipu_layer{i}' for i in range(int(clip_n))]
            )

    if args.include_input and 'input' not in layers:
        layers = ['input'] + layers

    if args.layers is not None:
        raw = [s.strip() for s in str(args.layers).split(',') if s.strip()]
        sel = []
        for item in raw:
            if args.model_type == 'clip' and item.isdigit():
                sel.append(f'layer{item}')
            else:
                sel.append(item)
        layers = sel

    # Build input transform for ImageDataset.
    # Model wrappers handle normalization internally; transforms should output float tensor in [0,1].
    if str(args.model_type) == 'clip':
        _clip_load = int(args.clip_load_size)
        if bool(getattr(args, 'clip_no_resize', False)):
            _clip_load = 0
        transform = CLIPImageTransform(
            load_size=_clip_load,
            crop_size=int(args.clip_crop_size),
            resize_mode=str(args.clip_resize_mode),
        )
    elif str(args.model_type) == 'forgelens':
        transform = ForgeLensImageTransform(size=int(args.img_crop_size))
    elif str(args.model_type) == 'csf':
        transform = CSFImageTransform(size=256, patch_size=32)
    else:
        transform = ImageTransform(
            load_size=int(args.img_load_size),
            crop_size=int(args.img_crop_size),
            resize_mode=str(args.img_resize_mode),
        )

    # Load image paths
    data_path = Path(args.data_dir)
    real_dir = data_path / '0_real'
    fake_dir = data_path / '1_fake'

    if real_dir.exists() and fake_dir.exists():
        real_roots = [real_dir]
        fake_roots = [fake_dir]
    else:
        real_roots = sorted([p for p in data_path.rglob('0_real') if p.is_dir()])
        fake_roots = sorted([p for p in data_path.rglob('1_fake') if p.is_dir()])
        if not real_roots:
            raise ValueError(
                f"Real directory does not exist under {data_path}: expected {real_dir} or any '**/0_real'"
            )
        if not fake_roots:
            raise ValueError(
                f"Fake directory does not exist under {data_path}: expected {fake_dir} or any '**/1_fake'"
            )

    if int(args.ref_samples) % 2 != 0:
        raise ValueError(f"ref_samples must be even (got {args.ref_samples})")
    if int(args.query_samples) <= 0:
        raise ValueError(f"query_samples must be positive (got {args.query_samples})")
    if int(getattr(args, 'external_clean_samples', 0)) < 0:
        raise ValueError(f"external_clean_samples must be >= 0 (got {args.external_clean_samples})")

    def _collect_images(roots):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        out = []
        for root in roots:
            for p in root.rglob('*'):
                if not p.is_file():
                    continue
                if p.suffix.lower() in exts:
                    out.append(p)
        return sorted(out)

    real_all = _collect_images(real_roots)
    fake_all = _collect_images(fake_roots)

    requested_ref_half = int(args.ref_samples) // 2
    requested_query = int(args.query_samples)
    need_real = requested_ref_half + requested_query
    need_fake = requested_ref_half + requested_query

    rng = np.random.default_rng(int(args.seed))
    if bool(args.shuffle_paths):
        rng.shuffle(real_all)
        rng.shuffle(fake_all)

    def _is_valid_image(p: Path) -> bool:
        try:
            with Image.open(p) as im:
                im.verify()
            return True
        except (UnidentifiedImageError, OSError, ValueError):
            return False

    def _collect_images_limited(roots, limit: int = 0, validate: bool = False):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        out = []
        bad = 0
        lim = int(limit)
        for root in roots:
            for p in root.rglob('*'):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in exts:
                    continue
                if bool(validate) and not _is_valid_image(p):
                    bad += 1
                    continue
                out.append(p)
                if lim > 0 and len(out) >= lim:
                    return out, bad
        return out, bad

    def _discover_binary_class_roots(base_root: Path):
        direct_real = base_root / '0_real'
        direct_fake = base_root / '1_fake'
        if direct_real.exists() and direct_fake.exists():
            return [direct_real], [direct_fake]
        found_real = sorted([p for p in base_root.rglob('0_real') if p.is_dir()])
        found_fake = sorted([p for p in base_root.rglob('1_fake') if p.is_dir()])
        if len(found_real) > 0 and len(found_fake) > 0:
            return found_real, found_fake
        return None, None

    def _collect_labeled_pairs_from_root(
        root: Path,
        sample_limit: int,
        seed: int,
        shuffle_paths: bool,
        subset_name: str,
        default_label: int,
    ) -> List[Tuple[Path, int]]:
        real_roots_local, fake_roots_local = _discover_binary_class_roots(root)
        if real_roots_local is not None and fake_roots_local is not None:
            requested_total = int(sample_limit)
            requested_even = requested_total if requested_total <= 0 else int(requested_total - (requested_total % 2))
            take_each = int(requested_even // 2) if requested_even > 0 else 0

            real_paths, real_bad = _collect_images_limited(
                real_roots_local,
                limit=take_each,
                validate=bool(args.validate_images),
            )
            fake_paths, fake_bad = _collect_images_limited(
                fake_roots_local,
                limit=take_each,
                validate=bool(args.validate_images),
            )
            if bool(args.validate_images) and (real_bad > 0 or fake_bad > 0):
                print(
                    f"[external-attack] validated {subset_name}: real_valid={int(len(real_paths))} (skipped {int(real_bad)}), "
                    f"fake_valid={int(len(fake_paths))} (skipped {int(fake_bad)})"
                )
            pairs = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
            return _select_balanced_labeled_pairs(
                pairs,
                sample_limit=int(sample_limit),
                seed=int(seed),
                shuffle_paths=bool(shuffle_paths),
                subset_name=subset_name,
            )

        unlabeled_paths, unlabeled_bad = _collect_images_limited(
            [root],
            limit=int(sample_limit),
            validate=bool(args.validate_images),
        )
        if bool(args.validate_images) and unlabeled_bad > 0:
            print(
                f"[external-attack] validated {subset_name}: usable={int(len(unlabeled_paths))} "
                f"(skipped {int(unlabeled_bad)})"
            )
        if len(unlabeled_paths) == 0:
            return []
        print(
            f"[external-attack][warn] {subset_name} has no 0_real/1_fake structure; "
            f"assigning all selected samples to label={int(default_label)}."
        )
        return [(p, int(default_label)) for p in unlabeled_paths]

    if bool(args.validate_images):
        real_valid = []
        fake_valid = []
        real_bad = 0
        fake_bad = 0
        for p in real_all:
            if _is_valid_image(p):
                real_valid.append(p)
                if len(real_valid) >= need_real:
                    break
            else:
                real_bad += 1
        for p in fake_all:
            if _is_valid_image(p):
                fake_valid.append(p)
                if len(fake_valid) >= need_fake:
                    break
            else:
                fake_bad += 1
        real_all = real_valid
        fake_all = fake_valid
        print(f"Validated images: real_valid={len(real_all)} (skipped {real_bad}), fake_valid={len(fake_all)} (skipped {fake_bad})")

    available_real = len(real_all)
    available_fake = len(fake_all)
    if available_real == 0:
        raise ValueError(f"Not enough real images under {data_path}: found 0 (roots={len(real_roots)})")
    if available_fake == 0:
        raise ValueError(f"Not enough fake images under {data_path}: found 0 (roots={len(fake_roots)})")

    total_balanced = min(available_real, available_fake)
    if total_balanced <= 0:
        raise ValueError(
            f"Not enough balanced images under {data_path}: need at least 1 per class after validation, "
            f"found real={available_real}, fake={available_fake} (roots={len(real_roots)}/{len(fake_roots)})"
        )

    if total_balanced >= requested_ref_half + requested_query:
        usable_ref_half = requested_ref_half
        usable_query = requested_query
    else:
        # Keep a reference split, but always reserve some samples for query when possible.
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
            f"[warn] reducing sample request for {data_path}: "
            f"requested_ref_half={requested_ref_half}, requested_query={requested_query}, "
            f"usable_ref_half={usable_ref_half}, usable_query={usable_query}, "
            f"available_real={available_real}, available_fake={available_fake}, total_balanced={total_balanced}"
        )

    real_images = real_all[:usable_ref_half + usable_query]
    fake_images = fake_all[:usable_ref_half + usable_query]
    
    # Split into reference (mixed) and query (balanced real+fake).
    ref_real = real_images[:usable_ref_half]
    ref_fake = fake_images[:usable_ref_half]
    query_real = real_images[usable_ref_half:usable_ref_half + usable_query]
    query_fake = fake_images[usable_ref_half:usable_ref_half + usable_query]
    guide_real = real_images[usable_ref_half + usable_query:usable_ref_half + 2 * max(1, usable_query)]
    if len(guide_real) == 0:
        guide_real = query_real if len(query_real) > 0 else ref_real

    print(
        f"[data] ref_real={len(ref_real)} ref_fake={len(ref_fake)} "
        f"query_real={len(query_real)} query_fake={len(query_fake)} guide_real={len(guide_real)} "
        f"usable_ref_half={usable_ref_half} usable_query={usable_query}"
    )
    if len(ref_real) == 0 or len(ref_fake) == 0:
        raise ValueError(
            f"Reference split missing a class: ref_real={len(ref_real)} ref_fake={len(ref_fake)}. "
            f"Check data_dir layout and --ref_samples (got {int(args.ref_samples)})."
        )
    if len(query_real) == 0 or len(query_fake) == 0:
        print(
            f"[warn] Query split missing samples: query_real={len(query_real)} query_fake={len(query_fake)}. "
            f"Proceeding with reference-only fallback for {data_path}."
        )
    
    # Create datasets
    ref_dataset = ImageDataset(
        ref_real + ref_fake,
        [0] * len(ref_real) + [1] * len(ref_fake),
        transform=transform
    )
    
    clean_dataset = ImageDataset(
        query_real + query_fake,
        [0] * len(query_real) + [1] * len(query_fake),
        transform=transform
    )

    guide_dataset = ImageDataset(
        guide_real,
        [0] * len(guide_real),
        transform=transform
    )
    
    ref_loader = DataLoader(ref_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    clean_loader = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    guide_loader = None
    if str(args.attack_type).lower() in {'stat', 'statattack'}:
        guide_loader = DataLoader(guide_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    ref_labels_np = np.asarray(ref_dataset.labels, dtype=np.int64).reshape(-1)
    clean_labels_np = np.asarray(clean_dataset.labels, dtype=np.int64).reshape(-1)
    u_ref, c_ref = np.unique(ref_labels_np, return_counts=True)
    u_clean, c_clean = np.unique(clean_labels_np, return_counts=True)
    print(f"[data] ref_dataset labels unique={list(zip(u_ref.tolist(), c_ref.tolist()))}")
    print(f"[data] clean_dataset labels unique={list(zip(u_clean.tolist(), c_clean.tolist()))}")

    external_adv_loader = None
    external_attack_metadata = None
    query_dir_raw = '' if getattr(args, 'query_dir', None) is None else str(args.query_dir).strip()
    if query_dir_raw != '':
        query_root = Path(query_dir_raw)
        if not query_root.exists():
            raise ValueError(f"query_dir does not exist: {query_root}")
        query_glob = str(getattr(args, 'query_glob', '*_adv_image.png'))

        query_paths = sorted([p for p in query_root.glob(query_glob) if p.is_file()])
        if bool(getattr(args, 'validate_images', True)):
            query_paths = [p for p in query_paths if _is_valid_image(p)]

        qlim = int(getattr(args, 'query_limit', 0))
        if qlim > 0:
            query_paths = query_paths[:qlim]

        if len(query_paths) == 0:
            raise ValueError(f"No query images found under {query_root} with pattern {query_glob}")

        query_label = int(getattr(args, 'query_label', 1))
        n_adv = int(len(query_paths))
        external_metadata_raw = str(getattr(args, 'external_metadata', '')).strip()
        if external_metadata_raw != '':
            external_clean_raw = str(getattr(args, 'external_clean_dir', '')).strip()
            external_ref_raw = str(getattr(args, 'external_ref_dir', '')).strip()
            if external_clean_raw == '':
                raise ValueError("external_metadata requires --external_clean_dir")

            metadata_map = _load_external_attack_metadata_csv(external_metadata_raw)
            missing_meta = [p.name for p in query_paths if p.name not in metadata_map]
            if len(missing_meta) > 0:
                preview = ', '.join(missing_meta[:5])
                raise ValueError(
                    f"external_metadata is missing {len(missing_meta)} query entries. "
                    f"First missing names: {preview}"
                )

            ordered_meta = [metadata_map[p.name] for p in query_paths]
            query_labels = [_parse_intish(m['primary_label'], default=query_label) for m in ordered_meta]
            if len(query_labels) > 0:
                query_label = int(query_labels[0])

            clean_root = Path(external_clean_raw)
            if not clean_root.exists():
                raise ValueError(f"external_clean_dir does not exist: {clean_root}")

            def _infer_binary_label_from_path(p: Path):
                parts = set([str(x) for x in p.parts])
                if '0_real' in parts:
                    return 0
                if '1_fake' in parts:
                    return 1
                return None

            clean_take = _collect_labeled_pairs_from_root(
                clean_root,
                sample_limit=int(getattr(args, 'external_clean_samples', 0)),
                seed=int(args.seed),
                shuffle_paths=bool(args.shuffle_paths),
                subset_name='external_clean_dir clean set',
                default_label=int(query_label),
            )
            if len(clean_take) == 0:
                raise ValueError(f"No clean images found under external_clean_dir={clean_root}")

            clean_selected = [p for (p, _) in clean_take]
            clean_selected_labels = [int(y) for (_, y) in clean_take]

            unique_query_labels = sorted({int(x) for x in query_labels})
            if len(unique_query_labels) > 1:
                print(
                    f"[external-attack][warn] metadata primary_label is not constant: {unique_query_labels}. "
                    f"adv samples will use per-image labels from metadata; clean/ref labels are inferred from external dirs (0_real/1_fake)."
                )

            ref_root = Path(external_ref_raw) if external_ref_raw != '' else clean_root
            if not ref_root.exists():
                raise ValueError(f"external_ref_dir does not exist: {ref_root}")
            ref_take = _collect_labeled_pairs_from_root(
                ref_root,
                sample_limit=int(args.ref_samples),
                seed=int(args.seed) + 1,
                shuffle_paths=bool(args.shuffle_paths),
                subset_name='external_ref_dir reference set',
                default_label=int(query_label),
            )
            if len(ref_take) == 0:
                raise ValueError(f"No reference images found under external_ref_dir={ref_root}")

            ref_dataset = ImageDataset(
                [str(p) for (p, _) in ref_take],
                [int(y) for (_, y) in ref_take],
                transform=transform,
            )
            ref_loader = DataLoader(ref_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

            clean_dataset = ImageDataset(
                [str(p) for p in clean_selected],
                [int(y) for y in clean_selected_labels],
                transform=transform,
            )
            clean_loader = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            ref_real = [str(p) for (p, y) in ref_take if int(y) == 0]
            ref_fake = [str(p) for (p, y) in ref_take if int(y) == 1]

            adv_dataset = ImageDataset(
                [str(p) for p in query_paths],
                query_labels,
                transform=transform,
            )
            external_adv_loader = DataLoader(adv_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

            external_attack_metadata = {
                'adv_correct_mask': np.asarray([bool(m['adv_correct_gen']) for m in ordered_meta], dtype=np.bool_),
                'attack_success_mask': np.asarray([bool(m['attack_success_gen']) for m in ordered_meta], dtype=np.bool_),
                'attack_failure_mask': np.asarray([bool(m['attack_failure_gen']) for m in ordered_meta], dtype=np.bool_),
                'attack_valid_mask': np.asarray([bool(m['attack_valid_gen']) for m in ordered_meta], dtype=np.bool_),
                'attack_ignored_mask': np.asarray([bool(m['attack_ignored_gen']) for m in ordered_meta], dtype=np.bool_),
            }

            print(
                f"[external-attack] query_dir={str(query_root.resolve())} pattern={query_glob} n={int(n_adv)} "
                f"label={int(query_label)} metadata={str(Path(external_metadata_raw).resolve())}"
            )
            print(
                f"[external-attack] clean_dir={str(clean_root.resolve())} n_clean={int(len(clean_selected))} "
                f"clean_labels={_label_counts_from_pairs(clean_take)} ref_dir={str(ref_root.resolve())} "
                f"n_ref={int(len(ref_take))} ref_labels={_label_counts_from_pairs(ref_take)}"
            )
            if external_ref_raw != '' and str(args.ref_mode) != 'global':
                print(
                    f"[external-attack][warn] external_ref_dir is provided but ref_mode={str(args.ref_mode)}. "
                    f"LID reference will still come from clean_loader unless you set --ref_mode global."
                )
        else:
            clean_pair_paths = []
            missing_pairs = 0
            for p in query_paths:
                name = p.name
                cand = None
                if name.endswith('_adv_image.png'):
                    cand = query_root / (name[:-len('_adv_image.png')] + '_originImage.png')
                if cand is not None and cand.exists():
                    clean_pair_paths.append(cand)
                else:
                    missing_pairs += 1

            if missing_pairs != 0 or len(clean_pair_paths) != n_adv:
                raise ValueError(
                    f"External query requires paired clean images. "
                    f"Expected mapping '*_adv_image.png' -> '*_originImage.png' under {query_root}, "
                    f"but missing {missing_pairs}/{n_adv} pairs."
                )

            print(f"[external-attack] query_dir={str(query_root.resolve())} pattern={query_glob} n={int(n_adv)} label={int(query_label)}")

            clean_dataset = ImageDataset(
                [str(p) for p in clean_pair_paths],
                [query_label] * int(n_adv),
                transform=transform,
            )
            clean_loader = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

            adv_dataset = ImageDataset(
                [str(p) for p in query_paths],
                [query_label] * int(n_adv),
                transform=transform,
            )
            external_adv_loader = DataLoader(adv_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    try:
        _probe_images, _probe_labels = next(iter(clean_loader))
        _probe_labels_np = np.asarray(_probe_labels, dtype=np.int64).reshape(-1)
        u_b, c_b = np.unique(_probe_labels_np, return_counts=True)
        print(f"[data] clean_loader first batch labels unique={list(zip(u_b.tolist(), c_b.tolist()))}")
    except StopIteration:
        raise ValueError("clean_loader is empty")

    # Sanity-check accuracy on a balanced (real+fake) batch.
    try:
        _probe_each_arg = getattr(args, 'sanity_probe_n_each', None)
        if _probe_each_arg is None:
            _n_each = max(1, min(int(args.batch_size) // 2, len(ref_real), len(ref_fake)))
        else:
            _n_each = max(1, min(int(_probe_each_arg), len(ref_real), len(ref_fake)))
        _probe_ds = ImageDataset(
            ref_real[:_n_each] + ref_fake[:_n_each],
            [0] * int(_n_each) + [1] * int(_n_each),
            transform=transform,
        )
        _probe_loader = DataLoader(_probe_ds, batch_size=int(2 * _n_each), shuffle=False, num_workers=0)
        _ref_imgs, _ref_labs = next(iter(_probe_loader))
        _ref_imgs = _ref_imgs.to(device)
        _ref_labs = _ref_labs.to(device)
        with torch.no_grad():
            if str(args.model_type) == 'clip' and hasattr(model, 'apply_input_normalize'):
                _norm_mode = str(getattr(args, 'clip_normalize_mode', 'auto')).lower().strip()
                if bool(getattr(args, 'clip_disable_normalize', False)):
                    _norm_mode = 'off'
                if _norm_mode == 'auto':
                    _prev = bool(getattr(model, 'apply_input_normalize', True))
                    model.apply_input_normalize = True
                    _out_on = model(_ref_imgs)
                    _pred_on = _pred_labels_from_model_output(_out_on)
                    _acc_on = float((_pred_on == _ref_labs).float().mean().item())

                    model.apply_input_normalize = False
                    _out_off = model(_ref_imgs)
                    _pred_off = _pred_labels_from_model_output(_out_off)
                    _acc_off = float((_pred_off == _ref_labs).float().mean().item())

                    if _acc_off > _acc_on:
                        model.apply_input_normalize = False
                        _acc_ref = _acc_off
                        _picked = 'off'
                    else:
                        model.apply_input_normalize = True
                        _acc_ref = _acc_on
                        _picked = 'on'

                    print(
                        f"[sanity] probe accuracy={_acc_ref:.4f} (balanced real+fake batch, n_each={int(_n_each)}, "
                        f"clip_norm_auto picked={_picked}, acc_on={_acc_on:.4f}, acc_off={_acc_off:.4f})"
                    )
                    if _picked != ('on' if _prev else 'off'):
                        args.clip_normalize_mode = _picked
                else:
                    model.apply_input_normalize = (True if _norm_mode == 'on' else False)
                    _out_ref = model(_ref_imgs)
                    _pred_ref = _pred_labels_from_model_output(_out_ref)
                    _acc_ref = float((_pred_ref == _ref_labs).float().mean().item())
                    print(
                        f"[sanity] probe accuracy={_acc_ref:.4f} (balanced real+fake batch, n_each={int(_n_each)}, "
                        f"clip_norm={_norm_mode})"
                    )
            else:
                _out_ref = model(_ref_imgs)
                _pred_ref = _pred_labels_from_model_output(_out_ref)
                _acc_ref = float((_pred_ref == _ref_labs).float().mean().item())
                print(f"[sanity] probe accuracy={_acc_ref:.4f} (balanced real+fake batch, n_each={int(_n_each)})")
    except Exception as _e:
        print(f"[sanity][warn] could not compute balanced probe accuracy: {_e}")

    if int(getattr(args, 'sanity_eval_batches', 0)) > 0:
        try:
            _n_batches = int(getattr(args, 'sanity_eval_batches', 0))
            _total = 0
            _correct = 0
            for _bi, (_imgs, _labs) in enumerate(clean_loader):
                if _bi >= _n_batches:
                    break
                _imgs = _imgs.to(device)
                _labs = _labs.to(device)
                with torch.no_grad():
                    _out = model(_imgs)
                    _pred = _pred_labels_from_model_output(_out)
                _total += int(_labs.numel())
                _correct += int((_pred == _labs).sum().item())
            if _total > 0:
                print(f"[sanity] clean_loader accuracy={float(_correct)/float(_total):.4f} (batches={_n_batches}, n={_total})")
            else:
                print("[sanity][warn] clean_loader accuracy skipped: empty loader")
        except Exception as _e:
            print(f"[sanity][warn] could not compute clean_loader accuracy: {_e}")
    
    # Extract clean LID features
    clean_lids = compute_lid_features(
        model, ref_loader, clean_loader, 'clean', 
        layers,
        k=args.lid_k,
        chunk_size=args.chunk_size,
        device=device,
        feat_norm=('l2' if args.feat_norm == 'l2' else 'none'),
        distance_metric=str(args.lid_distance_metric),
        ref_mode=str(args.ref_mode),
        batch_ref_loader=(clean_loader if args.ref_mode == 'batch_clean' else None),
        batch_ref_size=int(args.batch_ref_size),
        spatial_feat_mode=(str(args.resnet_feat_mode) if args.model_type in ['resnet', 'csf'] else 'flatten'),
        pca_dim=args.pca_dim,
        pca_whiten=args.pca_whiten,
        pca_standardize=args.pca_standardize,
        pca_seed=args.pca_seed,
    )
    print(f"Clean LID features: {clean_lids.shape}")

    calibrator = None
    calibrator_cfg = None
    calibrator_use_logits_input = False
    ckpt_raw = '' if args.calibrator_ckpt is None else str(args.calibrator_ckpt).strip()
    ckpt_norm = '' if ckpt_raw.lower() in {'', 'none', 'null'} else ckpt_raw
    if ckpt_norm != '':
        calibrator, calibrator_cfg = _load_trained_logit_calibrator(
            ckpt_norm,
            lid_dim=int(clean_lids.shape[1]),
            device=device,
        )
        calibrator_use_logits_input = bool(calibrator_cfg.get('use_logits_input', False))
        print(
            f"Loaded calibrator ckpt: {ckpt_norm} "
            f"(use_logits_input={calibrator_use_logits_input})"
        )
        if str(args.attack_target).lower() == 'calibrated':
            print("[attack] Using calibrator-adaptive attack: optimize calibrated output z_corr with fixed clean LID (BPDA-style)")

    if str(args.attack_target).lower() == 'calibrated' and calibrator is None:
        raise ValueError("attack_target='calibrated' requires --calibrator_ckpt")

    if (
        str(args.attack_type).lower() == 'autoattack'
        and str(args.autoattack_success_target).lower() == 'calibrated'
        and calibrator is None
    ):
        raise ValueError("autoattack_success_target='calibrated' requires --calibrator_ckpt")

    if (
        str(args.attack_type).lower() == 'autoattack'
        and str(args.autoattack_success_target).lower() == 'calibrated'
        and str(args.attack_target).lower() != 'calibrated'
    ):
        print(
            "[attack][warn] autoattack_success_target=calibrated only changes final success masks/metrics. "
            "Adversarial generation is still performed against attack_target (currently base), so AutoAttack stage logs "
            "may show near-100% success on the base/surrogate model while calibrated success remains low. "
            "If you want to attack the calibrator itself, set --attack_target calibrated (and optionally --aa_stage_eval_recompute_lid)."
        )
    
    # Generate adversarial samples and extract features
    if external_adv_loader is not None:
        print(f"Using external adversarial samples from query_dir (skip generation).")
    else:
        print(f"Generating {args.attack_type} adversarial samples...")
    adv_images = []
    adv_labels = []

    # Some pretrained binary heads may use an inverted label convention.
    # Probe a batch to decide whether to flip labels for attack generation & success metrics.
    flip_attack_labels = False
    flip_base_labels = False
    try:
        _n_each = max(1, min(int(args.batch_size) // 2, len(ref_real), len(ref_fake)))
        _probe_ds = ImageDataset(
            ref_real[:_n_each] + ref_fake[:_n_each],
            [0] * int(_n_each) + [1] * int(_n_each),
            transform=transform,
        )
        _probe_loader = DataLoader(_probe_ds, batch_size=int(2 * _n_each), shuffle=False, num_workers=0)
        probe_images, probe_labels = next(iter(_probe_loader))
        probe_images = probe_images.to(device)
        probe_labels = probe_labels.to(device)
        with torch.no_grad():
            probe_model = model
            if str(args.attack_target).lower() == 'calibrated' and calibrator is not None:
                bs_probe = int(probe_images.shape[0])
                lid_probe_np = clean_lids[:bs_probe]
                lid_probe_t = torch.from_numpy(np.asarray(lid_probe_np, dtype=np.float32)).to(device)
                empty_ref = {layer: torch.empty((0, 1), dtype=torch.float32) for layer in layers}
                probe_model = CalibratedAAWrapper(
                    base_model=model,
                    calibrator=calibrator,
                    lid=lid_probe_t,
                    use_logits_input=calibrator_use_logits_input,
                    layers=layers,
                    ref_feats_by_layer=empty_ref,
                    k=int(args.lid_k),
                    distance_metric=str(args.lid_distance_metric),
                    feat_norm=('l2' if args.feat_norm == 'l2' else 'none'),
                    spatial_feat_mode=(str(args.resnet_feat_mode) if args.model_type in ['resnet', 'csf'] else 'flatten'),
                    adaptive_lid_recompute_every=0,
                )

            out_probe = probe_model(probe_images)
            pred_probe = _pred_labels_from_model_output(out_probe)
            acc_probe = float((pred_probe == probe_labels).float().mean().item())
            acc_probe_flip = float((pred_probe == (1 - probe_labels)).float().mean().item())
        if acc_probe_flip > acc_probe:
            flip_attack_labels = True
            print(f"Detected inverted label convention for attack metrics (probe acc {acc_probe:.4f} -> {acc_probe_flip:.4f}), flipping labels for attack.")
        else:
            print(f"Attack metrics probe accuracy: {acc_probe:.4f} (no flip)")

        flip_base_labels = bool(flip_attack_labels)
        if str(args.attack_target).lower() == 'calibrated':
            with torch.no_grad():
                out_probe_base = model(probe_images)
                pred_probe_base = _pred_labels_from_model_output(out_probe_base)
                acc_probe_base = float((pred_probe_base == probe_labels).float().mean().item())
                acc_probe_base_flip = float((pred_probe_base == (1 - probe_labels)).float().mean().item())
            if acc_probe_base_flip > acc_probe_base:
                flip_base_labels = True
    except StopIteration:
        pass

    clean_correct_total = 0
    clean_total = 0
    adv_total = 0
    adv_correct_total = 0
    success_total = 0
    failure_total = 0

    wrapped_clean_correct_total = 0
    wrapped_adv_correct_total = 0
    wrapped_success_total = 0
    wrapped_failure_total = 0

    # Track per-sample attack success/failure
    attack_success_mask_list = []
    attack_failure_mask_list = []
    attack_valid_mask_list = []
    attack_ignored_mask_list = []

    clean_logits_list = []
    clean_pred_list = []
    clean_true_labels_list = []
    clean_effective_labels_list = []

    adv_logits_list = []
    adv_pred_list = []

    # Calibrator metrics (evaluated with clean_lids for clean, adv_lids for adversarial later)
    clean_calib_logits = None
    clean_calib_pred = None
    adv_calib_logits = None
    adv_calib_pred = None

    lid_offset = 0
    ref_bank_by_layer = None
    lid_spatial_feat_mode = (str(args.resnet_feat_mode) if args.model_type in ['resnet', 'csf'] else 'flatten')
    if str(args.ref_mode) == 'global':
        ref_bank_by_layer = {layer: [] for layer in layers}
        with torch.no_grad():
            for ref_images, _ in tqdm(ref_loader, desc='ReferenceBank'):
                ref_images = ref_images.to(device)
                ref_layer_feats = model.extract_layer_features(ref_images)
                for layer in layers:
                    if layer not in ref_layer_feats:
                        continue
                    feat = ref_layer_feats[layer]
                    if not torch.is_tensor(feat):
                        feat = torch.as_tensor(feat)
                    if feat.dim() > 2:
                        if str(lid_spatial_feat_mode) == 'avgpool':
                            feat = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)
                        else:
                            feat = feat.reshape(feat.shape[0], -1)
                    else:
                        feat = feat.reshape(feat.shape[0], -1)
                    if str(args.feat_norm) == 'l2':
                        feat = F.normalize(feat, p=2, dim=1)
                    ref_bank_by_layer[layer].append(feat.detach().cpu())

        for layer in layers:
            if len(ref_bank_by_layer[layer]) > 0:
                ref_bank_by_layer[layer] = torch.cat(ref_bank_by_layer[layer], dim=0)
            else:
                ref_bank_by_layer[layer] = torch.empty((0, 1), dtype=torch.float32)

    if bool(args.dynamic_attack):
        args.randomize_attack_params_per_batch = True
        if str(args.eps_schedule).strip() == '':
            args.eps_schedule = f"{2/255},{4/255},{8/255}"
        if str(args.eps_schedule_breaks).strip() == '':
            args.eps_schedule_breaks = "0.3333333333,0.6666666667"
        if str(args.steps_range).strip() == '' and str(args.steps_range_by_attack).strip() == '':
            args.steps_range_by_attack = "pgd:5,10;fab:50,120"
        if str(args.square_queries_range).strip() == '':
            args.square_queries_range = "500,3000"

    steps_range_default = (int(args.steps), int(args.steps))
    steps_range_global = _parse_int_range(str(args.steps_range), steps_range_default)
    steps_range_by_attack = _parse_steps_range_by_attack(str(args.steps_range_by_attack))
    eps_schedule = _parse_float_list(str(args.eps_schedule))
    eps_breaks = _parse_float_list(str(args.eps_schedule_breaks))
    cw_kappa_range = _parse_float_range(str(args.cw_kappa_range), (float(args.cw_kappa), float(args.cw_kappa)))
    square_queries_range = _parse_int_range(str(args.square_queries_range), (int(args.square_queries), int(args.square_queries)))

    per_batch_rng = random.Random(int(args.seed) + 12345)
    loader_steps = None
    try:
        loader_steps = int(len(clean_loader))
    except Exception:
        loader_steps = None

    _last_logged_eps = None
    _last_logged_steps = None
    _last_logged_kappa = None

    if str(args.attack_type).lower() in {'stat', 'statattack'} and guide_loader is None:
        raise RuntimeError('StatAttack requires a guide_loader but it was not constructed')

    class _GuideBatcher:
        def __init__(self, loader: DataLoader):
            self.loader = loader
            self.it = iter(loader)
            self.buf: Optional[torch.Tensor] = None

        def next(self, bs: int, device: torch.device) -> torch.Tensor:
            chunks = []
            if self.buf is not None and int(self.buf.shape[0]) > 0:
                chunks.append(self.buf)
                self.buf = None

            total = int(sum(int(t.shape[0]) for t in chunks))
            while total < int(bs):
                try:
                    x, _ = next(self.it)
                except StopIteration:
                    self.it = iter(self.loader)
                    x, _ = next(self.it)
                chunks.append(x)
                total += int(x.shape[0])

            x_all = torch.cat(chunks, dim=0)
            out = x_all[: int(bs)].to(device)
            rest = x_all[int(bs):]
            if int(rest.shape[0]) > 0:
                self.buf = rest
            return out

    guide_batcher = _GuideBatcher(guide_loader) if guide_loader is not None else None

    def _collect_loader_outputs(data_loader: DataLoader, desc: str, flip_labels: bool) -> Dict[str, object]:
        logits_local = []
        preds_local = []
        labels_local = []
        total_local = 0
        correct_local = 0
        for batch_images, batch_labels in tqdm(data_loader, desc=desc):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            batch_effective_labels = (1 - batch_labels) if flip_labels else batch_labels
            with torch.no_grad():
                batch_out = model(batch_images)
                batch_pred = _pred_labels_from_model_output(batch_out)
            logits_local.append(_model_output_to_numpy_logits(batch_out))
            preds_local.append(batch_pred.detach().cpu().numpy())
            labels_local.append(batch_labels.detach().cpu().numpy())
            total_local += int(batch_labels.numel())
            correct_local += int((batch_pred == batch_effective_labels).sum().item())
        return {
            'logits': np.concatenate(logits_local, axis=0) if len(logits_local) > 0 else None,
            'pred': np.concatenate(preds_local, axis=0) if len(preds_local) > 0 else None,
            'labels': np.concatenate(labels_local, axis=0) if len(labels_local) > 0 else None,
            'total': int(total_local),
            'correct': int(correct_local),
        }

    full_clean_eval = None
    metadata_valid_total = None
    attack_eval_clean_loader = clean_loader
    attack_eval_clean_lids = np.asarray(clean_lids)
    if external_adv_loader is not None and external_attack_metadata is not None:
        print("[external-attack] clean/ref LID uses external_clean_dir, adversarial LID uses query_dir, and attack success masks come from external_metadata.")
        full_clean_eval = _collect_loader_outputs(clean_loader, 'CleanEval', flip_base_labels)
        attack_eval_count = int(np.asarray(external_attack_metadata['attack_success_mask'], dtype=np.bool_).reshape(-1).shape[0])
        if attack_eval_count <= 0:
            raise ValueError('external_metadata produced an empty query set')
        clean_eval_paths = list(clean_dataset.images)
        clean_eval_labels = list(clean_dataset.labels)
        if len(clean_eval_paths) == 0 or len(clean_eval_labels) == 0:
            raise RuntimeError('clean_loader is empty in external metadata mode')
        aligned_clean_paths = [clean_eval_paths[i % len(clean_eval_paths)] for i in range(attack_eval_count)]
        aligned_clean_labels = [clean_eval_labels[i % len(clean_eval_labels)] for i in range(attack_eval_count)]
        attack_eval_clean_dataset = ImageDataset(aligned_clean_paths, aligned_clean_labels, transform=transform)
        attack_eval_clean_loader = DataLoader(attack_eval_clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        reps = int(np.ceil(float(attack_eval_count) / float(max(1, int(attack_eval_clean_lids.shape[0])))))
        attack_eval_clean_lids = np.concatenate([attack_eval_clean_lids] * reps, axis=0)[:attack_eval_count]
        try:
            loader_steps = int(len(attack_eval_clean_loader))
        except Exception:
            pass

    ref_buf_by_layer = {layer: None for layer in layers}
    external_adv_it = iter(external_adv_loader) if external_adv_loader is not None else None
    detector_feature_hook = None
    clean_detector_features_list = []
    adv_detector_features_list = []
    if bool(getattr(args, 'save_detector_features', False)):
        detector_feature_hook = DetectorPenultimateFeatureHook(model, model_type=str(args.model_type))
        print(
            "[detector-features] Saving detector penultimate features "
            f"from final head: {detector_feature_hook.layer.__class__.__name__}"
        )
    for batch_idx, (images, labels) in enumerate(tqdm(attack_eval_clean_loader, desc='Adversarial')):
        images = images.to(device)
        labels = labels.to(device)

        external_adv_batch = None
        if external_adv_it is not None:
            try:
                external_adv_batch = next(external_adv_it)
            except StopIteration:
                raise RuntimeError("external_adv_loader exhausted before clean_loader")

        bs = int(images.shape[0])
        lid_batch_np = attack_eval_clean_lids[lid_offset:lid_offset + bs]
        if int(lid_batch_np.shape[0]) != bs:
            raise RuntimeError(
                f"clean_lids misalignment: expected batch size {bs} at offset {lid_offset}, got {lid_batch_np.shape[0]}"
            )
        lid_offset += bs

        attack_effective_labels = (1 - labels) if flip_attack_labels else labels
        base_effective_labels = (1 - labels) if flip_base_labels else labels

        with torch.no_grad():
            out_clean = model(images)
            pred_clean = _pred_labels_from_model_output(out_clean)
            clean_logits_list.append(_model_output_to_numpy_logits(out_clean))
            if detector_feature_hook is not None:
                clean_detector_features_list.append(detector_feature_hook.pop_numpy(bs, tag='clean'))
            clean_pred_list.append(pred_clean.detach().cpu().numpy())
            clean_true_labels_list.append(labels.detach().cpu().numpy())
            clean_effective_labels_list.append(attack_effective_labels.detach().cpu().numpy())
            clean_correct = (pred_clean == base_effective_labels)
            clean_total += int(labels.numel())
            clean_correct_total += int(clean_correct.sum().item())

        if str(args.ref_mode) == 'batch_clean':
            with torch.no_grad():
                ref_layer_feats = model.extract_layer_features(images)
            for layer in layers:
                if layer not in ref_layer_feats:
                    continue
                ref_feat = ref_layer_feats[layer]
                if not torch.is_tensor(ref_feat):
                    ref_feat = torch.as_tensor(ref_feat)
                if ref_feat.dim() > 2:
                    if str(lid_spatial_feat_mode) == 'avgpool':
                        ref_feat = F.adaptive_avg_pool2d(ref_feat, (1, 1)).squeeze(-1).squeeze(-1)
                    else:
                        ref_feat = ref_feat.reshape(ref_feat.shape[0], -1)
                else:
                    ref_feat = ref_feat.reshape(ref_feat.shape[0], -1)
                if str(args.feat_norm) == 'l2':
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                ref_feat = ref_feat.detach().cpu()

                if ref_buf_by_layer[layer] is None:
                    buf = ref_feat
                else:
                    buf = torch.cat([ref_buf_by_layer[layer], ref_feat], dim=0)
                if int(args.batch_ref_size) > 0 and int(buf.shape[0]) > int(args.batch_ref_size):
                    buf = buf[-int(args.batch_ref_size):]
                ref_buf_by_layer[layer] = buf

        wrapped_model = None
        if str(args.attack_target).lower() == 'calibrated':
            lid_batch_t = torch.from_numpy(np.asarray(lid_batch_np, dtype=np.float32)).to(device)
            if str(args.ref_mode) == 'batch_clean':
                ref_feats_for_lid = {layer: (ref_buf_by_layer[layer] if ref_buf_by_layer[layer] is not None else torch.empty((0, 1), dtype=torch.float32)) for layer in layers}
            else:
                ref_feats_for_lid = ref_bank_by_layer if ref_bank_by_layer is not None else {}
            wrapped_model = CalibratedAAWrapper(
                base_model=model,
                calibrator=calibrator,
                lid=lid_batch_t,
                use_logits_input=calibrator_use_logits_input,
                layers=layers,
                ref_feats_by_layer=ref_feats_for_lid,
                k=int(args.lid_k),
                distance_metric=str(args.lid_distance_metric),
                feat_norm=('l2' if args.feat_norm == 'l2' else 'none'),
                spatial_feat_mode=(str(args.resnet_feat_mode) if args.model_type in ['resnet', 'csf'] else 'flatten'),
                adaptive_lid_recompute_every=int(args.adaptive_lid_recompute_every),
            )

        aa_stage_eval_wrapped_model = None
        if bool(args.aa_stage_eval_recompute_lid) and calibrator is not None:
            lid_batch_t = torch.from_numpy(np.asarray(lid_batch_np, dtype=np.float32)).to(device)
            if str(args.ref_mode) == 'batch_clean':
                ref_feats_for_lid = {layer: (ref_buf_by_layer[layer] if ref_buf_by_layer[layer] is not None else torch.empty((0, 1), dtype=torch.float32)) for layer in layers}
            else:
                ref_feats_for_lid = ref_bank_by_layer if ref_bank_by_layer is not None else {}
            aa_stage_eval_wrapped_model = CalibratedAAWrapper(
                base_model=model,
                calibrator=calibrator,
                lid=lid_batch_t,
                use_logits_input=calibrator_use_logits_input,
                layers=layers,
                ref_feats_by_layer=ref_feats_for_lid,
                k=int(args.lid_k),
                distance_metric=str(args.lid_distance_metric),
                feat_norm=('l2' if args.feat_norm == 'l2' else 'none'),
                spatial_feat_mode=(str(args.resnet_feat_mode) if args.model_type in ['resnet', 'csf'] else 'flatten'),
                adaptive_lid_recompute_every=0,
            )

        eps_cur = float(args.epsilon)
        alpha_cur = float(args.alpha)
        steps_cur = int(args.steps)
        apgd_restarts_cur = int(args.apgd_restarts)
        fab_restarts_cur = int(args.fab_restarts)
        fab_beta_cur = args.fab_beta
        fab_eta_cur = args.fab_eta
        fab_alpha_max_cur = args.fab_alpha_max
        aa_seed_cur = int(args.aa_seed)
        cw_kappa_cur = float(args.cw_kappa)
        square_queries_cur = int(args.square_queries)

        if bool(args.randomize_attack_params_per_batch):
            progress = 0.0
            if loader_steps is not None and int(loader_steps) > 0:
                progress = float(batch_idx) / float(max(1, int(loader_steps)))
            atk = str(args.attack_type).lower()

            eps_attacks = {'fgsm', 'pgd', 'apgd', 'bim', 'fab', 'pixle', 'square', 'autoattack'}
            step_attacks = {'pgd', 'apgd', 'bim', 'pixle', 'fab', 'cw'}
            alpha_attacks = {'pgd', 'bim'}

            if atk in eps_attacks and len(eps_schedule) > 0:
                eps_cur = float(_schedule_eps(progress, eps_schedule, eps_breaks))
            if atk in step_attacks:
                lo, hi = steps_range_by_attack.get(atk, steps_range_global)
                steps_cur = int(per_batch_rng.randint(int(lo), int(hi)))
            if atk in alpha_attacks:
                alpha_cur = float(eps_cur) / float(args.alpha_divisor)

            if atk == 'square':
                q_lo, q_hi = int(square_queries_range[0]), int(square_queries_range[1])
                square_queries_cur = int(per_batch_rng.randint(int(q_lo), int(q_hi)))

            if atk == 'apgd':
                apgd_restarts_cur = int(per_batch_rng.randint(1, 4))
            elif atk == 'fab':
                fab_restarts_cur = int(per_batch_rng.randint(1, 3))
                fab_beta_cur = float(per_batch_rng.uniform(0.8, 1.0))
                fab_eta_cur = float(per_batch_rng.uniform(1.0, 1.8))
                fab_alpha_max_cur = float(per_batch_rng.uniform(0.01, 0.2))
            elif atk == 'cw':
                k_lo, k_hi = float(cw_kappa_range[0]), float(cw_kappa_range[1])
                cw_kappa_cur = float(per_batch_rng.uniform(k_lo, k_hi))
            elif atk == 'autoattack':
                aa_seed_cur = int(per_batch_rng.randint(0, 2**31 - 1))

            should_log = (batch_idx < 3)
            if _last_logged_eps is None or abs(float(eps_cur) - float(_last_logged_eps)) > 1e-12:
                should_log = True
            if _last_logged_steps is None or int(steps_cur) != int(_last_logged_steps):
                should_log = True
            if atk == 'cw' and (_last_logged_kappa is None or abs(float(cw_kappa_cur) - float(_last_logged_kappa)) > 1e-12):
                should_log = True
            if bool(should_log):
                msg = (
                    f"[attack][per-batch] batch={int(batch_idx)} progress={float(progress):.3f} "
                    f"atk={atk} eps={float(eps_cur):.6f} steps={int(steps_cur)} alpha={float(alpha_cur):.6f}"
                )
                if atk == 'square':
                    msg += f" queries={int(square_queries_cur)}"
                if atk == 'apgd':
                    msg += f" restarts={int(apgd_restarts_cur)}"
                if atk == 'fab':
                    msg += f" restarts={int(fab_restarts_cur)}"
                if atk == 'cw':
                    msg += f" kappa={float(cw_kappa_cur):.4f}"
                print(msg)
                _last_logged_eps = float(eps_cur)
                _last_logged_steps = int(steps_cur)
                _last_logged_kappa = float(cw_kappa_cur)

        guided_images = None
        if str(args.attack_type).lower() in {'stat', 'statattack'}:
            guided_images = guide_batcher.next(int(images.shape[0]), device)

        if external_adv_batch is not None:
            adv_imgs, adv_labs = external_adv_batch
            adv_imgs = adv_imgs.to(device)
            adv_labs = adv_labs.to(device)
            if int(adv_imgs.shape[0]) != int(images.shape[0]):
                raise RuntimeError(
                    f"External adv batch size mismatch: clean bs={int(images.shape[0])} adv bs={int(adv_imgs.shape[0])}"
                )
            try:
                if external_attack_metadata is None and int(adv_labs.numel()) == int(labels.numel()):
                    if bool((adv_labs.reshape(-1) != labels.reshape(-1)).any().item()):
                        raise RuntimeError(
                            "External adv labels do not match clean labels. "
                            "For paired evaluation, clean/adv labels must be aligned."
                        )
            except Exception:
                pass
            adv = adv_imgs
        else:
            adv = generate_adversarial(
                model,
                images,
                attack_effective_labels,
                guided_images=guided_images,
                wrapped_model=wrapped_model,
                aa_stage_eval_wrapped_model=aa_stage_eval_wrapped_model,
                attack_type=args.attack_type,
                eps=eps_cur,
                alpha=alpha_cur,
                steps=steps_cur,
                apgd_restarts=apgd_restarts_cur,
                aa_norm=args.aa_norm,
                aa_version=args.aa_version,
                aa_seed=aa_seed_cur,
                aa_backend=args.aa_backend,
                aa_stage_eval_recompute_lid=bool(args.aa_stage_eval_recompute_lid),
                aa_stage_log_recompute_lid=bool(args.aa_stage_log_recompute_lid),
                fab_restarts=fab_restarts_cur,
                fab_beta=fab_beta_cur,
                fab_eta=fab_eta_cur,
                fab_alpha_max=fab_alpha_max_cur,
                pixle_pixels=args.pixle_pixels,
                pixle_restarts=args.pixle_restarts,
                square_queries=square_queries_cur,
                square_restarts=args.square_restarts,
                square_p_init=args.square_p_init,
                cw_c=args.cw_c,
                cw_kappa=cw_kappa_cur,
                cw_lr=args.cw_lr,
                cw_binary_search_steps=args.cw_binary_search_steps,
                device=device
            )

        with torch.no_grad():
            out_adv = model(adv)
            pred_adv = _pred_labels_from_model_output(out_adv)
            adv_logits_list.append(_model_output_to_numpy_logits(out_adv))
            if detector_feature_hook is not None:
                adv_detector_features_list.append(detector_feature_hook.pop_numpy(int(adv.shape[0]), tag='adversarial'))
            adv_pred_list.append(pred_adv.detach().cpu().numpy())
            adv_correct = (pred_adv == base_effective_labels)
            adv_correct_total += int(adv_correct.sum().item())
            
            # Attack success: model was correct on clean but wrong on adversarial
            batch_attack_success = (clean_correct & (~adv_correct))
            batch_attack_failure = (clean_correct & adv_correct)
            batch_attack_valid = clean_correct
            batch_attack_ignored = (~clean_correct)
            success_total += int(batch_attack_success.sum().item())
            failure_total += int(batch_attack_failure.sum().item())
            
            # Track per-sample masks
            attack_success_mask_list.append(batch_attack_success.cpu().numpy())
            attack_failure_mask_list.append(batch_attack_failure.cpu().numpy())
            attack_valid_mask_list.append(batch_attack_valid.cpu().numpy())
            attack_ignored_mask_list.append(batch_attack_ignored.cpu().numpy())

            if wrapped_model is not None:
                out_clean_w = wrapped_model(images)
                pred_clean_w = _pred_labels_from_model_output(out_clean_w)
                clean_ok_w = (pred_clean_w == attack_effective_labels)
                wrapped_clean_correct_total += int(clean_ok_w.sum().item())

                out_adv_w = wrapped_model(adv)
                pred_adv_w = _pred_labels_from_model_output(out_adv_w)
                adv_ok_w = (pred_adv_w == attack_effective_labels)
                wrapped_adv_correct_total += int(adv_ok_w.sum().item())

                succ_w = clean_ok_w & (~adv_ok_w)
                fail_w = clean_ok_w & adv_ok_w
                wrapped_success_total += int(succ_w.sum().item())
                wrapped_failure_total += int(fail_w.sum().item())

        adv_images.append(adv.detach().cpu())
        if external_adv_batch is not None:
            adv_labels.extend(adv_labs.detach().cpu().tolist())
        else:
            adv_labels.extend(labels.detach().cpu().tolist())

    if detector_feature_hook is not None:
        detector_feature_hook.remove()
    
    adv_images = torch.cat(adv_images, dim=0)
    adv_dataset = torch.utils.data.TensorDataset(adv_images, torch.tensor(adv_labels))
    adv_loader = DataLoader(adv_dataset, batch_size=args.batch_size, shuffle=False)
    adv_total = int(len(adv_labels))
    
    # Concatenate attack success/failure masks
    attack_success_mask = np.concatenate(attack_success_mask_list, axis=0)
    attack_failure_mask = np.concatenate(attack_failure_mask_list, axis=0)
    attack_valid_mask = np.concatenate(attack_valid_mask_list, axis=0)
    attack_ignored_mask = np.concatenate(attack_ignored_mask_list, axis=0)

    clean_model_logits = np.concatenate(clean_logits_list, axis=0) if len(clean_logits_list) > 0 else None
    clean_model_pred = np.concatenate(clean_pred_list, axis=0) if len(clean_pred_list) > 0 else None
    clean_true_labels = np.concatenate(clean_true_labels_list, axis=0) if len(clean_true_labels_list) > 0 else None
    clean_effective_labels = np.concatenate(clean_effective_labels_list, axis=0) if len(clean_effective_labels_list) > 0 else None
    clean_detector_features = (
        np.concatenate(clean_detector_features_list, axis=0)
        if len(clean_detector_features_list) > 0
        else np.empty((0, 0), dtype=np.float32)
    )

    adv_model_logits = np.concatenate(adv_logits_list, axis=0) if len(adv_logits_list) > 0 else None
    adv_model_pred = np.concatenate(adv_pred_list, axis=0) if len(adv_pred_list) > 0 else None
    adv_detector_features = (
        np.concatenate(adv_detector_features_list, axis=0)
        if len(adv_detector_features_list) > 0
        else np.empty((0, 0), dtype=np.float32)
    )

    if full_clean_eval is not None:
        clean_model_logits = full_clean_eval['logits']
        clean_model_pred = full_clean_eval['pred']
        clean_true_labels = full_clean_eval['labels']
        if clean_true_labels is None:
            clean_effective_labels = None
        else:
            clean_true_labels_np = np.asarray(clean_true_labels, dtype=np.int64).reshape(-1)
            clean_effective_labels = (1 - clean_true_labels_np) if flip_attack_labels else clean_true_labels_np
        clean_total = int(full_clean_eval['total'])
        clean_correct_total = int(full_clean_eval['correct'])

    def _describe_binary_labels(name: str, y: np.ndarray) -> None:
        if y is None:
            print(f"[labels] {name}: None")
            return
        y = np.asarray(y).reshape(-1)
        u, c = np.unique(y, return_counts=True)
        pairs = list(zip(u.tolist(), c.tolist()))
        print(f"[labels] {name}: n={int(len(y))} unique={pairs}")

    _describe_binary_labels('clean_true_labels', clean_true_labels)
    _describe_binary_labels('clean_effective_labels', clean_effective_labels)
    _describe_binary_labels('adv_true_labels', np.asarray(adv_labels, dtype=np.int64) if len(adv_labels) > 0 else None)
    print(f"[labels] flip_attack_labels={bool(flip_attack_labels)}")
    print(f"[labels] flip_base_labels={bool(flip_base_labels)}")

    if clean_true_labels is not None and int(len(np.unique(clean_true_labels))) < 2:
        pass
    if clean_effective_labels is not None and int(len(np.unique(clean_effective_labels))) < 2:
        pass
    
    if str(args.attack_type).lower() not in {'stat', 'statattack'} and external_attack_metadata is None:
        n_attack_success = int(attack_success_mask.sum())
        n_attack_failure = int(attack_failure_mask.sum())
        n_attack_valid = int(attack_valid_mask.sum())
        n_attack_ignored = int(attack_ignored_mask.sum())
        print(f"Base attack successful samples: {n_attack_success}")
        print(f"Base attack failed samples: {n_attack_failure}")
        print(f"Base attack valid samples (clean correct): {n_attack_valid}")
        print(f"Base attack ignored samples (clean incorrect): {n_attack_ignored}")

    _print_logits_stats('Clean', clean_model_logits)
    _print_logits_stats('Adversarial', adv_model_logits)

    adv_effective_labels = None
    if len(adv_labels) > 0:
        adv_labels_np = np.asarray(adv_labels, dtype=np.int64).reshape(-1)
        adv_effective_labels = (1 - adv_labels_np) if flip_attack_labels else adv_labels_np

    adv_batch_ref_loader = None
    if str(args.ref_mode) == 'batch_clean':
        adv_batch_ref_loader = attack_eval_clean_loader if external_attack_metadata is not None else clean_loader
    
    adv_lids = compute_lid_features(
        model, ref_loader, adv_loader, 'adversarial',
        layers,
        k=args.lid_k,
        chunk_size=args.chunk_size,
        device=device,
        feat_norm=('l2' if args.feat_norm == 'l2' else 'none'),
        distance_metric=str(args.lid_distance_metric),
        ref_mode=str(args.ref_mode),
        batch_ref_loader=adv_batch_ref_loader,
        batch_ref_size=int(args.batch_ref_size),
        spatial_feat_mode=(str(args.resnet_feat_mode) if args.model_type in ['resnet', 'csf'] else 'flatten'),
        pca_dim=args.pca_dim,
        pca_whiten=args.pca_whiten,
        pca_standardize=args.pca_standardize,
        pca_seed=args.pca_seed,
    )
    print(f"Adversarial LID features: {adv_lids.shape}")


    if bool(args.debug_attack_calibrated_postcheck) and str(args.attack_target).lower() == 'calibrated' and calibrator is not None and external_attack_metadata is None:
        try:
            if clean_model_logits is not None and adv_model_logits is not None and clean_effective_labels is not None:
                y_eff = np.asarray(clean_effective_labels, dtype=np.int64).reshape(-1)
                z_clean = _numpy_logits_to_z(clean_model_logits)
                z_adv = _numpy_logits_to_z(adv_model_logits)

                def _calib_pred(lid_np: np.ndarray, z_np: np.ndarray) -> np.ndarray:
                    z_t = torch.from_numpy(np.asarray(z_np, dtype=np.float32)).to(device)
                    outs = []
                    with torch.no_grad():
                        for i in range(0, int(z_t.shape[0]), int(args.batch_size)):
                            lid_b = torch.from_numpy(np.asarray(lid_np[i:i + int(args.batch_size)], dtype=np.float32)).to(device)
                            z_b = z_t[i:i + int(args.batch_size)]
                            out = calibrator(lid_b, z_b, use_logits_input=calibrator_use_logits_input)
                            outs.append(out['z_corr'].detach().cpu().numpy().reshape(-1))
                    z_corr = np.concatenate(outs, axis=0)
                    return (z_corr > 0).astype(np.int64)

                clean_pred_cal = _calib_pred(clean_lids, z_clean)
                adv_pred_cal = _calib_pred(adv_lids, z_adv)
                valid = (clean_pred_cal == y_eff)
                succ = valid & (adv_pred_cal != y_eff)
                print(
                    "[debug][calibrated-postcheck] "
                    f"valid={int(valid.sum())}/{int(len(valid))} succ={int(succ.sum())} "
                    f"succ_rate={float(succ.sum() / max(1, int(valid.sum()))):.4f} "
                    "(uses recomputed adv LID; should match 'Calibrator attack success rate' / final masks)"
                )
        except Exception as e:
            print(f"[debug][calibrated-postcheck][warn] failed: {e}")

    if calibrator is not None and clean_model_logits is not None and adv_model_logits is not None:
        def _batched_calibrate(lid_np: np.ndarray, logits_np: np.ndarray) -> np.ndarray:
            z_np = _numpy_logits_to_z(logits_np)
            z_t = torch.from_numpy(np.asarray(z_np, dtype=np.float32)).to(device)
            outs = []
            with torch.no_grad():
                for i in range(0, int(z_t.shape[0]), int(args.batch_size)):
                    lid_b = torch.from_numpy(np.asarray(lid_np[i:i + int(args.batch_size)], dtype=np.float32)).to(device)
                    z_b = z_t[i:i + int(args.batch_size)]
                    out = calibrator(lid_b, z_b, use_logits_input=calibrator_use_logits_input)
                    outs.append(out['z_corr'].detach().cpu().numpy())
            return np.concatenate(outs, axis=0)

        clean_calib_logits = _batched_calibrate(clean_lids, clean_model_logits)
        adv_calib_logits = _batched_calibrate(adv_lids, adv_model_logits)
        clean_calib_pred = (np.asarray(clean_calib_logits).reshape(-1) > 0).astype(np.int64)
        adv_calib_pred = (np.asarray(adv_calib_logits).reshape(-1) > 0).astype(np.int64)

        if clean_effective_labels is not None:
            y_eff_clean = np.asarray(clean_effective_labels, dtype=np.int64).reshape(-1)
            clean_acc_cal = float((clean_calib_pred == y_eff_clean).mean())
            y_eff_adv = None
            adv_acc_cal = float('nan')
            if adv_effective_labels is not None and len(np.asarray(adv_effective_labels).reshape(-1)) == len(np.asarray(adv_calib_pred).reshape(-1)):
                y_eff_adv = np.asarray(adv_effective_labels, dtype=np.int64).reshape(-1)
                adv_acc_cal = float((adv_calib_pred == y_eff_adv).mean())
            print(f"Calibrator accuracy (clean): {clean_acc_cal:.4f}")
            print(f"Calibrator accuracy (adv): {adv_acc_cal:.4f}")

            use_calib_masks = (str(args.attack_target).lower() == 'calibrated') or (
                str(args.attack_type).lower() == 'autoattack' and str(args.autoattack_success_target).lower() == 'calibrated'
            )

            if external_attack_metadata is None and y_eff_adv is not None and len(y_eff_clean) == len(y_eff_adv):
                valid_cal = (clean_calib_pred == y_eff_clean)
                succ_cal = valid_cal & (adv_calib_pred != y_eff_clean)
                succ_rate_cal = float(succ_cal.sum() / max(1, valid_cal.sum()))
                print(f"Calibrator attack success rate: {succ_rate_cal:.4f} (among calibrator-clean-correct samples)")

                if bool(use_calib_masks):
                    attack_valid_mask = np.asarray(valid_cal, dtype=bool).reshape(-1)
                    attack_success_mask = np.asarray(succ_cal, dtype=bool).reshape(-1)
                    attack_failure_mask = np.asarray(valid_cal & (adv_calib_pred == y_eff_clean), dtype=bool).reshape(-1)
                    attack_ignored_mask = np.asarray(~valid_cal, dtype=bool).reshape(-1)

                    tag = 'attack_target=calibrated' if (str(args.attack_target).lower() == 'calibrated') else 'autoattack_success_target=calibrated'
                    print(
                        f"[attack] Using calibrator-defined success masks ({tag}): "
                        f"succ={int(attack_success_mask.sum())} "
                        f"fail={int(attack_failure_mask.sum())} "
                        f"valid={int(attack_valid_mask.sum())} "
                        f"ignored={int(attack_ignored_mask.sum())}"
                    )
            elif bool(use_calib_masks) and external_attack_metadata is not None:
                print("[attack] Skipping calibrator-defined success masks because external_metadata defines attack outcomes for the adversarial query set.")

    # StatAttack only perturbs fake samples. Exclude un-attacked real samples from metrics and saved npz.
    stat_keep_mask = None
    if str(args.attack_type).lower() in {'stat', 'statattack'}:
        if clean_true_labels is None:
            raise RuntimeError('StatAttack filtering requires clean_true_labels')
        stat_keep_mask = (np.asarray(clean_true_labels, dtype=np.int64).reshape(-1) == 1)
        if int(stat_keep_mask.sum()) <= 0:
            raise ValueError('StatAttack produced no attacked fake samples (keep_mask sum == 0).')

        def _mask_np(x):
            if x is None:
                return None
            x = np.asarray(x)
            return x[stat_keep_mask]

        # Filter per-sample arrays/masks
        clean_model_logits = _mask_np(clean_model_logits)
        clean_model_pred = _mask_np(clean_model_pred)
        clean_true_labels = _mask_np(clean_true_labels)
        clean_effective_labels = _mask_np(clean_effective_labels)

        adv_model_logits = _mask_np(adv_model_logits)
        adv_model_pred = _mask_np(adv_model_pred)
        adv_labels = np.asarray(adv_labels, dtype=np.int64).reshape(-1)[stat_keep_mask].tolist()

        attack_success_mask = np.asarray(attack_success_mask, dtype=bool).reshape(-1)[stat_keep_mask]
        attack_failure_mask = np.asarray(attack_failure_mask, dtype=bool).reshape(-1)[stat_keep_mask]
        attack_valid_mask = np.asarray(attack_valid_mask, dtype=bool).reshape(-1)[stat_keep_mask]
        attack_ignored_mask = np.asarray(attack_ignored_mask, dtype=bool).reshape(-1)[stat_keep_mask]

        clean_lids = np.asarray(clean_lids)[stat_keep_mask]
        adv_lids = np.asarray(adv_lids)[stat_keep_mask]
        if clean_detector_features is not None and np.asarray(clean_detector_features).shape[0] > 0:
            clean_detector_features = np.asarray(clean_detector_features)[stat_keep_mask]
        if adv_detector_features is not None and np.asarray(adv_detector_features).shape[0] > 0:
            adv_detector_features = np.asarray(adv_detector_features)[stat_keep_mask]

        if clean_calib_logits is not None and np.asarray(clean_calib_logits).shape[0] > 0:
            clean_calib_logits = np.asarray(clean_calib_logits)[stat_keep_mask]
        if adv_calib_logits is not None and np.asarray(adv_calib_logits).shape[0] > 0:
            adv_calib_logits = np.asarray(adv_calib_logits)[stat_keep_mask]
        if clean_calib_pred is not None and np.asarray(clean_calib_pred).shape[0] > 0:
            clean_calib_pred = np.asarray(clean_calib_pred)[stat_keep_mask]
        if adv_calib_pred is not None and np.asarray(adv_calib_pred).shape[0] > 0:
            adv_calib_pred = np.asarray(adv_calib_pred)[stat_keep_mask]

        # Rebuild adv_images tensor for later noisy generation / saving consistency.
        try:
            km_t = torch.from_numpy(stat_keep_mask.astype(np.bool_))
            adv_images = adv_images[km_t]
        except Exception:
            pass

        # Recompute base-model metrics on the filtered subset.
        if clean_model_pred is not None and adv_model_pred is not None and clean_true_labels is not None:
            y_base = np.asarray(clean_true_labels, dtype=np.int64).reshape(-1)
            if bool(flip_base_labels):
                y_base = 1 - y_base
            pred_c = np.asarray(clean_model_pred).reshape(-1).astype(np.int64)
            pred_a = np.asarray(adv_model_pred).reshape(-1).astype(np.int64)
            clean_ok = (pred_c == y_base)
            adv_ok = (pred_a == y_base)
            clean_total = int(len(y_base))
            clean_correct_total = int(clean_ok.sum())
            adv_correct_total = int(adv_ok.sum())
            success_total = int((clean_ok & (~adv_ok)).sum())
            failure_total = int((clean_ok & adv_ok).sum())

            # If attack_target is base, use base-model success definition; otherwise keep the (possibly calibrator) masks.
            if str(args.attack_target).lower() != 'calibrated' and external_attack_metadata is None:
                attack_valid_mask = clean_ok
                attack_success_mask = clean_ok & (~adv_ok)
                attack_failure_mask = clean_ok & adv_ok
                attack_ignored_mask = ~clean_ok

    if external_attack_metadata is not None:
        attack_success_mask = np.asarray(external_attack_metadata['attack_success_mask'], dtype=np.bool_).reshape(-1)
        attack_failure_mask = np.asarray(external_attack_metadata['attack_failure_mask'], dtype=np.bool_).reshape(-1)
        attack_valid_mask = np.asarray(external_attack_metadata['attack_valid_mask'], dtype=np.bool_).reshape(-1)
        attack_ignored_mask = np.asarray(external_attack_metadata['attack_ignored_mask'], dtype=np.bool_).reshape(-1)
        adv_correct_mask_meta = np.asarray(external_attack_metadata['adv_correct_mask'], dtype=np.bool_).reshape(-1)
        metadata_valid_total = int(attack_valid_mask.sum())
        adv_total = max(int(adv_total), int(attack_success_mask.shape[0]))
        adv_correct_total = int(adv_correct_mask_meta.sum())
        success_total = int(attack_success_mask.sum())
        failure_total = int(attack_failure_mask.sum())
        print(
            f"[attack] Using metadata-defined success masks: succ={int(success_total)} total={int(attack_success_mask.shape[0])}"
        )

    base_clean_acc = (clean_correct_total / clean_total) if clean_total > 0 else float('nan')
    base_adv_acc = (adv_correct_total / adv_total) if adv_total > 0 else float('nan')
    if metadata_valid_total is not None:
        base_attack_success_rate = (success_total / metadata_valid_total) if metadata_valid_total > 0 else float('nan')
    else:
        base_attack_success_rate = (success_total / clean_correct_total) if clean_correct_total > 0 else float('nan')

    if stat_keep_mask is not None:
        n_attack_success = int(np.asarray(attack_success_mask, dtype=bool).sum())
        n_attack_failure = int(np.asarray(attack_failure_mask, dtype=bool).sum())
        n_attack_valid = int(np.asarray(attack_valid_mask, dtype=bool).sum())
        n_attack_ignored = int(np.asarray(attack_ignored_mask, dtype=bool).sum())
        print(
            f"[stat] Filtered to attacked fake samples only: n={int(clean_total)} "
            f"succ={n_attack_success} fail={n_attack_failure} valid={n_attack_valid} ignored={n_attack_ignored}"
        )

    clean_acc = base_clean_acc
    adv_acc = base_adv_acc
    attack_success_rate = base_attack_success_rate

    if str(args.attack_target).lower() == 'calibrated':
        if attack_valid_mask is not None and attack_success_mask is not None:
            try:
                n_valid = int(np.asarray(attack_valid_mask, dtype=bool).sum())
                n_succ = int(np.asarray(attack_success_mask, dtype=bool).sum())
                attack_success_rate = float(n_succ / max(1, n_valid))
            except Exception:
                pass

        if clean_effective_labels is not None and clean_calib_pred is not None and adv_calib_pred is not None:
            try:
                y_eff_clean = np.asarray(clean_effective_labels, dtype=np.int64).reshape(-1)
                clean_acc = float((np.asarray(clean_calib_pred).reshape(-1) == y_eff_clean).mean())
                if adv_effective_labels is not None and len(np.asarray(adv_effective_labels).reshape(-1)) == len(np.asarray(adv_calib_pred).reshape(-1)):
                    y_eff_adv = np.asarray(adv_effective_labels, dtype=np.int64).reshape(-1)
                    adv_acc = float((np.asarray(adv_calib_pred).reshape(-1) == y_eff_adv).mean())
            except Exception:
                pass

        print(f"Base clean accuracy (before attack): {base_clean_acc:.4f}")
        print(f"Base accuracy on adversarial samples: {base_adv_acc:.4f}")
        print(f"Base attack success rate: {base_attack_success_rate:.4f}")
        wrapped_eval_total = int(adv_total) if external_attack_metadata is not None else int(clean_total)
        if wrapped_eval_total > 0:
            wrapped_clean_acc = float(wrapped_clean_correct_total / max(1, wrapped_eval_total))
            wrapped_adv_acc = float(wrapped_adv_correct_total / max(1, wrapped_eval_total))
        else:
            wrapped_clean_acc = float('nan')
            wrapped_adv_acc = float('nan')
        if wrapped_clean_correct_total > 0:
            wrapped_succ_rate = float(wrapped_success_total / max(1, int(wrapped_clean_correct_total)))
        else:
            wrapped_succ_rate = float('nan')
        print(f"Surrogate calibrated clean accuracy (fixed clean LID): {wrapped_clean_acc:.4f}")
        print(f"Surrogate calibrated adv accuracy (fixed clean LID): {wrapped_adv_acc:.4f}")
        print(f"Surrogate calibrated attack success rate (fixed clean LID): {wrapped_succ_rate:.4f}")
        print(f"Calibrated clean accuracy (before attack): {clean_acc:.4f}")
        print(f"Calibrated accuracy on adversarial samples: {adv_acc:.4f}")
        print(f"Calibrated attack success rate: {attack_success_rate:.4f}")
    else:
        print(f"Clean accuracy (before attack): {clean_acc:.4f}")
        print(f"Accuracy on adversarial samples: {adv_acc:.4f}")
        print(f"Attack success rate: {attack_success_rate:.4f}")

    adv_success_lids = adv_lids[attack_success_mask]
    adv_failure_lids = adv_lids[attack_failure_mask]
    adv_ignored_lids = adv_lids[attack_ignored_mask]
    
    # Generate noisy samples if requested
    noisy_lids = None
    noisy_model_logits = None
    noisy_model_pred = None
    noisy_true_labels = None
    if args.add_noisy:
        print(f"Generating noisy samples...")
        
        # Compute L2 norm of adversarial perturbations
        clean_images = []
        l2_clean_loader = attack_eval_clean_loader if external_attack_metadata is not None else clean_loader
        for images, _ in l2_clean_loader:
            clean_images.append(images)
        clean_images = torch.cat(clean_images, dim=0)
        if stat_keep_mask is not None:
            _km_t = torch.from_numpy(stat_keep_mask.astype(np.bool_))
            clean_images = clean_images[_km_t]
        
        l2_diffs = torch.norm((adv_images - clean_images).view(len(clean_images), -1), dim=1)
        avg_l2 = l2_diffs.mean().item()
        print(f"Average L2 perturbation of adversarial samples: {avg_l2:.4f}")
        
        # Generate noisy samples with matching L2
        noisy_images = []
        noisy_labels = []
        noisy_logits_list = []
        noisy_pred_list = []
        
        with torch.no_grad():
            for images, labels in clean_loader:
                images = images.to(device)
                noisy = generate_noisy(images, match_l2=avg_l2, device=device)
                out_noisy = model(noisy)
                pred_noisy = _pred_labels_from_model_output(out_noisy)
                noisy_logits_list.append(_model_output_to_numpy_logits(out_noisy))
                noisy_pred_list.append(pred_noisy.detach().cpu().numpy())
                noisy_images.append(noisy.cpu())
                noisy_labels.extend(labels.tolist())
        
        noisy_images = torch.cat(noisy_images, dim=0)

        noisy_dataset = torch.utils.data.TensorDataset(noisy_images, torch.tensor(noisy_labels))
        noisy_loader = DataLoader(noisy_dataset, batch_size=args.batch_size, shuffle=False)

        noisy_model_logits = np.concatenate(noisy_logits_list, axis=0) if len(noisy_logits_list) > 0 else None
        noisy_model_pred = np.concatenate(noisy_pred_list, axis=0) if len(noisy_pred_list) > 0 else None
        noisy_true_labels = np.asarray(noisy_labels, dtype=np.int64)

        _print_logits_stats('Noisy', noisy_model_logits)
        
        noisy_lids = compute_lid_features(
            model, ref_loader, noisy_loader, 'noisy',
            layers,
            k=args.lid_k,
            chunk_size=args.chunk_size,
            device=device,
            feat_norm=('l2' if args.feat_norm == 'l2' else 'none'),
            distance_metric=str(args.lid_distance_metric),
            ref_mode=str(args.ref_mode),
            batch_ref_loader=(clean_loader if args.ref_mode == 'batch_clean' else None),
            batch_ref_size=int(args.batch_ref_size),
            spatial_feat_mode=(str(args.resnet_feat_mode) if args.model_type in ['resnet', 'csf'] else 'flatten'),
            pca_dim=args.pca_dim,
            pca_whiten=args.pca_whiten,
            pca_standardize=args.pca_standardize,
            pca_seed=args.pca_seed,
        )
        print(f"Noisy LID features: {noisy_lids.shape}")

        if stat_keep_mask is not None:
            noisy_model_logits = np.asarray(noisy_model_logits)[stat_keep_mask] if noisy_model_logits is not None else None
            noisy_model_pred = np.asarray(noisy_model_pred)[stat_keep_mask] if noisy_model_pred is not None else None
            noisy_true_labels = np.asarray(noisy_true_labels)[stat_keep_mask] if noisy_true_labels is not None else None
            noisy_lids = np.asarray(noisy_lids)[stat_keep_mask] if noisy_lids is not None else None
    
    # Save features
    output_file = output_dir / f"lid_{args.exp_name}.npz"
    print(f"[save] output_file={str(output_file.resolve())}")
    
    # Prepare labels: 0 for negative (clean/noisy), 1 for positive (adversarial)
    if args.add_noisy:
        # Negative class: clean + noisy
        neg_features = np.vstack([clean_lids, noisy_lids])
        neg_labels = np.zeros(len(neg_features))
    else:
        # Negative class: clean only
        neg_features = clean_lids
        neg_labels = np.zeros(len(neg_features))
    
    # Positive class: adversarial
    pos_features = adv_lids
    pos_labels = np.ones(len(pos_features))
    
    # Combine
    features = np.vstack([neg_features, pos_features])
    labels = np.hstack([neg_labels, pos_labels])
    
    # Save
    if noisy_model_logits is None:
        noisy_model_logits = np.empty((0, 1), dtype=np.float32)
    if noisy_model_pred is None:
        noisy_model_pred = np.empty((0,), dtype=np.int64)
    if noisy_true_labels is None:
        noisy_true_labels = np.empty((0,), dtype=np.int64)

    if clean_calib_logits is None:
        clean_calib_logits = np.empty((0, 1), dtype=np.float32)
    if adv_calib_logits is None:
        adv_calib_logits = np.empty((0, 1), dtype=np.float32)
    if clean_calib_pred is None:
        clean_calib_pred = np.empty((0,), dtype=np.int64)
    if adv_calib_pred is None:
        adv_calib_pred = np.empty((0,), dtype=np.int64)

    np.savez(
        output_file,
        features=features,
        labels=labels,
        clean_model_logits=clean_model_logits,
        adv_model_logits=adv_model_logits,
        noisy_model_logits=noisy_model_logits,
        clean_calib_logits=clean_calib_logits,
        adv_calib_logits=adv_calib_logits,
        clean_detector_features=clean_detector_features,
        adv_detector_features=adv_detector_features,
        clean_model_pred=clean_model_pred,
        adv_model_pred=adv_model_pred,
        noisy_model_pred=noisy_model_pred,
        clean_calib_pred=clean_calib_pred,
        adv_calib_pred=adv_calib_pred,
        clean_true_labels=clean_true_labels,
        adv_true_labels=np.asarray(adv_labels, dtype=np.int64),
        noisy_true_labels=noisy_true_labels,
        clean_effective_labels=clean_effective_labels,
        layers=layers,
        model_type=args.model_type,
        attack_type=args.attack_type,
        attack_target=str(args.attack_target),
        calibrator_ckpt=str(args.calibrator_ckpt),
        epsilon=args.epsilon,
        lid_k=args.lid_k,
        feat_norm=str(args.feat_norm),
        lid_distance_metric=str(args.lid_distance_metric),
        ref_mode=str(args.ref_mode),
        batch_ref_size=int(args.batch_ref_size),
        include_input=bool(args.include_input),
        save_detector_features=bool(getattr(args, 'save_detector_features', False)),
        resnet_feat_mode=str(args.resnet_feat_mode),
        clip_feat_mode=str(args.clip_feat_mode),
        pca_dim=args.pca_dim,
        pca_whiten=bool(args.pca_whiten),
        pca_standardize=bool(args.pca_standardize),
        pca_seed=int(args.pca_seed),
        add_noisy=args.add_noisy,
        base_clean_acc=base_clean_acc,
        base_adv_acc=base_adv_acc,
        base_attack_success_rate=base_attack_success_rate,
        external_success_source=('metadata' if external_attack_metadata is not None else ('paired' if external_adv_loader is not None else 'recomputed')),
        external_metadata_path=str(getattr(args, 'external_metadata', '')),
        clean_acc=clean_acc,
        adv_acc=adv_acc,
        attack_success_rate=attack_success_rate,
        flip_attack_labels=flip_attack_labels,
        attack_success_mask=attack_success_mask,
        attack_failure_mask=attack_failure_mask,
        attack_valid_mask=attack_valid_mask,
        attack_ignored_mask=attack_ignored_mask,
        adv_success_features=adv_success_lids,
        adv_failure_features=adv_failure_lids,
        adv_ignored_features=adv_ignored_lids,
        n_neg_samples=len(neg_features),  # number of negative samples (clean + noisy if add_noisy)
    )
    
    print(f"\nSaved LID features to {output_file}")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    if bool(getattr(args, 'save_detector_features', False)):
        print(f"Clean detector features shape: {np.asarray(clean_detector_features).shape}")
        print(f"Adv detector features shape: {np.asarray(adv_detector_features).shape}")
    print(f"Negative samples: {np.sum(labels == 0)}")
    print(f"Positive samples: {np.sum(labels == 1)}")
    
    # Print per-layer statistics
    print("\nPer-layer LID statistics:")
    print(f"{'Layer':<10} {'Clean Mean':<12} {'Clean Std':<12} {'Adv Mean':<12} {'Adv Std':<12}")
    print("-" * 60)
    for i, layer in enumerate(layers):
        clean_mean = np.nanmean(clean_lids[:, i])
        clean_std = np.nanstd(clean_lids[:, i])
        adv_mean = np.nanmean(adv_lids[:, i])
        adv_std = np.nanstd(adv_lids[:, i])
        print(f"{layer:<10} {clean_mean:<12.4f} {clean_std:<12.4f} {adv_mean:<12.4f} {adv_std:<12.4f}")


if __name__ == '__main__':
    main()
