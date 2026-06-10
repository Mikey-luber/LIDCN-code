#!/usr/bin/env python3
"""Grad-CAM visualization for CNNSpot clean/adversarial samples.

This script loads a local CNNSpot ResNet-50 checkpoint, samples clean images
from a binary AIGI-style dataset (`0_real` / `1_fake`), generates APGD, PGD,
FAB and Square adversarial samples, and saves side-by-side Grad-CAM visualizations:

    Clean | APGD | PGD | FAB | Square

The attack implementation is reused from `extract_lid_features.py` so attacks
behave consistently with the rest of this repository.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm import tqdm

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from extract_lid_features import generate_adversarial
from networks.resnet import resnet50

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


class Normalize(nn.Module):
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


class NormalizedModel(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalize(x))


class ImagePathDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[Path, int]], transform):
        self.pairs = list(pairs)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        path, label = self.pairs[int(idx)]
        with Image.open(path) as img:
            img = img.convert('RGB')
            x = self.transform(img)
        return x, torch.tensor(int(label), dtype=torch.long), str(path)


def _clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state.items():
        k = str(k)
        if k.startswith('module.'):
            k = k[len('module.'):]
        if k.startswith('model.'):
            k = k[len('model.'):]
        if k.startswith('backbone.'):
            k = k[len('backbone.'):]
        if k.startswith('0.'):
            # Drop Normalize branch from nn.Sequential(Normalize, backbone).
            continue
        if k.startswith('1.'):
            k = k[2:]
        cleaned[k] = v
    return cleaned


def _extract_state_dict(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if not isinstance(obj, dict):
        raise RuntimeError(f'Unsupported checkpoint object: {type(obj)}')
    for key in ['model', 'state_dict', 'model_state_dict', 'net', 'network']:
        inner = obj.get(key, None)
        if isinstance(inner, nn.Module):
            return inner.state_dict()
        if isinstance(inner, dict):
            return inner
    return obj


def _infer_num_classes(state: Dict[str, torch.Tensor]) -> int:
    for key in ['fc.weight', 'model.fc.weight', '1.fc.weight', 'module.fc.weight', 'module.1.fc.weight']:
        w = state.get(key, None)
        if torch.is_tensor(w) and w.dim() == 2:
            return int(w.shape[0])
    for k, v in state.items():
        if str(k).endswith('fc.weight') and torch.is_tensor(v) and v.dim() == 2:
            return int(v.shape[0])
    return 1


def load_cnnspot(model_path: str, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    checkpoint = torch.load(str(model_path), map_location='cpu')
    raw_state = _extract_state_dict(checkpoint)
    num_classes = _infer_num_classes(raw_state)
    base = resnet50(num_classes=int(num_classes))
    state = _clean_state_dict(raw_state)
    result = base.load_state_dict(state, strict=False)
    print(
        f"[model] loaded {model_path} num_classes={num_classes} "
        f"missing={len(getattr(result, 'missing_keys', []))} "
        f"unexpected={len(getattr(result, 'unexpected_keys', []))}"
    )
    base.eval()
    wrapped = NormalizedModel(base).to(device).eval()
    return wrapped, base


def discover_binary_roots(data_dir: Path) -> Tuple[List[Path], List[Path]]:
    direct_real = data_dir / '0_real'
    direct_fake = data_dir / '1_fake'
    if direct_real.is_dir() and direct_fake.is_dir():
        return [direct_real], [direct_fake]
    real_roots = sorted([p for p in data_dir.rglob('0_real') if p.is_dir()])
    fake_roots = sorted([p for p in data_dir.rglob('1_fake') if p.is_dir()])
    if not real_roots or not fake_roots:
        raise ValueError(f"Cannot find 0_real/1_fake under {data_dir}")
    return real_roots, fake_roots


def collect_images(roots: Sequence[Path]) -> List[Path]:
    out = []
    for root in roots:
        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.append(p)
    return sorted(out)


def build_pairs(data_dir: Path, num_each: int, classes: str, seed: int) -> List[Tuple[Path, int]]:
    real_roots, fake_roots = discover_binary_roots(data_dir)
    rng = np.random.default_rng(int(seed))
    real = collect_images(real_roots)
    fake = collect_images(fake_roots)
    rng.shuffle(real)
    rng.shuffle(fake)

    classes = str(classes).lower()
    pairs: List[Tuple[Path, int]] = []
    if classes in {'both', 'real'}:
        take = len(real) if int(num_each) <= 0 else min(int(num_each), len(real))
        pairs.extend([(p, 0) for p in real[:take]])
    if classes in {'both', 'fake'}:
        take = len(fake) if int(num_each) <= 0 else min(int(num_each), len(fake))
        pairs.extend([(p, 1) for p in fake[:take]])
    if len(pairs) == 0:
        raise ValueError(f"No images selected from {data_dir}")
    return pairs


def logits_to_pred(logits: torch.Tensor) -> torch.Tensor:
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if logits.dim() == 1:
        return (logits > 0).long()
    if logits.dim() == 2 and logits.shape[1] == 1:
        return (logits[:, 0] > 0).long()
    return torch.argmax(logits, dim=1).long()


def target_scores(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    target = target.long().view(-1)
    if logits.dim() == 1:
        z = logits.view(-1)
        return torch.where(target == 1, z, -z)
    if logits.dim() == 2 and logits.shape[1] == 1:
        z = logits[:, 0]
        return torch.where(target == 1, z, -z)
    return logits.gather(1, target.view(-1, 1)).view(-1)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self):
        for h in self.handles:
            h.remove()

    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)
        scores = target_scores(logits, targets)
        scores.sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")
        acts = self.activations.detach()
        grads = self.gradients.detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cams = (weights * acts).sum(dim=1)
        cams = F.relu(cams)
        cams = F.interpolate(cams.unsqueeze(1), size=images.shape[-2:], mode='bilinear', align_corners=False)
        cams = cams[:, 0]

        cams_flat = cams.flatten(1)
        mins = cams_flat.min(dim=1).values.view(-1, 1, 1)
        maxs = cams_flat.max(dim=1).values.view(-1, 1, 1)
        cams = (cams - mins) / (maxs - mins + 1e-8)
        return cams.detach().cpu().numpy()


def tensor_to_rgb_uint8(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]))
    return cv2.addWeighted(rgb, 1.0 - float(alpha), heat, float(alpha), 0)


def save_cam_artifacts(
    rgb: np.ndarray,
    cam: np.ndarray,
    output_prefix: Path,
    alpha: float,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_cam(rgb, cam, alpha=alpha)
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]))

    Image.fromarray(rgb).save(str(output_prefix) + '_image.png')
    Image.fromarray(heat).save(str(output_prefix) + '_heatmap.png')
    Image.fromarray(overlay).save(str(output_prefix) + '_overlay.png')
    np.save(str(output_prefix) + '_cam.npy', cam.astype(np.float32))

    for name, arr in [('image', rgb), ('heatmap', heat), ('overlay', overlay)]:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(arr)
        ax.axis('off')
        fig.tight_layout(pad=0)
        fig.savefig(str(output_prefix) + f'_{name}.pdf', bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def save_summary_figure(
    panels: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    title: str,
    out_path: Path,
    alpha: float,
) -> None:
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6), squeeze=False)
    for j, (name, rgb, cam) in enumerate(panels):
        axes[0, j].imshow(rgb)
        axes[0, j].set_title(f'{name} image')
        axes[0, j].axis('off')
        axes[1, j].imshow(overlay_cam(rgb, cam, alpha=alpha))
        axes[1, j].set_title(f'{name} Grad-CAM')
        axes[1, j].axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description='CNNSpot clean/APGD/PGD/FAB/Square Grad-CAM visualization')
    p.add_argument('--model_path', type=str, required=True, help='Local CNNSpot checkpoint path')
    p.add_argument('--data_dir', type=str, required=True, help='Dataset with 0_real/1_fake folders')
    p.add_argument('--output_dir', type=str, default='./gradcam_cnnspot_adv')
    p.add_argument('--num_each', type=int, default=4, help='Number of real and fake samples. <=0 means all.')
    p.add_argument('--classes', type=str, default='both', choices=['both', 'real', 'fake'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=1, help='Grad-CAM is saved per sample; 1 is recommended.')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--resize_size', type=int, default=256)
    p.add_argument('--layer', type=str, default='layer4', choices=['layer1', 'layer2', 'layer3', 'layer4'])
    p.add_argument('--target_mode', type=str, default='pred', choices=['pred', 'label', 'fake', 'real'])
    p.add_argument('--overlay_alpha', type=float, default=0.40)

    p.add_argument('--epsilon', type=float, default=8 / 255)
    p.add_argument('--alpha', type=float, default=2 / 255)
    p.add_argument('--apgd_steps', type=int, default=20)
    p.add_argument('--apgd_restarts', type=int, default=1)
    p.add_argument('--pgd_steps', type=int, default=20)
    p.add_argument('--fab_steps', type=int, default=20)
    p.add_argument('--fab_restarts', type=int, default=1)
    p.add_argument('--square_queries', type=int, default=500)
    p.add_argument('--square_restarts', type=int, default=1)
    p.add_argument('--square_p_init', type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, base_model = load_cnnspot(args.model_path, device=device)
    target_layer = getattr(base_model, str(args.layer))

    transform = T.Compose([
        T.Resize(int(args.resize_size)),
        T.CenterCrop(int(args.image_size)),
        T.ToTensor(),
    ])
    pairs = build_pairs(Path(args.data_dir), args.num_each, args.classes, args.seed)
    loader = DataLoader(ImagePathDataset(pairs, transform), batch_size=int(args.batch_size), shuffle=False)

    gradcam = GradCAM(model, target_layer)
    summary_path = out_dir / 'summary.tsv'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('index\tpath\tlabel\tclean_pred\tapgd_pred\tpgd_pred\tfab_pred\tsquare_pred\tsummary\tindividual_dir\n')

        global_idx = 0
        for images, labels, paths in tqdm(loader, desc='GradCAM'):
            images = images.to(device)
            labels = labels.to(device)

            variants: Dict[str, torch.Tensor] = {
                'clean': images.detach(),
                'apgd': generate_adversarial(
                    model, images, labels,
                    attack_type='apgd',
                    eps=float(args.epsilon),
                    steps=int(args.apgd_steps),
                    apgd_restarts=int(args.apgd_restarts),
                    device=device,
                ).detach(),
                'pgd': generate_adversarial(
                    model, images, labels,
                    attack_type='pgd',
                    eps=float(args.epsilon),
                    alpha=float(args.alpha),
                    steps=int(args.pgd_steps),
                    device=device,
                ).detach(),
                'fab': generate_adversarial(
                    model, images, labels,
                    attack_type='fab',
                    eps=float(args.epsilon),
                    steps=int(args.fab_steps),
                    fab_restarts=int(args.fab_restarts),
                    device=device,
                ).detach(),
                'square': generate_adversarial(
                    model, images, labels,
                    attack_type='square',
                    eps=float(args.epsilon),
                    square_queries=int(args.square_queries),
                    square_restarts=int(args.square_restarts),
                    square_p_init=args.square_p_init,
                    device=device,
                ).detach(),
            }

            with torch.no_grad():
                preds = {
                    name: logits_to_pred(model(x))
                    for name, x in variants.items()
                }

            if args.target_mode == 'pred':
                targets = {name: pred for name, pred in preds.items()}
            elif args.target_mode == 'label':
                targets = {name: labels for name in variants.keys()}
            elif args.target_mode == 'fake':
                targets = {name: torch.ones_like(labels) for name in variants.keys()}
            else:
                targets = {name: torch.zeros_like(labels) for name in variants.keys()}

            cams = {
                name: gradcam(x, targets[name])
                for name, x in variants.items()
            }

            for i in range(int(images.shape[0])):
                src = Path(paths[i])
                label_i = int(labels[i].detach().cpu().item())
                stem = src.stem
                pred_i = {name: int(preds[name][i].detach().cpu().item()) for name in variants.keys()}
                out_name = (
                    f'{global_idx:04d}_{stem}_label{label_i}'
                    f'_clean{pred_i["clean"]}_apgd{pred_i["apgd"]}_pgd{pred_i["pgd"]}'
                    f'_fab{pred_i["fab"]}_square{pred_i["square"]}.png'
                )
                out_path = out_dir / out_name
                indiv_dir = out_dir / 'individual' / f'{global_idx:04d}_{stem}_label{label_i}'

                title = (
                    f'{src.name} | label={label_i} '
                    f'clean={pred_i["clean"]} apgd={pred_i["apgd"]} pgd={pred_i["pgd"]} '
                    f'fab={pred_i["fab"]} square={pred_i["square"]} '
                    f'target_mode={args.target_mode}'
                )

                panel_order = ['clean', 'apgd', 'pgd', 'fab', 'square']
                panel_titles = {
                    'clean': 'Clean',
                    'apgd': 'APGD',
                    'pgd': 'PGD',
                    'fab': 'FAB',
                    'square': 'Square',
                }
                panels = []
                for name in panel_order:
                    rgb = tensor_to_rgb_uint8(variants[name][i])
                    cam = cams[name][i]
                    panels.append((panel_titles[name], rgb, cam))
                    save_cam_artifacts(
                        rgb,
                        cam,
                        indiv_dir / name,
                        alpha=float(args.overlay_alpha),
                    )

                save_summary_figure(
                    panels,
                    title,
                    out_path,
                    alpha=float(args.overlay_alpha),
                )
                f.write(
                    f'{global_idx}\t{src}\t{label_i}\t{pred_i["clean"]}\t{pred_i["apgd"]}\t'
                    f'{pred_i["pgd"]}\t{pred_i["fab"]}\t{pred_i["square"]}\t{out_path}\t{indiv_dir}\n'
                )
                global_idx += 1

    gradcam.close()
    print(f'[done] saved Grad-CAM figures to: {out_dir}')
    print(f'[done] summary: {summary_path}')


if __name__ == '__main__':
    main()
