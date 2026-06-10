#!/usr/bin/env python3
"""Unified LID detection pipeline.

This file preserves the original two scripts and provides a single entrypoint
that can run the full pipeline end-to-end:
1) extract LID features
2) train / evaluate the logit calibrator

The implementation keeps the original extraction and training scripts intact
and orchestrates them. By default, extraction uses dynamic PGD/FAB/Square
adversarial generation with multiple perturbation sizes before calibrator
training.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parent
EXTRACT_SCRIPT = ROOT / "extract_lid_features.py"
CALIB_SCRIPT = ROOT / "train_logit_calibrator.py"


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def _extend_kv(cmd: List[str], key: str, value: Optional[object]) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(key)
        return
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return
        cmd.extend([key, ",".join(str(x) for x in value)])
        return
    text = str(value)
    if text == "":
        return
    cmd.extend([key, text])


def _run(cmd: List[str]) -> None:
    printable = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"[run] {printable}")
    subprocess.run(cmd, check=True)


def _parse_attack_list(value: str) -> List[str]:
    attacks = [x.strip().lower() for x in str(value).split(",") if x.strip()]
    if len(attacks) == 0:
        raise ValueError("--attacks must contain at least one attack when --attack_type is empty")
    return attacks


def _float_tag(value: float) -> str:
    raw = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return raw.replace(".", "p")


def _model_tag(args: argparse.Namespace) -> str:
    if str(args.model_type) == "clip":
        return f"clip_{str(args.clip_model).replace('/', '-')}"
    return str(args.model_type)


def _default_exp_name(args: argparse.Namespace, attack: str) -> str:
    base = _model_tag(args)
    eps = _float_tag(float(args.epsilon))
    suffix = "dyn" if bool(args.dynamic_attack) else "static"
    if attack == "pgd":
        alpha = _float_tag(float(args.alpha))
        return f"{base}_{attack}_eps{eps}_a{alpha}_s{int(args.steps)}_k{int(args.lid_k)}_{suffix}"
    if attack == "fab":
        beta = _float_tag(float(args.fab_beta))
        alpha_max = _float_tag(float(args.fab_alpha_max))
        eta = _float_tag(float(args.fab_eta))
        return f"{base}_{attack}_eps{eps}_it{int(args.fab_steps)}_b{beta}_am{alpha_max}_eta{eta}_k{int(args.lid_k)}_{suffix}"
    if attack == "square":
        p_init = _float_tag(float(args.square_p_init))
        return f"{base}_{attack}_eps{eps}_q{int(args.square_queries)}_p{p_init}_k{int(args.lid_k)}_{suffix}"
    return f"{base}_{attack}_eps{eps}_k{int(args.lid_k)}_{suffix}"


def _extract_args_for_attack(args: argparse.Namespace, attack: str) -> argparse.Namespace:
    out = argparse.Namespace(**vars(args))
    out.attack_type = str(attack)

    # extract_lid_features.py uses the generic --steps argument for both PGD and FAB.
    if str(attack) == "fab":
        out.steps = int(args.fab_steps)
    else:
        out.steps = int(args.steps)

    if str(args.exp_name or "").strip():
        out.exp_name = f"{args.exp_name}_{attack}"
    else:
        out.exp_name = _default_exp_name(out, attack)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified LID extraction + calibrator training pipeline")

    p.add_argument("--stage", type=str, default="all", choices=["extract", "train", "all"],
                   help="Which part of the pipeline to run")

    # Shared / passthrough config
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./lid_pipeline_outputs")

    # Extraction-related passthrough
    p.add_argument("--model_type", type=str, default="clip", choices=["resnet", "clip", "forgelens", "gram", "csf"])
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--clip_model", type=str, default="ViT-L/14")
    p.add_argument("--fc_weights", type=str, default="./UniversalFakeDetect-main/pretrained_weights/fc_weights.pth")
    p.add_argument("--forgelens_stage", type=int, default=2)
    p.add_argument("--forgelens_feature_set", type=str, default="all_proj")
    p.add_argument("--forgelens_wsgm_count", type=int, default=4)
    p.add_argument("--forgelens_wsgm_reduction_factor", type=int, default=4)
    p.add_argument("--forgelens_faformer_layers", type=int, default=2)
    p.add_argument("--forgelens_faformer_reduction_factor", type=int, default=1)
    p.add_argument("--forgelens_faformer_head", type=int, default=2)
    p.add_argument("--include_input", action="store_true")
    p.add_argument("--no_include_input", action="store_false", dest="include_input")
    p.set_defaults(include_input=True)
    p.add_argument("--layers", type=str, default=None)
    p.add_argument("--resnet_feat_mode", type=str, default="flatten")
    p.add_argument("--clip_feat_mode", type=str, default="flatten_tokens")
    p.add_argument("--clip_load_size", type=int, default=256)
    p.add_argument("--clip_crop_size", type=int, default=224)
    p.add_argument("--clip_no_resize", action="store_true")
    p.add_argument("--clip_resize_mode", type=str, default="short_side")
    p.add_argument("--clip_disable_normalize", action="store_true")
    p.add_argument("--clip_normalize_mode", type=str, default="auto")
    p.add_argument("--img_load_size", type=int, default=224)
    p.add_argument("--img_crop_size", type=int, default=224)
    p.add_argument("--img_resize_mode", type=str, default="short_side")
    p.add_argument("--sanity_probe_n_each", type=int, default=None)
    p.add_argument("--sanity_eval_batches", type=int, default=0)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ref_samples", type=int, default=1000)
    p.add_argument("--query_samples", type=int, default=5000)
    p.add_argument("--query_dir", type=str, default="")
    p.add_argument("--query_glob", type=str, default="*_adv_image.png")
    p.add_argument("--query_label", type=int, default=1)
    p.add_argument("--query_limit", type=int, default=0)
    p.add_argument("--external_metadata", type=str, default="")
    p.add_argument("--external_clean_dir", type=str, default="")
    p.add_argument("--external_clean_samples", type=int, default=0)
    p.add_argument("--external_ref_dir", type=str, default="")
    p.add_argument("--no_shuffle_paths", action="store_false", dest="shuffle_paths")
    p.set_defaults(shuffle_paths=True)
    p.add_argument("--no_validate_images", action="store_false", dest="validate_images")
    p.set_defaults(validate_images=True)
    p.add_argument("--ref_mode", type=str, default="batch_clean")
    p.add_argument("--batch_ref_size", type=int, default=100)
    p.add_argument(
        "--attack_type",
        type=str,
        default="",
        help="Run a single attack if set. If empty, --attacks is used.",
    )
    p.add_argument(
        "--attacks",
        type=str,
        default="pgd,fab,square",
        help="Comma-separated attacks for extraction when --attack_type is empty.",
    )
    p.add_argument("--attack_target", type=str, default="base")
    p.add_argument("--epsilon", type=float, default=8/255)
    p.add_argument("--alpha", type=float, default=2/255)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--apgd_restarts", type=int, default=1)
    p.add_argument("--aa_norm", type=str, default="Linf")
    p.add_argument("--aa_version", type=str, default="standard")
    p.add_argument("--aa_seed", type=int, default=0)
    p.add_argument("--aa_backend", type=str, default="torchattacks")
    p.add_argument("--aa_stage_eval_recompute_lid", action="store_true")
    p.add_argument("--aa_stage_log_recompute_lid", action="store_true")
    p.add_argument("--autoattack_success_target", type=str, default="base")
    p.add_argument("--adaptive_lid_recompute_every", type=int, default=0)
    p.add_argument("--debug_attack_calibrated_postcheck", action="store_true")
    p.add_argument("--fab_steps", type=int, default=101)
    p.add_argument("--fab_restarts", type=int, default=1)
    p.add_argument("--fab_beta", type=float, default=0.9)
    p.add_argument("--fab_eta", type=float, default=1.3)
    p.add_argument("--fab_alpha_max", type=float, default=0.05)
    p.add_argument("--pixle_pixels", type=int, default=1)
    p.add_argument("--pixle_restarts", type=int, default=1)
    p.add_argument("--square_queries", type=int, default=10000)
    p.add_argument("--square_restarts", type=int, default=1)
    p.add_argument("--square_p_init", type=float, default=0.05)
    p.add_argument("--square_queries_range", type=str, default="3000,10000")
    p.add_argument("--cw_c", type=float, default=1.0)
    p.add_argument("--cw_kappa", type=float, default=0.0)
    p.add_argument("--cw_lr", type=float, default=0.01)
    p.add_argument("--cw_binary_search_steps", type=int, default=None)
    p.add_argument("--add_noisy", action="store_true")
    p.set_defaults(add_noisy=True)
    p.add_argument("--no_add_noisy", action="store_false", dest="add_noisy")
    p.add_argument("--dynamic_attack", action="store_true")
    p.add_argument("--no_dynamic_attack", action="store_false", dest="dynamic_attack")
    p.set_defaults(dynamic_attack=True)
    p.add_argument("--randomize_attack_params_per_batch", action="store_true")
    p.add_argument("--eps_schedule", type=str, default="0.00784313725490196,0.01568627450980392,0.03137254901960784")
    p.add_argument("--eps_schedule_breaks", type=str, default="0.3333333333,0.6666666667")
    p.add_argument("--steps_range", type=str, default="")
    p.add_argument("--steps_range_by_attack", type=str, default="pgd:5,10;fab:50,120")
    p.add_argument("--alpha_divisor", type=float, default=5.0)
    p.add_argument("--cw_kappa_range", type=str, default="")
    p.add_argument("--lid_k", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--feat_norm", type=str, default="none")
    p.add_argument("--lid_distance_metric", type=str, default="euclidean")
    p.add_argument("--pca_dim", type=int, default=0)
    p.add_argument("--pca_whiten", action="store_true")
    p.add_argument("--pca_standardize", action="store_true")
    p.add_argument("--pca_seed", type=int, default=42)
    p.add_argument("--exp_name", type=str, default=None)

    # Training-related passthrough
    p.add_argument("--npz_dir", type=str, default=None, help="Directory containing generated LID npz files")
    p.add_argument(
        "--feature_dir",
        type=str,
        default=None,
        help="Directory containing extracted features/npz files. Defaults to <output_dir>/lid_features.",
    )
    p.add_argument("--held_out_attack", type=str, default="")
    p.add_argument("--eval_all_attacks", action="store_true")
    p.set_defaults(eval_all_attacks=True)
    p.add_argument("--loao", action="store_true")
    p.add_argument("--load_model_path", type=str, default="")
    p.add_argument("--save_model_path", type=str, default="")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_test", action="store_true")
    p.add_argument("--method", type=str, default="residual")
    p.add_argument("--input_mode", type=str, default="auto")
    p.add_argument("--use_logits_input", action="store_true")
    p.set_defaults(use_logits_input=True)
    p.add_argument("--struct_mode", type=str, default="auto")
    p.add_argument("--use_gate", action="store_true")
    p.set_defaults(use_gate=True)
    p.add_argument("--gate_use_logits_input", action="store_true")
    p.set_defaults(gate_use_logits_input=True)
    p.add_argument("--gate_target", type=str, default="is_success")
    p.add_argument("--gate_weight_valid_mask", action="store_true")
    p.set_defaults(gate_weight_valid_mask=True)
    p.add_argument("--epsilon_gate", type=float, default=0.05)
    p.add_argument("--lambda_gate", type=float, default=0.5)
    p.add_argument("--gate_pos_weight", type=float, default=5.0)
    p.add_argument("--use_flip_expert", action="store_true")
    p.set_defaults(use_flip_expert=True)
    p.add_argument("--flip_init_scale", type=float, default=1.0)
    p.add_argument("--flip_init_bias", type=float, default=0.0)
    p.add_argument("--success_weight", type=float, default=3.0)
    p.add_argument("--lambda_id", type=float, default=2.0)
    p.add_argument("--lambda_kd", type=float, default=1.0)
    p.add_argument("--lambda_delta", type=float, default=0.0)
    p.add_argument("--lambda_fail_id", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log_every_epochs", type=int, default=1)
    p.add_argument("--log_every_steps", type=int, default=0)
    p.add_argument("--ema_decay", type=float, default=0.0)
    p.add_argument("--lr_schedule", type=str, default="constant")
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lr_warmup_ratio", type=float, default=0.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--output_csv", type=str, default="")

    return p


def build_extract_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [sys.executable, str(EXTRACT_SCRIPT)]
    direct = [
        "model_type", "model_path", "clip_model", "fc_weights",
        "forgelens_stage", "forgelens_feature_set", "forgelens_wsgm_count",
        "forgelens_wsgm_reduction_factor", "forgelens_faformer_layers",
        "forgelens_faformer_reduction_factor", "forgelens_faformer_head",
        "layers", "resnet_feat_mode", "clip_feat_mode", "clip_load_size",
        "clip_crop_size", "clip_resize_mode", "clip_normalize_mode", "img_load_size", "img_crop_size",
        "img_resize_mode", "data_dir", "ref_samples", "query_samples",
        "query_dir", "query_glob", "query_label", "query_limit",
        "external_metadata", "external_clean_dir", "external_clean_samples",
        "external_ref_dir", "ref_mode", "batch_ref_size", "attack_type",
        "attack_target", "epsilon", "alpha", "steps", "apgd_restarts",
        "aa_norm", "aa_version", "aa_seed", "aa_backend", "fab_restarts",
        "fab_beta", "fab_eta", "fab_alpha_max", "pixle_pixels",
        "pixle_restarts", "square_queries", "square_restarts", "square_p_init",
        "square_queries_range", "cw_c", "cw_kappa", "cw_lr", "cw_binary_search_steps",
        "eps_schedule", "eps_schedule_breaks", "steps_range", "steps_range_by_attack",
        "alpha_divisor", "cw_kappa_range", "lid_k", "batch_size", "chunk_size",
        "feat_norm", "lid_distance_metric", "pca_dim", "pca_seed", "exp_name",
        "output_dir", "device", "sanity_eval_batches",
    ]
    for name in direct:
        _extend_kv(cmd, f"--{name}", getattr(args, name, None))

    for flag in [
        "include_input", "clip_no_resize", "clip_disable_normalize",
        "add_noisy", "dynamic_attack", "randomize_attack_params_per_batch",
        "pca_whiten", "pca_standardize", "sanity_probe_n_each",
    ]:
        val = getattr(args, flag, None)
        if isinstance(val, bool):
            if val:
                cmd.append(f"--{flag}")
        elif val is not None:
            _extend_kv(cmd, f"--{flag}", val)

    if not bool(args.validate_images):
        cmd.append("--no_validate_images")
    if not bool(args.shuffle_paths):
        cmd.append("--no_shuffle_paths")

    return cmd


def build_train_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [sys.executable, str(CALIB_SCRIPT)]

    feature_dir = args.feature_dir
    if feature_dir is None or str(feature_dir).strip() == "":
        if args.npz_dir is not None and str(args.npz_dir).strip() != "":
            feature_dir = args.npz_dir
        else:
            # extract_lid_features.py writes npz files directly under --output_dir
            feature_dir = str(Path(args.output_dir))

    feature_path = Path(feature_dir).resolve()
    feature_path.mkdir(parents=True, exist_ok=True)
    cmd.extend(["--feature_dir", str(feature_path)])
    cmd.extend(["--output_dir", str(Path(args.output_dir).resolve())])

    direct = [
        "held_out_attack", "eval_all_attacks", "loao",
        "load_model_path", "save_model_path", "skip_train", "skip_test", "method", "input_mode",
        "use_logits_input", "struct_mode", "use_gate", "gate_use_logits_input",
        "gate_target", "gate_weight_valid_mask", "epsilon_gate", "lambda_gate",
        "gate_pos_weight", "use_flip_expert", "flip_init_scale", "flip_init_bias",
        "success_weight", "lambda_id", "lambda_kd", "lambda_delta", "lambda_fail_id",
        "epochs", "lr", "log_every_epochs", "log_every_steps", "ema_decay",
        "lr_schedule", "grad_clip", "weight_decay", "lr_warmup_ratio",
        "label_smoothing", "train_frac", "split_seed", "output_csv", "device",
    ]
    for name in direct:
        val = getattr(args, name)
        if isinstance(val, bool):
            if val:
                cmd.append(f"--{name}")
        else:
            _extend_kv(cmd, f"--{name}", val)
    return cmd


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    feature_dir = (
        Path(args.feature_dir).resolve()
        if args.feature_dir
        else (Path(args.output_dir).resolve() / "lid_features")
    )

    if args.stage in {"extract", "all"}:
        if not EXTRACT_SCRIPT.exists():
            raise FileNotFoundError(f"Missing script: {EXTRACT_SCRIPT}")

        if str(args.attack_type).strip():
            attacks = [str(args.attack_type).strip().lower()]
        else:
            attacks = _parse_attack_list(args.attacks)

        print(
            "[info] extraction attacks: "
            f"{attacks} | dynamic_attack={bool(args.dynamic_attack)} | "
            f"eps_schedule={args.eps_schedule} | steps_range_by_attack={args.steps_range_by_attack} | "
            f"square_queries_range={args.square_queries_range}"
        )

        for attack in attacks:
            extract_args = _extract_args_for_attack(args, attack)
            extract_args.output_dir = str(feature_dir)
            out_npz = feature_dir / f"lid_{extract_args.exp_name}.npz"
            if out_npz.exists():
                print(f"[skip] existing npz for attack={attack}: {out_npz}")
                continue
            extract_cmd = build_extract_cmd(extract_args)
            _run(extract_cmd)

        npz_files = sorted(feature_dir.glob("*.npz"))
        if len(npz_files) == 0:
            raise FileNotFoundError(
                f"No .npz files were created in {feature_dir}. "
                f"Please check the extraction logs and output_dir settings."
            )
        print(f"[info] found {len(npz_files)} npz files under {feature_dir}")
        if args.stage == "extract":
            return

    if args.stage in {"train", "all"}:
        if not CALIB_SCRIPT.exists():
            raise FileNotFoundError(f"Missing script: {CALIB_SCRIPT}")

        train_args = argparse.Namespace(**vars(args))
        train_args.feature_dir = str(feature_dir)
        train_args.skip_train = False
        train_args.load_model_path = ""
        if str(train_args.save_model_path).strip() == "":
            train_args.save_model_path = str(Path(train_args.output_dir).resolve() / "seen_model.pt")
        train_cmd = build_train_cmd(train_args)
        _run(train_cmd)

        test_args = argparse.Namespace(**vars(args))
        test_args.feature_dir = str(feature_dir)
        test_args.skip_train = True
        if str(test_args.load_model_path).strip() == "":
            test_args.load_model_path = str(Path(test_args.output_dir).resolve() / "seen_model.pt")
        test_cmd = build_train_cmd(test_args)
        _run(test_cmd)


if __name__ == "__main__":
    main()
