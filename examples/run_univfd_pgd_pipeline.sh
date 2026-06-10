#!/usr/bin/env bash
set -euo pipefail

# Example: extract dynamic LID features for UnivFD/CLIP under PGD/FAB/Square
# and train/evaluate a residual gated logit calibrator.
#
# Edit these paths before running.
DATA_DIR=${DATA_DIR:-/path/to/dataset/test}
UNIVFD_CKPT=${UNIVFD_CKPT:-/path/to/univfd_checkpoint.pth}
OUT_DIR=${OUT_DIR:-./example_outputs_univfd}
QUERY_SAMPLES=${QUERY_SAMPLES:-500}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[error] DATA_DIR does not exist: $DATA_DIR" >&2
  echo "Set DATA_DIR to a dataset root containing 0_real and 1_fake." >&2
  exit 1
fi

if [[ ! -f "$UNIVFD_CKPT" ]]; then
  echo "[error] UNIVFD_CKPT does not exist: $UNIVFD_CKPT" >&2
  exit 1
fi

python "$PROJECT_ROOT/lid_detection_pipeline.py" \
  --stage all \
  --model_type clip \
  --clip_model ViT-L/14 \
  --fc_weights "$UNIVFD_CKPT" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR" \
  --exp_name univfd_dynamic_example \
  --query_samples "$QUERY_SAMPLES" \
  --query_samples 2500