#!/usr/bin/env bash
set -euo pipefail

# Example: extract dynamic LID features for CNNSpot under PGD/FAB/Square and
# train/evaluate a residual gated logit calibrator.
#
# Edit these paths before running.
DATA_DIR=${DATA_DIR:-/path/to/dataset/test}
CNNSPOT_CKPT=${CNNSPOT_CKPT:-/path/to/cnnspot.pth}
OUT_DIR=${OUT_DIR:-./example_outputs}
QUERY_SAMPLES=${QUERY_SAMPLES:-500}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[error] DATA_DIR does not exist: $DATA_DIR" >&2
  echo "Set DATA_DIR to a dataset root containing 0_real and 1_fake." >&2
  exit 1
fi

if [[ ! -f "$CNNSPOT_CKPT" ]]; then
  echo "[error] CNNSPOT_CKPT does not exist: $CNNSPOT_CKPT" >&2
  exit 1
fi

python "$PROJECT_ROOT/lid_detection_pipeline.py" \
  --stage all \
  --model_type resnet \
  --model_path "$CNNSPOT_CKPT" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR" \
  --exp_name cnnspot_dynamic_example \
  --query_samples "$QUERY_SAMPLES" \
  --query_samples 2500