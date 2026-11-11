#!/usr/bin/env bash
set -euo pipefail

# Generate a DINOv3-style qualitative figure for one image.  This command
# re-runs TokenCut because the full evaluation snapshots store boxes, not the
# patch connected-component masks needed for an overlay.
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 IMAGE_PATH [TAU]" >&2
    exit 2
fi

IMAGE_PATH=$1
TAU=${2:-${TAU:-0.45}}
TASK_ROOT=/data0/cuijiequan/zhangruobei/CondGen_DownStream/unsupervised_object_discovery
OBJECT_PYTHON="${OBJECT_PYTHON:-/data0/cuijiequan/env/miniconda3/envs/irepa/bin/python}"
CHECKPOINT=/data0/cuijiequan/zhangruobei/CondGen_DownStream/encoder/condgen_vitb16.pth
OUTPUT_DIR="${OUTPUT_DIR:-$TASK_ROOT/outputs/visualizations/condgen_tau${TAU}/$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-auto}"

[[ -f "$IMAGE_PATH" ]] || { echo "image does not exist: $IMAGE_PATH" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
cd "$TASK_ROOT/TokenCut"

# --resize=1024 means the short side is 1024 pixels, preserving aspect ratio;
# TokenCut pads only the right/bottom edge to a multiple of patch size.
# No annotation, GT box, bilateral solver, or other post-processing is used.
exec "$OBJECT_PYTHON" -u main_tokencut.py \
    --image_path="$IMAGE_PATH" \
    --resize=1024 \
    --encoder=condgen \
    --arch=vit_base \
    --patch_size=16 \
    --checkpoint_key=model \
    --pretrained_weights="$CHECKPOINT" \
    --which_features=output \
    --tau="$TAU" \
    --eps=1e-5 \
    --device="$DEVICE" \
    --visualize=mask \
    --save_predictions=false \
    --output_dir="$OUTPUT_DIR"
