#!/usr/bin/env bash
set -euo pipefail

# Inherit the caller's CUDA_VISIBLE_DEVICES; the queue assigns GPU 6 or 7.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

METHOD=ibot
TAU="${1:-${TAU:-0.2}}"
if [[ $# -gt 1 ]]; then
    echo "usage: $0 [tau]" >&2
    exit 2
fi

TASK_ROOT=/data0/cuijiequan/zhangruobei/CondGen_DownStream/unsupervised_object_discovery
OBJECT_PYTHON="${OBJECT_PYTHON:-/data0/cuijiequan/env/miniconda3/envs/irepa/bin/python}"
DATASET="${DATASET:-VOC07}"
DATA_ARGS=()
case "$DATASET" in
    VOC07)
        DATA_PATH="${DATA_PATH:-/data0/datasets/dino_downstream/voc2007}"
        SPLIT=trainval
        DATA_TAG=voc2007
        DATA_ARGS=(--data_path="$DATA_PATH")
        ;;
    VOC12)
        DATA_PATH="${DATA_PATH:-/data0/datasets/dino_object_discovery/voc2012}"
        SPLIT=trainval
        DATA_TAG=voc2012
        DATA_ARGS=(--data_path="$DATA_PATH")
        ;;
    COCO20k)
        # TokenCut resolves COCO paths through its datasets/COCO symlinks.
        SPLIT=train
        DATA_TAG=coco20k
        ;;
    *)
        printf 'Unsupported DATASET=%s; choose VOC07, VOC12, or COCO20k.\n' "$DATASET" >&2
        exit 2
        ;;
esac
CHECKPOINT=/data0/cuijiequan/zhangruobei/CondGen_DownStream/encoder/ibot_vitb16.pth
OUTPUT_DIR="${OUTPUT_DIR:-$TASK_ROOT/outputs/${DATA_TAG}_${SPLIT}_${METHOD}_vitb16_tau${TAU}/$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-auto}"
mkdir -p "$OUTPUT_DIR"

cd "$TASK_ROOT/TokenCut"

# Full-resolution images, final-layer output, binary graph, eps=1e-5.
# Do not pass --no_hard: retain difficult/truncated objects; max IoU >= 0.5.
"$OBJECT_PYTHON" -u main_tokencut.py \
    --encoder="$METHOD" \
    --arch=vit_base \
    --patch_size=16 \
    --checkpoint_key=model \
    --pretrained_weights="$CHECKPOINT" \
    "${DATA_ARGS[@]}" \
    --dataset="$DATASET" \
    --set="$SPLIT" \
    --which_features=output \
    --tau="$TAU" \
    --eps=1e-5 \
    --device="$DEVICE" \
    --output_dir="$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/discovery.log"
