#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

METHOD=dinov3
TAU="${1:-${TAU:-0.2}}"
if [[ $# -gt 1 ]]; then
    echo "usage: $0 [tau]" >&2
    exit 2
fi

TASK_ROOT=/data0/cuijiequan/zhangruobei/CondGen_DownStream/unsupervised_object_discovery
OBJECT_PYTHON="${OBJECT_PYTHON:-/data0/cuijiequan/env/miniconda3/envs/irepa/bin/python}"
DINOV3_WEIGHTS="${DINOV3_WEIGHTS:-/data1/pretrain_models}"
DINOV3_REVISION="${DINOV3_REVISION:-b80367753773648a6793235ab9c65cdbb029506f}"
DATASET="${DATASET:-VOC07}"
DATA_ARGS=()
case "$DATASET" in
    VOC07)
        DATA_PATH="${DATA_PATH:-/data0/datasets/dino_downstream/voc2007}"
        SPLIT=trainval
        DATA_ARGS=(--data_path="$DATA_PATH")
        ;;
    VOC12)
        DATA_PATH="${DATA_PATH:-/data0/datasets/dino_object_discovery/voc2012}"
        SPLIT=trainval
        DATA_ARGS=(--data_path="$DATA_PATH")
        ;;
    COCO20k)
        SPLIT=train
        ;;
    *)
        printf 'Unsupported DATASET=%s; choose VOC07, VOC12, or COCO20k.\n' "$DATASET" >&2
        exit 2
        ;;
esac

if [[ "$TAU" == *.* ]]; then
    TAU_FRACTION="${TAU#*.}"
    if [[ ${#TAU_FRACTION} -eq 1 ]]; then
        TAU_LABEL="${TAU}0"
    else
        TAU_LABEL="$TAU"
    fi
else
    TAU_LABEL="${TAU}.00"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$TASK_ROOT/outputs/$DATASET/$METHOD/tau_$TAU_LABEL}"
DEVICE="${DEVICE:-cuda}"
mkdir -p "$(dirname -- "$OUTPUT_DIR")"
if ! mkdir -- "$OUTPUT_DIR" 2>/dev/null; then
    if [[ -e "$OUTPUT_DIR" || -L "$OUTPUT_DIR" ]]; then
        echo "Refusing to overwrite existing output directory: $OUTPUT_DIR" >&2
    else
        echo "Could not create output directory: $OUTPUT_DIR" >&2
    fi
    exit 3
fi

cd "$TASK_ROOT/TokenCut"

"$OBJECT_PYTHON" -u main_tokencut.py \
    --encoder="$METHOD" \
    --arch=vit_7b \
    --patch_size=16 \
    --pretrained_weights="$DINOV3_WEIGHTS" \
    --checkpoint_revision="$DINOV3_REVISION" \
    "${DATA_ARGS[@]}" \
    --dataset="$DATASET" \
    --set="$SPLIT" \
    --which_features=output \
    --tau="$TAU" \
    --eps=1e-5 \
    --device="$DEVICE" \
    --output_dir="$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/discovery.log"
