#!/usr/bin/env bash
set -euo pipefail

# Common implementation for the VOC12 and COCO20k entry scripts.
TASK_ROOT=/data0/cuijiequan/zhangruobei/CondGen_DownStream/unsupervised_object_discovery
QUEUE_SCRIPT="$TASK_ROOT/scripts/queue_object_discovery_tau_gpu67.sh"
QUEUE_PYTHON="${OBJECT_PYTHON:-/data0/cuijiequan/env/miniconda3/envs/irepa/bin/python}"
METHODS=(dino ibot mae condgen)
TAUS=(0.10 0.15 0.20 0.25 0.30)
DATASET="${1:-}"
case "$DATASET" in
    VOC12)
        DATA_TAG=voc12; SPLIT=trainval; SESSION=ruobei_uod_voc12
        export DATA_PATH=/data0/datasets/dino_object_discovery/voc2012
        ;;
    COCO20k)
        DATA_TAG=coco20k; SPLIT=train; SESSION=ruobei_uod_coco20k
        unset DATA_PATH
        ;;
    *) printf 'Usage: %s {VOC12|COCO20k} [--dry-run|--foreground|--preflight]\n' "$0" >&2; exit 2 ;;
esac
shift

summarize() {
    "$QUEUE_PYTHON" "$TASK_ROOT/scripts/summarize_object_discovery_queue.py" \
        "$RUN_ROOT" --dataset "$DATASET" "$@"
}

# A failed experiment is recorded and the worker continues to the next row.
if [[ "${1:-}" == "--worker" ]]; then
    GPU_ID="$2"
    RUN_ROOT="$3"
    STATUS_FILE="$RUN_ROOT/gpu${GPU_ID}_status.tsv"
    printf 'method\ttau\tstarted_at\tfinished_at\texit_code\toutput_dir\n' > "$STATUS_FILE"
    FAILURES=0
    while IFS=$'\t' read -r ASSIGNED_GPU METHOD TAU_VALUE EXPERIMENT_DIR; do
        [[ "$ASSIGNED_GPU" == "$GPU_ID" ]] || continue
        STARTED_AT=$(date --iso-8601=seconds)
        printf '[%s] GPU %s: starting %s %s tau=%s\n' "$STARTED_AT" "$GPU_ID" "$DATASET" "$METHOD" "$TAU_VALUE"
        if CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" \
            OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            DATASET="$DATASET" TAU="$TAU_VALUE" OUTPUT_DIR="$EXPERIMENT_DIR" DEVICE=cuda \
            bash "$TASK_ROOT/scripts/eval_object_discovery_${METHOD}_b16.sh"; then
            EXIT_CODE=0
        else
            EXIT_CODE=$?
            FAILURES=$((FAILURES + 1))
        fi
        FINISHED_AT=$(date --iso-8601=seconds)
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$METHOD" "$TAU_VALUE" "$STARTED_AT" \
            "$FINISHED_AT" "$EXIT_CODE" "$EXPERIMENT_DIR" >> "$STATUS_FILE"
        printf '[%s] GPU %s: finished %s %s tau=%s, exit=%s\n' \
            "$FINISHED_AT" "$GPU_ID" "$DATASET" "$METHOD" "$TAU_VALUE" "$EXIT_CODE"
        summarize --gpu "$GPU_ID" \
            || printf 'Summary failed; experiment status remains in %s\n' "$STATUS_FILE" >&2
    done < "$RUN_ROOT/manifest.tsv"
    printf '%s\n' "$FAILURES" > "$RUN_ROOT/gpu${GPU_ID}_failures.txt"
    printf 'GPU %s queue finished; failures=%s\n' "$GPU_ID" "$FAILURES"
    exit "$((FAILURES > 0))"
fi

# The supervisor reserves both GPUs for this project's new queues, starts the
# two serial workers, then waits for BOTH before returning to a sequence caller.
if [[ "${1:-}" == "--run" ]]; then
    RUN_ROOT="$2"
    mkdir -p "$TASK_ROOT/outputs/.queue_locks"
    exec 6> "$TASK_ROOT/outputs/.queue_locks/gpu6.lock"
    exec 7> "$TASK_ROOT/outputs/.queue_locks/gpu7.lock"
    if ! flock -n 6 || ! flock -n 7; then
        printf 'GPU 6/7 are reserved by another object-discovery queue.\n' | tee "$RUN_ROOT/queue_failed.txt" >&2
        exit 1
    fi
    printf '%s %s: 4 methods x 5 tau values, 10 jobs per GPU.\nRun directory: %s\n' "$DATASET" "$SPLIT" "$RUN_ROOT"
    bash "$QUEUE_SCRIPT" "$DATASET" --worker 6 "$RUN_ROOT" > "$RUN_ROOT/gpu6.log" 2>&1 &
    PID6=$!
    bash "$QUEUE_SCRIPT" "$DATASET" --worker 7 "$RUN_ROOT" > "$RUN_ROOT/gpu7.log" 2>&1 &
    PID7=$!
    # Stop both worker groups on cancellation; each worker's pipeline receives
    # the signal through the explicit descendant traversal below.
    stop_descendants() {
        local PARENT_PID="$1" CHILD_PID
        for CHILD_PID in $(pgrep -P "$PARENT_PID" || true); do
            stop_descendants "$CHILD_PID"
        done
        kill -TERM "$PARENT_PID" 2>/dev/null || true
    }
    trap 'stop_descendants "$PID6"; stop_descendants "$PID7"; wait || true; exit 130' INT TERM HUP
    printf 'GPU 6 worker PID=%s; GPU 7 worker PID=%s\n' "$PID6" "$PID7"
    printf 'Follow logs: tail -f %q %q\n' "$RUN_ROOT/gpu6.log" "$RUN_ROOT/gpu7.log"
    CODE6=0
    CODE7=0
    SUMMARY_CODE=0
    wait "$PID6" || CODE6=$?
    wait "$PID7" || CODE7=$?
    summarize --require-complete || SUMMARY_CODE=$?
    printf 'Supervisor exit codes: GPU6=%s GPU7=%s summary=%s\n' "$CODE6" "$CODE7" "$SUMMARY_CODE" | tee "$RUN_ROOT/queue_finished.txt"
    exit "$((CODE6 != 0 || CODE7 != 0 || SUMMARY_CODE != 0))"
fi

MODE="${1:-}"
if [[ $# -gt 1 || ( "$MODE" != "" && "$MODE" != "--dry-run" && "$MODE" != "--foreground" && "$MODE" != "--preflight" ) ]]; then
    printf 'Usage: %s %s [--dry-run|--foreground|--preflight]\n' "$0" "$DATASET" >&2
    exit 2
fi

# Dry run only prints the plan: no tmux, output directory or GPU process.
if [[ "$MODE" == "--dry-run" ]]; then
    INDEX=0
    for METHOD in "${METHODS[@]}"; do
        for TAU_VALUE in "${TAUS[@]}"; do
            printf 'GPU %s\t%s\t%s\ttau=%s\n' "$((6 + INDEX % 2))" "$DATASET" "$METHOD" "$TAU_VALUE"
            INDEX=$((INDEX + 1))
        done
    done
    exit 0
fi

command -v flock > /dev/null
if [[ "$MODE" == "" ]]; then
    command -v tmux > /dev/null
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        printf 'Session already exists: %s. Attach with: tmux attach -t %s\n' "$SESSION" "$SESSION" >&2
        exit 1
    fi
fi
for METHOD in "${METHODS[@]}"; do
    test -r "$TASK_ROOT/../encoder/${METHOD}_vitb16.pth"
    bash -n "$TASK_ROOT/scripts/eval_object_discovery_${METHOD}_b16.sh"
done

# Validate the real loader and all referenced image/annotation paths before
# launching either worker. COCO annotations must be prepared once in advance.
cd "$TASK_ROOT/TokenCut"
"$QUEUE_PYTHON" - "$DATASET" <<'PY'
import json
import sys
from pathlib import Path

import scipy.linalg
import torch
import torchvision
from datasets import Dataset

dataset = sys.argv[1]
if dataset == 'VOC12':
    root = Path('/data0/datasets/dino_object_discovery/voc2012')
    voc = root / 'VOCdevkit/VOC2012'
    names = (voc / 'ImageSets/Main/trainval.txt').read_text().split()
    assert len(names) == len(set(names)) == 11540, 'Unexpected VOC12 trainval size'
    for name in names:
        assert (voc / 'JPEGImages' / (name + '.jpg')).is_file(), name
        assert (voc / 'Annotations' / (name + '.xml')).is_file(), name
    data = Dataset('VOC12', 'trainval', False, data_path=str(root))
    expected = 11540
else:
    import pycocotools.coco

    annotation_path = Path('datasets/instances_train2014_sel20k.json')
    assert annotation_path.is_file(), 'Prepare COCO20k selected annotations before queue launch'
    assert Path('datasets/COCO/annotations/instances_train2014.json').is_file()
    annotations = json.loads(annotation_path.read_text())
    selected = Path('datasets/coco_20k_filenames.txt').read_text().split()
    images = annotations['images']
    assert len(images) == len(selected) == len(set(selected)) == 19817, 'Unexpected COCO20k size'
    assert {image['file_name'] for image in images} == {Path(name).name for name in selected}
    for image in images:
        assert (Path('datasets/COCO/images/train2014') / image['file_name']).is_file(), image['file_name']
    data = Dataset('COCO20k', 'train', False)
    expected = 19817
assert len(data.dataloader) == expected
image, targets = data.dataloader[0]
name = data.get_image_name(targets)
boxes, _ = data.extract_gt(targets, name)
assert image.ndim == 3 and len(boxes) > 0, 'First sample failed to load'
print(f'{dataset} preflight passed: {expected} images; first={name}; shape={tuple(image.shape)}')
PY

if [[ "$MODE" == "--preflight" ]]; then
    exit 0
fi
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 "$QUEUE_PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 2, 'GPU 6 and 7 are required'
print('Queue GPU devices:', [torch.cuda.get_device_name(i) for i in range(2)])
PY

RUN_ID=$(date +%Y%m%d_%H%M%S)_$$
RUN_ROOT="$TASK_ROOT/outputs/${DATA_TAG}_tau_sweep/$RUN_ID"
mkdir -p "$RUN_ROOT/launch_scripts"
INDEX=0
for METHOD in "${METHODS[@]}"; do
    for TAU_VALUE in "${TAUS[@]}"; do
        printf '%s\t%s\t%s\t%s\n' "$((6 + INDEX % 2))" "$METHOD" "$TAU_VALUE" \
            "$RUN_ROOT/${METHOD}/tau_${TAU_VALUE}" >> "$RUN_ROOT/manifest.tsv"
        INDEX=$((INDEX + 1))
    done
done
printf '%s\n' "$DATASET" > "$RUN_ROOT/dataset.txt"
cp "$QUEUE_SCRIPT" "$TASK_ROOT/scripts/queue_${DATA_TAG}_tau_gpu67.sh" \
    "$TASK_ROOT/scripts/summarize_object_discovery_queue.py" \
    "$TASK_ROOT"/scripts/eval_object_discovery_*_b16.sh "$RUN_ROOT/launch_scripts/"
printf '%s\n' "$RUN_ROOT" > "$TASK_ROOT/outputs/${DATA_TAG}_tau_sweep/latest_run.txt"

if [[ "$MODE" == "--foreground" ]]; then
    exec bash "$QUEUE_SCRIPT" "$DATASET" --run "$RUN_ROOT"
fi
printf '%s\n' "$SESSION" > "$RUN_ROOT/tmux_session.txt"
# Paths are fixed; timestamp and PID are generated locally.
tmux new-session -d -s "$SESSION" -n queue -c "$TASK_ROOT" \
    "bash '$QUEUE_SCRIPT' '$DATASET' --run '$RUN_ROOT'"
tmux set-option -t "$SESSION" remain-on-exit on
printf 'Started tmux session: %s\nRun directory: %s\n' "$SESSION" "$RUN_ROOT"
