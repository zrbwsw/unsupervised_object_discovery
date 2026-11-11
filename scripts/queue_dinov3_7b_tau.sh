#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT=/data0/cuijiequan/zhangruobei/CondGen_DownStream/unsupervised_object_discovery
QUEUE_SCRIPT="$TASK_ROOT/scripts/queue_dinov3_7b_tau.sh"
QUEUE_PYTHON="${OBJECT_PYTHON:-/data0/cuijiequan/env/miniconda3/envs/irepa/bin/python}"
SESSION="${SESSION:-ruobei_uod_dinov3_7b}"
QUEUE_NAME="${QUEUE_NAME:-dinov3_7b_tau_sweep}"
QUEUE_CONFIG="${QUEUE_CONFIG:-$TASK_ROOT/configs/dinov3_7b_tau_sweep.json}"
METHOD=dinov3
export TAUS="${TAUS:-0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40}"
IFS=',' read -r -a TAUS <<< "$TAUS"
DATASETS=(VOC07 VOC12 COCO20k)
export GPU_IDS="${GPU_IDS:-6,7}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
export DINOV3_WEIGHTS="${DINOV3_WEIGHTS:-/data1/pretrain_models}"
export DINOV3_REVISION="${DINOV3_REVISION:-b80367753773648a6793235ab9c65cdbb029506f}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

usage() {
    printf 'Usage: %s [--dry-run|--preflight|--foreground]\n' "$0" >&2
}

run_summary() {
    "$QUEUE_PYTHON" "$RUN_ROOT/launch_scripts/summarize_dinov3_7b_queue.py" "$RUN_ROOT" --taus "$TAUS" "$@"
}

if [[ "${1:-}" == "--worker" ]]; then
    GPU_ID="$2"
    RUN_ROOT="$3"
    STATUS_FILE="$RUN_ROOT/gpu${GPU_ID}_status.tsv"
    printf 'dataset\tmethod\ttau\tstarted_at\tfinished_at\texit_code\toutput_dir\n' > "$STATUS_FILE"
    FAILURES=0
    while IFS=$'\t' read -r ASSIGNED_GPU DATASET METHOD_NAME TAU_VALUE EXPERIMENT_DIR; do
        [[ "$ASSIGNED_GPU" == "$GPU_ID" ]] || continue
        STARTED_AT=$(date --iso-8601=seconds)
        printf '[%s] GPU %s: starting %s %s tau=%s\n' "$STARTED_AT" "$GPU_ID" "$DATASET" "$METHOD_NAME" "$TAU_VALUE"
        if [[ -e "$EXPERIMENT_DIR" || -L "$EXPERIMENT_DIR" ]]; then
            printf 'Refusing to overwrite existing experiment directory: %s\n' "$EXPERIMENT_DIR" >&2
            EXIT_CODE=3
            FAILURES=$((FAILURES + 1))
            FINISHED_AT=$(date --iso-8601=seconds)
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$DATASET" "$METHOD_NAME" "$TAU_VALUE" \
                "$STARTED_AT" "$FINISHED_AT" "$EXIT_CODE" "$EXPERIMENT_DIR" >> "$STATUS_FILE"
            continue
        fi
        if [[ "$DATASET" == VOC07 ]]; then
            DATA_PATH=/data0/datasets/dino_downstream/voc2007
            export DATA_PATH
        elif [[ "$DATASET" == VOC12 ]]; then
            DATA_PATH=/data0/datasets/dino_object_discovery/voc2012
            export DATA_PATH
        else
            unset DATA_PATH || true
        fi
        if CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" \
            OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            DATASET="$DATASET" TAU="$TAU_VALUE" OUTPUT_DIR="$EXPERIMENT_DIR" DEVICE=cuda \
            DINOV3_WEIGHTS="$DINOV3_WEIGHTS" DINOV3_REVISION="$DINOV3_REVISION" \
            bash "$RUN_ROOT/launch_scripts/eval_object_discovery_dinov3_7b.sh"; then
            EXIT_CODE=0
        else
            EXIT_CODE=$?
            FAILURES=$((FAILURES + 1))
        fi
        FINISHED_AT=$(date --iso-8601=seconds)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$DATASET" "$METHOD_NAME" "$TAU_VALUE" \
            "$STARTED_AT" "$FINISHED_AT" "$EXIT_CODE" "$EXPERIMENT_DIR" >> "$STATUS_FILE"
        printf '[%s] GPU %s: finished %s %s tau=%s, exit=%s\n' \
            "$FINISHED_AT" "$GPU_ID" "$DATASET" "$METHOD_NAME" "$TAU_VALUE" "$EXIT_CODE"
        run_summary --gpu "$GPU_ID" || printf 'Summary failed; status remains in %s\n' "$STATUS_FILE" >&2
    done < "$RUN_ROOT/manifest.tsv"
    printf '%s\n' "$FAILURES" > "$RUN_ROOT/gpu${GPU_ID}_failures.txt"
    printf 'GPU %s queue finished; failures=%s\n' "$GPU_ID" "$FAILURES"
    exit "$((FAILURES > 0))"
fi

if [[ "${1:-}" == "--run" ]]; then
    RUN_ROOT="$2"
    exec > >(tee -a "$RUN_ROOT/queue.log") 2>&1
    mkdir -p "$TASK_ROOT/outputs/.queue_locks"
    exec 9> "$TASK_ROOT/outputs/.queue_locks/dinov3_7b_tau.lock"
    if ! flock -n 9; then
        printf 'Another DINOv3 ViT-7B tau queue is already running.\n' | tee "$RUN_ROOT/queue_failed.txt" >&2
        exit 1
    fi
    printf '[%s] Starting DINOv3 workers; each GPU runs its assigned experiments serially.\n' "$(date --iso-8601=seconds)" \
        | tee "$RUN_ROOT/queue_started.txt"
    declare -a PIDS=()
    for GPU_ID in "${GPUS[@]}"; do
        bash "$RUN_ROOT/launch_scripts/queue_dinov3_7b_tau.sh" --worker "$GPU_ID" "$RUN_ROOT" \
            > "$RUN_ROOT/gpu${GPU_ID}.log" 2>&1 &
        PIDS+=("$!")
        printf 'GPU %s worker PID=%s\n' "$GPU_ID" "${PIDS[-1]}"
    done
    stop_descendants() {
        local PARENT_PID="$1" CHILD_PID
        for CHILD_PID in $(pgrep -P "$PARENT_PID" || true); do
            stop_descendants "$CHILD_PID"
        done
        kill -TERM "$PARENT_PID" 2>/dev/null || true
    }
    trap 'for PID in "${PIDS[@]}"; do stop_descendants "$PID"; done; wait || true; exit 130' INT TERM HUP
    declare -a CODES=()
    QUEUE_CODE=0
    for INDEX in "${!GPUS[@]}"; do
        CODES[$INDEX]=0
        wait "${PIDS[$INDEX]}" || CODES[$INDEX]=$?
        if [[ "${CODES[$INDEX]}" -ne 0 ]]; then
            QUEUE_CODE=1
        fi
        printf 'GPU %s worker exit=%s\n' "${GPUS[$INDEX]}" "${CODES[$INDEX]}"
    done
    SUMMARY_CODE=0
    run_summary --require-complete || SUMMARY_CODE=$?
    printf 'Supervisor worker_exit=%s summary=%s\n' "$QUEUE_CODE" "$SUMMARY_CODE" | tee "$RUN_ROOT/queue_finished.txt"
    exit "$((QUEUE_CODE != 0 || SUMMARY_CODE != 0))"
fi

MODE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|--preflight|--foreground)
            [[ -z "$MODE" ]] || { usage; exit 2; }
            MODE="$1"
            shift
            ;;
        *) usage; exit 2 ;;
    esac
done

if [[ ${#GPUS[@]} -eq 0 ]]; then
    printf 'GPU_IDS must contain at least one comma-separated GPU index.\n' >&2
    exit 2
fi
declare -A SEEN_GPUS=()
for GPU_ID in "${GPUS[@]}"; do
    if [[ ! "$GPU_ID" =~ ^[0-9]+$ || -n "${SEEN_GPUS[$GPU_ID]:-}" ]]; then
        printf 'Invalid or duplicate GPU_IDS entry: %s\n' "$GPU_ID" >&2
        exit 2
    fi
    SEEN_GPUS[$GPU_ID]=1
done

if [[ "$MODE" == "--dry-run" ]]; then
    INDEX=0
    for DATASET in "${DATASETS[@]}"; do
        for TAU_VALUE in "${TAUS[@]}"; do
            GPU_ID="${GPUS[$((INDEX % ${#GPUS[@]}))]}"
            EXPERIMENT_DIR="$TASK_ROOT/outputs/$DATASET/$METHOD/tau_${TAU_VALUE}"
            printf '%s\t%s\t%s\t%s\t%s\n' "$GPU_ID" "$DATASET" "$METHOD" "$TAU_VALUE" "$EXPERIMENT_DIR"
            INDEX=$((INDEX + 1))
        done
    done
    exit 0
fi

command -v flock >/dev/null
if [[ "$MODE" == "" ]]; then
    command -v tmux >/dev/null
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        printf 'Session already exists: %s. Attach with: tmux attach -t %s\n' "$SESSION" "$SESSION" >&2
        exit 1
    fi
fi
bash -n "$TASK_ROOT/scripts/eval_object_discovery_dinov3_7b.sh"
PYTHONDONTWRITEBYTECODE=1 "$QUEUE_PYTHON" - <<PY
from pathlib import Path
root = Path("$TASK_ROOT")
for name in ("TokenCut/main_tokencut.py", "TokenCut/networks.py", "TokenCut/dinov3_adapter.py",
             "scripts/summarize_dinov3_7b_queue.py"):
    compile((root / name).read_text(), str(root / name), "exec")
print("DINOv3 source syntax checks passed")
PY

cd "$TASK_ROOT/TokenCut"
PYTHONDONTWRITEBYTECODE=1 "$QUEUE_PYTHON" - <<'PY'
from pathlib import Path
import json
from datasets import Dataset

voc07 = Path('/data0/datasets/dino_downstream/voc2007/VOCdevkit/VOC2007')
voc12 = Path('/data0/datasets/dino_object_discovery/voc2012/VOCdevkit/VOC2012')
for root, expected in ((voc07, 5011), (voc12, 11540)):
    split = root / 'ImageSets/Main/trainval.txt'
    names = split.read_text().split()
    assert len(names) == len(set(names)) == expected, (split, len(names))
    for name in names[:3]:
        assert (root / 'JPEGImages' / f'{name}.jpg').is_file()
        assert (root / 'Annotations' / f'{name}.xml').is_file()
    dataset = 'VOC07' if expected == 5011 else 'VOC12'
    data = Dataset(dataset, 'trainval', False, data_path=str(root.parent.parent))
    assert len(data.dataloader) == expected
    image, targets = data.dataloader[0]
    assert image.ndim == 3 and len(data.extract_gt(targets, data.get_image_name(targets))[0]) > 0

annotation = Path('datasets/instances_train2014_sel20k.json')
selected = Path('datasets/coco_20k_filenames.txt').read_text().split()
assert annotation.is_file() and len(selected) == len(set(selected)) == 19817
payload = json.loads(annotation.read_text())
assert len(payload['images']) == 19817
assert {x['file_name'] for x in payload['images']} == {Path(x).name for x in selected}
for image in payload['images'][:3]:
    assert (Path('datasets/COCO/images/train2014') / image['file_name']).is_file()
data = Dataset('COCO20k', 'train', False)
assert len(data.dataloader) == 19817
image, targets = data.dataloader[0]
assert image.ndim == 3 and len(data.extract_gt(targets, data.get_image_name(targets))[0]) > 0
print('Dataset preflight passed: VOC07=5011 VOC12=11540 COCO20k=19817')
PY

PYTHONDONTWRITEBYTECODE=1 "$QUEUE_PYTHON" - <<'PY'
import json
import os
from dinov3_adapter import inspect_dinov3_checkpoint
report = inspect_dinov3_checkpoint(os.environ['DINOV3_WEIGHTS'])
print(json.dumps(report, indent=2))
print('DINOv3 checkpoint preflight passed')
PY

EXPECTED_GPUS="${#GPUS[@]}" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    PYTHONDONTWRITEBYTECODE=1 "$QUEUE_PYTHON" - <<'PY'
import os
import torch
expected = int(os.environ['EXPECTED_GPUS'])
assert torch.cuda.is_available() and torch.cuda.device_count() == expected, (
    f'Expected {expected} visible GPUs, found {torch.cuda.device_count()}'
)
for index in range(expected):
    properties = torch.cuda.get_device_properties(index)
    print(f'GPU {index}: {properties.name}, total_memory={properties.total_memory / 2**30:.2f} GiB')
PY

if [[ "$MODE" == "--preflight" ]]; then
    exit 0
fi

for DATASET in "${DATASETS[@]}"; do
    for TAU_VALUE in "${TAUS[@]}"; do
        EXPERIMENT_DIR="$TASK_ROOT/outputs/$DATASET/$METHOD/tau_${TAU_VALUE}"
        if [[ -e "$EXPERIMENT_DIR" || -L "$EXPERIMENT_DIR" ]]; then
            printf 'Refusing to overwrite existing experiment directory: %s\n' "$EXPERIMENT_DIR" >&2
            exit 1
        fi
    done
done

RUN_ID=$(date +%Y%m%d_%H%M%S)_$$
RUN_ROOT="$TASK_ROOT/outputs/_queues/$QUEUE_NAME/$RUN_ID"
mkdir -p "$RUN_ROOT/launch_scripts"
INDEX=0
for DATASET in "${DATASETS[@]}"; do
    for TAU_VALUE in "${TAUS[@]}"; do
        GPU_ID="${GPUS[$((INDEX % ${#GPUS[@]}))]}"
        printf '%s\t%s\t%s\t%s\t%s\n' "$GPU_ID" "$DATASET" "$METHOD" "$TAU_VALUE" \
            "$TASK_ROOT/outputs/$DATASET/$METHOD/tau_${TAU_VALUE}" >> "$RUN_ROOT/manifest.tsv"
        INDEX=$((INDEX + 1))
    done
done
printf '%s\n' "$SESSION" > "$RUN_ROOT/tmux_session.txt"
printf '%s\n' "$GPU_IDS" > "$RUN_ROOT/gpu_ids.txt"
printf '%s\n' "$DINOV3_WEIGHTS" > "$RUN_ROOT/checkpoint_path.txt"
printf '%s\n' "$DINOV3_REVISION" > "$RUN_ROOT/checkpoint_revision.txt"
printf '%s\n' "$RUN_ROOT" > "$TASK_ROOT/outputs/_queues/$QUEUE_NAME/latest_run.txt"
cp "$QUEUE_SCRIPT" "$TASK_ROOT/scripts/eval_object_discovery_dinov3_7b.sh" \
   "$TASK_ROOT/scripts/summarize_dinov3_7b_queue.py" "$TASK_ROOT/TokenCut/dinov3_adapter.py" \
   "$RUN_ROOT/launch_scripts/"
cp "$QUEUE_CONFIG" "$RUN_ROOT/protocol_config.json"

if [[ "$MODE" == "--foreground" ]]; then
    exec bash "$RUN_ROOT/launch_scripts/queue_dinov3_7b_tau.sh" --run "$RUN_ROOT"
fi
tmux new-session -d -s "$SESSION" -n queue -c "$TASK_ROOT" \
    "bash '$RUN_ROOT/launch_scripts/queue_dinov3_7b_tau.sh' --run '$RUN_ROOT'"
tmux set-option -t "$SESSION" remain-on-exit on
printf 'Started tmux session: %s\nRun directory: %s\n' "$SESSION" "$RUN_ROOT"
