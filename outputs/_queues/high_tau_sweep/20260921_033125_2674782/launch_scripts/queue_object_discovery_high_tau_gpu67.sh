#!/usr/bin/env bash
set -euo pipefail

# Run the two high-threshold experiments (tau=0.35 and 0.40) for all four
# encoders on VOC07, VOC12 and COCO20K.  The 24 rows are interleaved between
# GPUs 6 and 7, giving each worker twelve serial jobs.
TASK_ROOT=/data0/cuijiequan/zhangruobei/CondGen_DownStream/unsupervised_object_discovery
QUEUE_SCRIPT="$TASK_ROOT/scripts/queue_object_discovery_high_tau_gpu67.sh"
QUEUE_PYTHON="${OBJECT_PYTHON:-/data0/cuijiequan/env/miniconda3/envs/irepa/bin/python}"
SESSION=ruobei_uod_high_tau
METHODS=(dino ibot mae condgen)
TAUS=(0.35 0.40)
DATASETS=(VOC07 VOC12 COCO20k)

usage() {
    printf 'Usage: %s [--dry-run|--preflight|--foreground] [--after-sequence DIRECTORY]\n' "$0" >&2
}

run_summary() {
    "$QUEUE_PYTHON" "$TASK_ROOT/scripts/summarize_high_tau_queue.py" "$RUN_ROOT" "$@"
}

# Each worker consumes only its assigned rows and continues after an individual
# experiment failure.  DATASET/DATA_PATH are set per row because the same worker
# handles all three datasets.
if [[ "${1:-}" == "--worker" ]]; then
    GPU_ID="$2"
    RUN_ROOT="$3"
    STATUS_FILE="$RUN_ROOT/gpu${GPU_ID}_status.tsv"
    printf 'dataset\tmethod\ttau\tstarted_at\tfinished_at\texit_code\toutput_dir\n' > "$STATUS_FILE"
    FAILURES=0
    while IFS=$'\t' read -r ASSIGNED_GPU DATASET METHOD TAU_VALUE EXPERIMENT_DIR; do
        [[ "$ASSIGNED_GPU" == "$GPU_ID" ]] || continue
        STARTED_AT=$(date --iso-8601=seconds)
        printf '[%s] GPU %s: starting %s %s tau=%s\n' "$STARTED_AT" "$GPU_ID" "$DATASET" "$METHOD" "$TAU_VALUE"
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
            bash "$TASK_ROOT/scripts/eval_object_discovery_${METHOD}_b16.sh"; then
            EXIT_CODE=0
        else
            EXIT_CODE=$?
            FAILURES=$((FAILURES + 1))
        fi
        FINISHED_AT=$(date --iso-8601=seconds)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$DATASET" "$METHOD" "$TAU_VALUE" \
            "$STARTED_AT" "$FINISHED_AT" "$EXIT_CODE" "$EXPERIMENT_DIR" >> "$STATUS_FILE"
        printf '[%s] GPU %s: finished %s %s tau=%s, exit=%s\n' \
            "$FINISHED_AT" "$GPU_ID" "$DATASET" "$METHOD" "$TAU_VALUE" "$EXIT_CODE"
        run_summary --gpu "$GPU_ID" || printf 'Summary failed; status remains in %s\n' "$STATUS_FILE" >&2
    done < "$RUN_ROOT/manifest.tsv"
    printf '%s\n' "$FAILURES" > "$RUN_ROOT/gpu${GPU_ID}_failures.txt"
    printf 'GPU %s queue finished; failures=%s\n' "$GPU_ID" "$FAILURES"
    exit "$((FAILURES > 0))"
fi

# The supervisor takes the same project GPU locks as the existing VOC12/COCO
# queues, so a high-tau sweep cannot overlap another object-discovery sweep.
if [[ "${1:-}" == "--run" ]]; then
    RUN_ROOT="$2"
    exec > >(tee -a "$RUN_ROOT/queue.log") 2>&1
    # Wait for the complete VOC12->COCO sequence before taking either GPU lock.
    # Locking during this wait would block that sequence's COCO stage.
    if [[ -s "$RUN_ROOT/after_sequence.txt" ]]; then
        AFTER_SEQUENCE=$(cat "$RUN_ROOT/after_sequence.txt")
        printf '[%s] Waiting for previous sequence: %s\n' "$(date --iso-8601=seconds)" "$AFTER_SEQUENCE" \
            | tee "$RUN_ROOT/waiting_for_sequence.txt"
        trap 'printf "Queue cancelled while waiting for prior sequence.\n" > "$RUN_ROOT/queue_failed.txt"; exit 130' INT TERM HUP
        while true; do
            if [[ -e "$AFTER_SEQUENCE/sequence_failed.txt" ]]; then
                printf 'Previous sequence failed; high-tau queue was not started: %s\n' "$AFTER_SEQUENCE" \
                    | tee "$RUN_ROOT/queue_failed.txt" >&2
                cat "$AFTER_SEQUENCE/sequence_failed.txt" >&2
                exit 1
            fi
            if [[ -s "$AFTER_SEQUENCE/sequence_finished.txt" ]]; then
                printf '[%s] Previous sequence completed successfully.\n' "$(date --iso-8601=seconds)"
                mv "$RUN_ROOT/waiting_for_sequence.txt" "$RUN_ROOT/wait_completed.txt"
                break
            fi
            sleep 30
        done
    fi
    mkdir -p "$TASK_ROOT/outputs/.queue_locks"
    exec 6> "$TASK_ROOT/outputs/.queue_locks/gpu6.lock"
    exec 7> "$TASK_ROOT/outputs/.queue_locks/gpu7.lock"
    if ! flock -n 6 || ! flock -n 7; then
        printf 'GPU 6/7 are reserved by another object-discovery queue.\n' | tee "$RUN_ROOT/queue_failed.txt" >&2
        exit 1
    fi
    printf '[%s] GPU 6/7 reserved; starting high-tau workers.\n' "$(date --iso-8601=seconds)" \
        | tee "$RUN_ROOT/queue_started.txt"
    printf 'High-tau sweep: 3 datasets x 4 methods x 2 tau values, 12 jobs per GPU.\nRun directory: %s\n' "$RUN_ROOT"
    bash "$QUEUE_SCRIPT" --worker 6 "$RUN_ROOT" > "$RUN_ROOT/gpu6.log" 2>&1 &
    PID6=$!
    bash "$QUEUE_SCRIPT" --worker 7 "$RUN_ROOT" > "$RUN_ROOT/gpu7.log" 2>&1 &
    PID7=$!
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
    CODE6=0; CODE7=0; SUMMARY_CODE=0
    wait "$PID6" || CODE6=$?
    wait "$PID7" || CODE7=$?
    run_summary --require-complete || SUMMARY_CODE=$?
    printf 'Supervisor exit codes: GPU6=%s GPU7=%s summary=%s\n' "$CODE6" "$CODE7" "$SUMMARY_CODE" | tee "$RUN_ROOT/queue_finished.txt"
    exit "$((CODE6 != 0 || CODE7 != 0 || SUMMARY_CODE != 0))"
fi

MODE=""
AFTER_SEQUENCE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|--preflight|--foreground)
            [[ -z "$MODE" ]] || { usage; exit 2; }
            MODE="$1"
            shift
            ;;
        --after-sequence)
            [[ $# -ge 2 && -z "$AFTER_SEQUENCE" ]] || { usage; exit 2; }
            [[ -d "$2" ]] || { printf 'Sequence directory does not exist: %s\n' "$2" >&2; exit 2; }
            AFTER_SEQUENCE=$(readlink -f -- "$2")
            shift 2
            ;;
        *) usage; exit 2 ;;
    esac
done

# Dry-run is side-effect free and is useful for checking the exact 12/12 split.
if [[ "$MODE" == "--dry-run" ]]; then
    if [[ -n "$AFTER_SEQUENCE" ]]; then
        printf 'After successful sequence: %s\n' "$AFTER_SEQUENCE" >&2
    fi
    INDEX=0
    for DATASET in "${DATASETS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            for TAU_VALUE in "${TAUS[@]}"; do
                printf 'GPU %s\t%s\t%s\ttau=%s\n' "$((6 + INDEX % 2))" "$DATASET" "$METHOD" "$TAU_VALUE"
                INDEX=$((INDEX + 1))
            done
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
for METHOD in "${METHODS[@]}"; do
    test -r "$TASK_ROOT/../encoder/${METHOD}_vitb16.pth"
    bash -n "$TASK_ROOT/scripts/eval_object_discovery_${METHOD}_b16.sh"
done

# Validate all three loaders and the fixed COCO20K subset before any GPU process.
cd "$TASK_ROOT/TokenCut"
"$QUEUE_PYTHON" - <<'PY'
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
print('High-tau preflight passed: VOC07=5011 VOC12=11540 COCO20k=19817')
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
RUN_ROOT="$TASK_ROOT/outputs/high_tau_sweep/$RUN_ID"
mkdir -p "$RUN_ROOT/launch_scripts"
if [[ -n "$AFTER_SEQUENCE" ]]; then
    printf '%s\n' "$AFTER_SEQUENCE" > "$RUN_ROOT/after_sequence.txt"
fi
INDEX=0
for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        for TAU_VALUE in "${TAUS[@]}"; do
            printf '%s\t%s\t%s\t%s\t%s\n' "$((6 + INDEX % 2))" "$DATASET" "$METHOD" "$TAU_VALUE" \
                "$RUN_ROOT/$DATASET/$METHOD/tau_${TAU_VALUE}" >> "$RUN_ROOT/manifest.tsv"
            INDEX=$((INDEX + 1))
        done
    done
done
printf '%s\n' "$SESSION" > "$RUN_ROOT/tmux_session.txt"
printf '%s\n' "$RUN_ROOT" > "$TASK_ROOT/outputs/high_tau_sweep/latest_run.txt"
cp "$QUEUE_SCRIPT" "$TASK_ROOT/scripts/summarize_high_tau_queue.py" \
   "$TASK_ROOT"/scripts/eval_object_discovery_*_b16.sh "$RUN_ROOT/launch_scripts/"

if [[ "$MODE" == "--foreground" ]]; then
    exec bash "$QUEUE_SCRIPT" --run "$RUN_ROOT"
fi
tmux new-session -d -s "$SESSION" -n queue -c "$TASK_ROOT" \
    "bash '$QUEUE_SCRIPT' --run '$RUN_ROOT'"
tmux set-option -t "$SESSION" remain-on-exit on
printf 'Started tmux session: %s\nRun directory: %s\n' "$SESSION" "$RUN_ROOT"
