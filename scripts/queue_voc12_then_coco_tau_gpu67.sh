#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT=/data0/cuijiequan/zhangruobei/CondGen_DownStream/unsupervised_object_discovery
SEQUENCE_SCRIPT="$TASK_ROOT/scripts/queue_voc12_then_coco_tau_gpu67.sh"
SESSION=ruobei_uod_voc12_then_coco
VOC12_SCRIPT="$TASK_ROOT/scripts/queue_voc12_tau_gpu67.sh"
COCO_SCRIPT="$TASK_ROOT/scripts/queue_coco20k_tau_gpu67.sh"

# --foreground blocks for both dataset workers. Only a successful, complete
# VOC12 sweep allows COCO20k to start; a failed sweep can be inspected/re-run.
if [[ "${1:-}" == "--run" ]]; then
    SEQUENCE_ROOT="$2"
    printf 'Stage 1/2: VOC12 (20 jobs; GPU 6/7 each run 10).\n'
    VOC12_EXIT=0
    bash "$VOC12_SCRIPT" --foreground 2>&1 | tee "$SEQUENCE_ROOT/voc12_queue.log" || VOC12_EXIT=$?
    if [[ "$VOC12_EXIT" -ne 0 ]]; then
        printf 'VOC12 failed with exit=%s; COCO20k was not started.\n' "$VOC12_EXIT" | tee "$SEQUENCE_ROOT/sequence_failed.txt" >&2
        exit "$VOC12_EXIT"
    fi
    printf 'Stage 2/2: VOC12 completed on BOTH GPUs; starting COCO20k.\n'
    COCO_EXIT=0
    bash "$COCO_SCRIPT" --foreground 2>&1 | tee "$SEQUENCE_ROOT/coco20k_queue.log" || COCO_EXIT=$?
    if [[ "$COCO_EXIT" -ne 0 ]]; then
        printf 'COCO20k failed with exit=%s.\n' "$COCO_EXIT" | tee "$SEQUENCE_ROOT/sequence_failed.txt" >&2
        exit "$COCO_EXIT"
    fi
    printf 'VOC12 and COCO20k completed; 40 experiments total.\n' | tee "$SEQUENCE_ROOT/sequence_finished.txt"
    exit 0
fi

MODE="${1:-}"
if [[ $# -gt 1 || ( "$MODE" != "" && "$MODE" != "--dry-run" && "$MODE" != "--foreground" && "$MODE" != "--preflight" ) ]]; then
    printf 'Usage: %s [--dry-run|--foreground|--preflight]\n' "$0" >&2
    exit 2
fi
if [[ "$MODE" == "--dry-run" ]]; then
    printf 'Stage 1/2: VOC12\n'
    bash "$VOC12_SCRIPT" --dry-run
    printf 'Stage 2/2: COCO20k (starts after the entire VOC12 sweep succeeds)\n'
    bash "$COCO_SCRIPT" --dry-run
    exit 0
fi
if [[ "$MODE" == "" ]]; then
    command -v tmux > /dev/null
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        printf 'Session already exists: %s. Attach with: tmux attach -t %s\n' "$SESSION" "$SESSION" >&2
        exit 1
    fi
fi
# Check both datasets first, so missing COCO inputs cannot waste a VOC12 sweep.
bash "$VOC12_SCRIPT" --preflight
bash "$COCO_SCRIPT" --preflight
if [[ "$MODE" == "--preflight" ]]; then
    exit 0
fi
SEQUENCE_ID=$(date +%Y%m%d_%H%M%S)_$$
SEQUENCE_ROOT="$TASK_ROOT/outputs/_queues/voc12_then_coco_tau_sweep/$SEQUENCE_ID"
mkdir -p "$SEQUENCE_ROOT"
printf '%s\n' "$SEQUENCE_ROOT" > "$TASK_ROOT/outputs/_queues/voc12_then_coco_tau_sweep/latest_run.txt"
cp "$SEQUENCE_SCRIPT" "$SEQUENCE_ROOT/"
if [[ "$MODE" == "--foreground" ]]; then
    exec bash "$SEQUENCE_SCRIPT" --run "$SEQUENCE_ROOT"
fi
printf '%s\n' "$SESSION" > "$SEQUENCE_ROOT/tmux_session.txt"
tmux new-session -d -s "$SESSION" -n queue -c "$TASK_ROOT" \
    "bash '$SEQUENCE_SCRIPT' --run '$SEQUENCE_ROOT'"
tmux set-option -t "$SESSION" remain-on-exit on
printf 'Started tmux session: %s\nSequence logs: %s\n' "$SESSION" "$SEQUENCE_ROOT"
