#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT=/data0/cuijiequan/zhangruobei/CondGen_DownStream/unsupervised_object_discovery
QUEUE_SCRIPT="$TASK_ROOT/scripts/queue_voc07_tau_gpu67.sh"
QUEUE_PYTHON=/data0/cuijiequan/env/miniconda3/envs/irepa/bin/python
SESSION=ruobei_unsupervised_object_discovery
METHODS=(dino ibot mae condgen)
TAUS=(0.10 0.15 0.20 0.25 0.30)

# A worker executes only its ten rows, sequentially, and records failures.
if [[ "${1:-}" == "--worker" ]]; then
    GPU_ID="$2"
    RUN_ROOT="$3"
    STATUS_FILE="$RUN_ROOT/gpu${GPU_ID}_status.tsv"
    printf 'method\ttau\tstarted_at\tfinished_at\texit_code\toutput_dir\n' > "$STATUS_FILE"
    FAILURES=0
    while IFS=$'\t' read -r ASSIGNED_GPU METHOD TAU_VALUE EXPERIMENT_DIR; do
        [[ "$ASSIGNED_GPU" == "$GPU_ID" ]] || continue
        STARTED_AT=$(date --iso-8601=seconds)
        printf '[%s] GPU %s: starting %s tau=%s\n' "$STARTED_AT" "$GPU_ID" "$METHOD" "$TAU_VALUE"
        # Method scripts preserve these settings. Each Python process sees one GPU.
        if CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" \
            OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            TAU="$TAU_VALUE" OUTPUT_DIR="$EXPERIMENT_DIR" DEVICE=cuda \
            bash "$TASK_ROOT/scripts/eval_object_discovery_${METHOD}_b16.sh"; then
            EXIT_CODE=0
        else
            EXIT_CODE=$?
            FAILURES=$((FAILURES + 1))
        fi
        FINISHED_AT=$(date --iso-8601=seconds)
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$METHOD" "$TAU_VALUE" "$STARTED_AT" \
            "$FINISHED_AT" "$EXIT_CODE" "$EXPERIMENT_DIR" >> "$STATUS_FILE"
        printf '[%s] GPU %s: finished %s tau=%s, exit=%s\n' "$FINISHED_AT" "$GPU_ID" "$METHOD" "$TAU_VALUE" "$EXIT_CODE"
        "$QUEUE_PYTHON" "$TASK_ROOT/scripts/summarize_voc07_queue.py" "$RUN_ROOT" --gpu "$GPU_ID" \
            || printf 'Summary failed; the experiment status is preserved in %s\n' "$STATUS_FILE" >&2
    done < "$RUN_ROOT/manifest.tsv"
    printf '%s\n' "$FAILURES" > "$RUN_ROOT/gpu${GPU_ID}_failures.txt"
    printf 'GPU %s queue finished; failures=%s\n' "$GPU_ID" "$FAILURES"
    exit "$((FAILURES > 0))"
fi

# The tmux supervisor owns both workers and waits for the complete sweep.
if [[ "${1:-}" == "--run" ]]; then
    RUN_ROOT="$2"
    printf 'VOC07 trainval: 4 methods x 5 tau values, 10 jobs per GPU.\nRun directory: %s\n' "$RUN_ROOT"
    bash "$QUEUE_SCRIPT" --worker 6 "$RUN_ROOT" > "$RUN_ROOT/gpu6.log" 2>&1 &
    PID6=$!
    bash "$QUEUE_SCRIPT" --worker 7 "$RUN_ROOT" > "$RUN_ROOT/gpu7.log" 2>&1 &
    PID7=$!
    printf 'GPU 6 worker PID=%s; GPU 7 worker PID=%s\n' "$PID6" "$PID7"
    printf 'Follow logs: tail -f %q %q\n' "$RUN_ROOT/gpu6.log" "$RUN_ROOT/gpu7.log"
    CODE6=0
    CODE7=0
    wait "$PID6" || CODE6=$?
    wait "$PID7" || CODE7=$?
    "$QUEUE_PYTHON" "$TASK_ROOT/scripts/summarize_voc07_queue.py" "$RUN_ROOT"
    printf 'Supervisor worker exit codes: GPU6=%s GPU7=%s\n' "$CODE6" "$CODE7" | tee "$RUN_ROOT/queue_finished.txt"
    exit "$((CODE6 != 0 || CODE7 != 0))"
fi

if [[ "${1:-}" != "" && "${1:-}" != "--dry-run" ]]; then
    printf 'Usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi

# Interleave the 20 method/tau combinations; each GPU receives exactly ten.
if [[ "${1:-}" == "--dry-run" ]]; then
    INDEX=0
    for METHOD in "${METHODS[@]}"; do
        for TAU_VALUE in "${TAUS[@]}"; do
            printf 'GPU %s\t%s\ttau=%s\n' "$((6 + INDEX % 2))" "$METHOD" "$TAU_VALUE"
            INDEX=$((INDEX + 1))
        done
    done
    exit 0
fi

command -v tmux > /dev/null
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    printf 'Session already exists: %s. Attach with: tmux attach -t %s\n' "$SESSION" "$SESSION" >&2
    exit 1
fi
for METHOD in "${METHODS[@]}"; do
    test -r "$TASK_ROOT/../encoder/${METHOD}_vitb16.pth"
    bash -n "$TASK_ROOT/scripts/eval_object_discovery_${METHOD}_b16.sh"
done
test -r /data0/datasets/dino_downstream/voc2007/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt

RUN_ID=$(date +%Y%m%d_%H%M%S)_$$
RUN_ROOT="$TASK_ROOT/outputs/voc07_tau_sweep/$RUN_ID"
mkdir -p "$RUN_ROOT"
INDEX=0
for METHOD in "${METHODS[@]}"; do
    for TAU_VALUE in "${TAUS[@]}"; do
        GPU_ID=$((6 + INDEX % 2))
        printf '%s\t%s\t%s\t%s\n' "$GPU_ID" "$METHOD" "$TAU_VALUE" \
            "$RUN_ROOT/${METHOD}/tau_${TAU_VALUE}" >> "$RUN_ROOT/manifest.tsv"
        INDEX=$((INDEX + 1))
    done
done
printf '%s\n' "$SESSION" > "$RUN_ROOT/tmux_session.txt"
printf '%s\n' "$RUN_ROOT" > "$TASK_ROOT/outputs/voc07_tau_sweep/latest_run.txt"
# Fixed repository paths and generated timestamp contain no shell metacharacters.
tmux new-session -d -s "$SESSION" -n queue -c "$TASK_ROOT" \
    "bash '$QUEUE_SCRIPT' --run '$RUN_ROOT'"
tmux set-option -t "$SESSION" remain-on-exit on
printf 'Started tmux session: %s\nRun directory: %s\n' "$SESSION" "$RUN_ROOT"
