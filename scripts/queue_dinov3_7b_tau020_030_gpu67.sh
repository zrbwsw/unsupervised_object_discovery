#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT=/data0/cuijiequan/zhangruobei/CondGen_DownStream/unsupervised_object_discovery
export TAUS=0.20,0.30
export GPU_IDS="${GPU_IDS:-6,7}"
export SESSION=ruobei_uod_dinov3_7b_tau020_030
export QUEUE_NAME=dinov3_7b_tau020_030_sweep
export QUEUE_CONFIG="$TASK_ROOT/configs/dinov3_7b_tau020_030_sweep.json"

exec bash "$TASK_ROOT/scripts/queue_dinov3_7b_tau.sh" "$@"
