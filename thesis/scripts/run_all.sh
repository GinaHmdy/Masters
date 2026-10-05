#!/usr/bin/env bash
# Reproduce the whole benchmark end to end on 2 GPUs + a CPU pool:
#   LightGCN backbones (GPU slots, round-robin) -> TSP/DAP tuning (GPU) + PPD tuning (CPU pool:
#   popularity scores, one process per beta, selection) -> ASTD-PD (GPU) -> report.
# Datasets run small first (Coat, Yahoo, KuaiRec, Gowalla). Idempotent: rerun to resume.
#   bash scripts/run_all.sh                         # everything
#   bash scripts/run_all.sh --datasets Coat         # one dataset (all 5 seeds)
# Extra args go to `python run.py all` (--seeds, --gpus, --gpu-slots, --cpu-jobs).
# Logs: logs/run_all.log (scheduler), logs/<dataset>_s<seed>_<job>.log, logs/jobs.jsonl (wall
# time per job), logs/status.txt (live progress).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
mkdir -p logs
echo "=== $(date '+%F %T') run_all.sh $*" >> logs/run_all.log
"$PY" run.py all "$@" >> logs/run_all.log 2>&1
