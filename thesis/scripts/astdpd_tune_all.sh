#!/usr/bin/env bash
# Validation-only re-tuning of ASTD-PD on Yahoo, KuaiRec, Gowalla (src/astdpd/tune.py), run
# alongside `run.py all`:
#   setsid nohup scripts/astdpd_tune_all.sh > logs/astdpd_tune_all.log 2>&1 &
# Searches run one dataset at a time (6 workers = 3 per GPU), each after its baselines exist.
# Each dataset's final step runs in the background: it waits for `run.py all`'s shipped-config
# ASTD-PD runs of that dataset (kept as astdpd_shipped.*), installs the retuned winner as
# astdpd.* (copying runs already made during the search) and regenerates the report.
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-/home/mohab/conda/envs/gMasters/bin/python}
W=(--workers 6 --gpu 0 1 --threads 2)

final() {
  ds=$1
  (
    $PY -m src.astdpd.tune --dataset "$ds" --final-only --wait --workers 2 --gpu 0 1 \
      >> "logs/astdpd_tune_$ds.log" 2>&1 \
    && flock logs/.astdpd_report.lock $PY run.py report --gpu -1 --threads 2 >> "logs/astdpd_tune_$ds.log" 2>&1
    echo "$(date '+%F %T') $ds final step exit $?"
  ) &
}

search() {
  ds=$1; shift
  echo "$(date '+%F %T') $ds search start ($*)"
  if $PY -m src.astdpd.tune --dataset "$ds" --wait --no-final "${W[@]}" "$@" >> "logs/astdpd_tune_$ds.log" 2>&1; then
    echo "$(date '+%F %T') $ds search done"
    final "$ds"
  else
    echo "$(date '+%F %T') $ds search FAILED (see logs/astdpd_tune_$ds.log)"
  fi
}

search Yahoo --n-random 60 --top-k 6
search KuaiRec --n-random 18 --top-k 4
search Gowalla --n-random 7 --top-k 3 --mu-grid 0 1 2 4 8
wait
echo "$(date '+%F %T') all done"
