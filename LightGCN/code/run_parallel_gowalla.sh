#!/bin/bash
# Finish the remaining Gowalla work across multiple GPUs.
#
# The remaining units are independent (one per arm x seed), so they parallelise cleanly. Two
# things make that safe, and both are load-bearing:
#
#   1. THESIS_CKPT_TAG gives each worker a private default checkpoint. Without it every
#      concurrent run writes the same ppd_v6-Gowalla-3-64.pth.tar and the drivers copy whichever
#      weights happened to land there last -- silent cross-arm contamination that the mtime
#      guard cannot catch, because the clobbered file is newer rather than older.
#   2. Each worker writes its own results shard (--out-suffix) and the shards are merged at the
#      end, so no two processes read-modify-write the same JSON.
#
# Usage:
#   ./run_parallel_gowalla.sh              # one worker per GPU
#   WORKERS_PER_GPU=2 ./run_parallel_gowalla.sh
#   GPUS=0,1 ./run_parallel_gowalla.sh
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY="${THESIS_PYTHON:-python}"
GPUS="${GPUS:-$($PY -c 'import torch;print(",".join(str(i) for i in range(torch.cuda.device_count())))')}"
WPG="${WORKERS_PER_GPU:-1}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
NWORK=$((NGPU * WPG))
echo "GPUs: $GPUS   workers/GPU: $WPG   total workers: $NWORK"

ARMS=(a_full b_no_spectral c_no_simplex d_no_tail e_recipe_only f_no_sampler)

# --- Stage A: seed extension (sequential; it trains ONE shared LightGCN per seed) -------------
echo "### stage A: Gowalla seed extension (2027, 2028)"
THESIS_CKPT_TAG=ext CUDA_VISIBLE_DEVICES=${GPU_ARR[0]} $PY run_phase1_baselines.py Gowalla --seeds 2027,2028
THESIS_CKPT_TAG=ext CUDA_VISIBLE_DEVICES=${GPU_ARR[0]} $PY run_phase1_v6.py Gowalla --seeds 2027,2028

# --- Stage B: ablation, one arm per worker ----------------------------------------------------
echo "### stage B: Gowalla ablation across $NWORK workers"
pids=()
for i in "${!ARMS[@]}"; do
  arm="${ARMS[$i]}"
  gpu="${GPU_ARR[$(( i % NGPU ))]}"
  (
    THESIS_CKPT_TAG="ab$i" CUDA_VISIBLE_DEVICES="$gpu" \
      $PY run_phase2_ablation.py Gowalla --arms "$arm" --out-suffix "_shard$i" \
      > "logs_gowalla_ablate_${arm}.log" 2>&1
    echo "  [done] $arm (gpu $gpu)"
  ) &
  pids+=($!)
  while [ "$(jobs -rp | wc -l)" -ge "$NWORK" ]; do sleep 10; done
done
wait "${pids[@]}" 2>/dev/null
$PY merge_shards.py phase2_ablation_gowalla

# --- Stage C: gate transfer (only yahoo_init needs training; the rest alias/reuse) ------------
echo "### stage C: Gowalla gate transfer"
THESIS_CKPT_TAG=tr CUDA_VISIBLE_DEVICES=${GPU_ARR[0]} $PY run_phase3_gate_transfer.py Gowalla

echo "### regenerating derived artifacts"
$PY make_canonical.py && $PY make_readme_tables.py && $PY make_numbers_pack.py && $PY check_sanity_floors.py
echo "### ALL DONE $(date +%H:%M)"
