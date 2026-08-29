#!/bin/bash
# Resume the thesis experiment queue after an interruption.
#
# Safe to run repeatedly: every stage skips work whose checkpoint already exists, and every
# driver merges into its results JSON rather than overwriting it. Nothing already completed
# is recomputed.
#
# Completed and preserved as of the 2026-08-27 GPU fault:
#   - Phase 1: all 4 datasets x 5 methods x 3 seeds (final)
#   - Phase 2 ablation: Coat (5 arms), Yahoo (5 arms)
#   - Phase 2 ablation KuaiRec: a_full x3, b_no_spectral x2 cached
#   - Phase 2 ablation Gowalla: a_full x2 cached
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY=/home/mohabmostafa/miniconda3/envs/gMasters/bin/python

if ! $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "ERROR: CUDA is not available. On WSL2 this usually needs 'wsl --shutdown' from Windows,"
  echo "       then reopening the terminal. Refusing to start - CPU runs would take weeks and"
  echo "       would not be comparable to the GPU results already collected."
  exit 1
fi
echo "CUDA OK: $($PY -c 'import torch;print(torch.cuda.get_device_name(0))')"

stage () { echo ""; echo "######## STAGE $1 :: $(date +%m-%d_%H:%M) ########"; }

stage "1/7 ablation KuaiRec (resumes from cache)"
$PY run_phase2_ablation.py KuaiRec

stage "2/7 gate transfer Coat+Yahoo"
$PY run_phase3_gate_transfer.py --skip-gowalla

stage "3/7 arm f (sampler isolation) on completed datasets"
for ds in Coat Yahoo KuaiRec; do $PY run_phase2_ablation.py $ds --arms f_no_sampler; done

stage "4/7 seed extension KuaiRec (2027, 2028)"
$PY run_phase1_baselines.py KuaiRec --seeds 2027,2028
$PY run_phase1_v6.py KuaiRec --seeds 2027,2028

stage "5/7 seed extension Gowalla (2027, 2028) - decides the headline recall significance"
$PY run_phase1_baselines.py Gowalla --seeds 2027,2028
$PY run_phase1_v6.py Gowalla --seeds 2027,2028

stage "6/7 ablation Gowalla (~3.5 h/run)"
$PY run_phase2_ablation.py Gowalla

stage "7/7 gate transfer Gowalla"
$PY run_phase3_gate_transfer.py Gowalla

echo ""; echo "######## QUEUE COMPLETE $(date +%m-%d_%H:%M) ########"
echo "Now regenerate the derived artifacts:"
echo "  $PY make_canonical.py && $PY make_readme_tables.py && $PY make_numbers_pack.py"
echo "  $PY check_sanity_floors.py"
