#!/bin/bash
# Resume the thesis experiment queue after an interruption.
#
# Safe to run repeatedly: every stage skips work whose checkpoint already exists, and every
# driver merges into its results JSON rather than overwriting it. Nothing already completed
# is recomputed.
#
# STATUS 2026-09-05: this queue is COMPLETE. Every stage below has run on all four datasets
# and every row of the THESIS_AUDIT.md remediation ledger is closed. The script is kept because
# it is idempotent -- re-running it recomputes nothing, it only verifies that each stage's
# checkpoints and results are still present. Phases 4 and 5, added after this file was written,
# are not in it: see run_phase4_inference_penalty.py and run_phase5_hpo.py / run_phase5_final.py.
set -u
cd "$(dirname "$(readlink -f "$0")")"
PY="${THESIS_PYTHON:-$(command -v python)}"   # override with THESIS_PYTHON when the env moves

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
