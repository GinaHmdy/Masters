#!/bin/bash
# Build a self-contained bundle to continue the Gowalla experiments on another machine.
#
# What travels, and why:
#   code + docs        the experiment drivers and the four study artifacts
#   data/              the four datasets (76 MB)
#   results/           every completed metric, per-user dump and significance test (5 MB).
#                      This is the irreplaceable part -- it is the evidence, not the weights.
#   .cache/            precomputed popularity/density features; regenerable, but free to carry
#   2 checkpoints      ppd_v6-Gowalla-3-64-ablate-a_full-seed{2024,2025} (72 MB) so the two
#                      already-trained ablation runs are not repeated
#
# What does NOT travel: the other ~3 GB of checkpoints. Every completed result is already in
# results/*.json, and no remaining run reads those weights. Carry them only if you intend to
# re-evaluate a completed arm.
set -eu
OUT="${1:-$HOME/thesis_migration.tar.zst}"
cd "$(dirname "$(readlink -f "$0")")"

COMP="zstd -19 -T0"; command -v zstd >/dev/null || { COMP="gzip -9"; OUT="${OUT%.zst}.gz"; }

tar --use-compress-program="$COMP" -cf "$OUT" \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  LightGCN/code/*.py \
  LightGCN/code/*.sh \
  LightGCN/code/*.md \
  LightGCN/code/sources \
  LightGCN/code/results \
  LightGCN/code/.cache \
  LightGCN/code/checkpoints/ppd_v6-Gowalla-3-64-ablate-a_full-seed2024-best.pth.tar \
  LightGCN/code/checkpoints/ppd_v6-Gowalla-3-64-ablate-a_full-seed2025-best.pth.tar \
  LightGCN/code/checkpoints/ppd_v6-Gowalla-3-64-ablate-a_full-seed2024-gates.json \
  LightGCN/code/checkpoints/ppd_v6-Gowalla-3-64-ablate-a_full-seed2025-gates.json \
  data \
  *.md \
  docs

echo "bundle: $OUT"
ls -lh "$OUT" | awk '{print "size: "$5}'
echo
echo "checksum (verify after transfer):"
sha256sum "$OUT"
