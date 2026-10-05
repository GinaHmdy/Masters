#!/usr/bin/env bash
# Re-clone every vendored repo at its pinned commit and diff it against third_party/.
# Exits non-zero on any difference. Excluded paths come from third_party/SOURCES.txt.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
fail=0
while IFS='|' read -r dir url sha _date _lic excl; do
  dir=$(echo $dir); url=$(echo $url); sha=$(echo $sha)
  [[ -z "$dir" || "$dir" == \#* ]] && continue
  echo "== $dir @ $sha"
  git clone -q "$url" "$TMP/$dir"
  git -C "$TMP/$dir" checkout -q "$sha"
  [[ "$(git -C "$TMP/$dir" rev-parse HEAD)" == "$sha" ]] || { echo "SHA mismatch"; fail=1; }
  IFS=',' read -ra ex <<< "$excl"
  for e in "${ex[@]}"; do e=$(echo $e); e=${e%/}; rm -rf "${TMP:?}/$dir/$e"; done
  if diff -r -x __pycache__ "$TMP/$dir" "$TP/$dir"; then echo "   identical"; else echo "   DIFFERS"; fail=1; fi
done < "$TP/SOURCES.txt"
[[ $fail == 0 ]] && echo "third_party OK" || { echo "third_party VERIFY FAILED"; exit 1; }
