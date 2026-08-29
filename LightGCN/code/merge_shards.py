"""Merge per-worker result shards back into a single results JSON.

Parallel workers each write `<base>_shard<N>.json` so that no two processes read-modify-write
the same file. This merges them into `<base>.json`, preserving anything already there.

Refuses to silently overwrite a differing value for the same key: if two shards disagree about
an arm, that is a real problem (a checkpoint collision, a stale shard) and must be seen, not
averaged away.

Usage:  python merge_shards.py phase2_ablation_gowalla
"""
import glob
import json
import os
import sys

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    base = sys.argv[1]
    target = os.path.join(RESULTS, f"{base}.json")

    merged = {}
    if os.path.exists(target):
        with open(target) as f:
            merged = json.load(f)

    shards = sorted(glob.glob(os.path.join(RESULTS, f"{base}_shard*.json")))
    if not shards:
        print(f"no shards matching {base}_shard*.json — nothing to merge")
        return

    conflicts = []
    for path in shards:
        with open(path) as f:
            data = json.load(f)
        for key, val in data.items():
            if key in merged and merged[key] != val:
                old = merged[key].get("mean", {}).get("recall@20")
                new = val.get("mean", {}).get("recall@20")
                conflicts.append((os.path.basename(path), key, old, new))
            merged[key] = val

    with open(target, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[merged] {len(shards)} shard(s) -> {target}")
    print(f"         keys: {', '.join(sorted(merged))}")

    if conflicts:
        print("\n!!!! CONFLICTING VALUES — the later shard won, but investigate before trusting !!!!")
        for shard, key, old, new in conflicts:
            print(f"  {shard}: {key} recall@20 {old} -> {new}")
        sys.exit(1)

    for path in shards:
        os.remove(path)
    print(f"         removed {len(shards)} merged shard file(s)")


if __name__ == "__main__":
    main()
