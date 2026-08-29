"""Merge per-worker result shards back into a single results JSON.

Parallel workers each write `<base>_shard<N>.json` so that no two processes read-modify-write
the same file. This merges them into `<base>.json`, preserving anything already there.

Two sharding schemes are in use and both must merge correctly:

  * by ARM (Phase 2): each shard holds a different top-level key, so the shards are disjoint.
  * by SEED (Phase 3): every shard holds the SAME keys (native / gowalla_init / yahoo_init),
    each carrying one seed. Taking one shard's value wholesale would silently drop the other
    seeds, so entries that share a key are merged at the `seeds` level and their mean/std are
    recomputed over the union.

Refuses to silently overwrite a differing value: if two shards disagree about the same arm AND
seed, that is a real problem (a checkpoint collision, a stale shard) and must be seen, not
averaged away.

Usage:  python merge_shards.py phase2_ablation_gowalla
"""
import glob
import json
import os
import sys

from run_phase1_baselines import summarize

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def merge_entry(old, new, key, shard, conflicts):
    """Combine two records for the same arm, seed-wise. Returns the merged record."""
    if not (isinstance(old, dict) and isinstance(new, dict)
            and "seeds" in old and "seeds" in new):
        # Not the {seeds, mean, std} shape -- fall back to last-writer-wins, loudly.
        if old != new:
            conflicts.append((shard, key, None, "record replaced wholesale"))
        return new

    seeds = dict(old["seeds"])
    for seed, metrics in new["seeds"].items():
        if seed in seeds and seeds[seed] != metrics:
            conflicts.append((shard, key, seed,
                              f"recall@20 {seeds[seed].get('recall@20')} -> "
                              f"{metrics.get('recall@20')}"))
        seeds[seed] = metrics

    merged = dict(old)
    merged.update({k: v for k, v in new.items() if k not in ("seeds", "mean", "std")})
    mean, std = summarize(seeds)
    merged["seeds"], merged["mean"], merged["std"] = seeds, mean, std
    return merged


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
        name = os.path.basename(path)
        for key, val in data.items():
            merged[key] = merge_entry(merged[key], val, key, name, conflicts) \
                if key in merged else val

    with open(target, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[merged] {len(shards)} shard(s) -> {target}")
    for key in sorted(merged):
        n = len(merged[key].get("seeds", {})) if isinstance(merged[key], dict) else 0
        print(f"         {key}: {n} seed(s)")

    if conflicts:
        print("\n!!!! CONFLICTING VALUES — the later shard won, but investigate before trusting !!!!")
        for shard, key, seed, detail in conflicts:
            where = f"{key} seed {seed}" if seed else key
            print(f"  {shard}: {where}: {detail}")
        sys.exit(1)

    for path in shards:
        os.remove(path)
    print(f"         removed {len(shards)} merged shard file(s)")


if __name__ == "__main__":
    main()
