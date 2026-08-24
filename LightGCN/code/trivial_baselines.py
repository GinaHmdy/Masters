"""
trivial_baselines.py

MostPopular and Random reference rankers, on every dataset, with the same four
metrics the thesis reports.

Why this exists
---------------
A recommender that loses to "show everyone the most popular items" has not
demonstrated anything, and this check was missing from every report in the repo.
On KuaiRec it is decisive: MostPopular scores Recall@20 0.1233 against the best
model's 0.0139.

This script deliberately reads data/<ds>/{train,test}.txt directly rather than
going through world/register, so it is an independent check on the evaluation
path and needs neither a GPU nor a trained checkpoint.

Metric definitions mirror LightGCN/code/utils.py:222-266 exactly:
  recall@k    = mean over test users of  hits_u / |test_u|
  precision@k = mean over test users of  hits_u / k
  ndcg@k      = binary relevance, IDCG over min(k, |test_u|), discount 1/log2(i+2)
  avgpop@k    = mean training degree of the recommended items
Training positives are masked out of the recommendation list, as in
Procedure.Test.

Run:  python trivial_baselines.py [--datasets Gowalla Yahoo KuaiRec Coat] [--k 20]
Out:  results/trivial_baselines.json
"""
import argparse
import json
import os

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def read_lists(path):
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            out[int(parts[0])] = [int(x) for x in parts[1:]]
    return out


def evaluate(rank_fn, train, test, n_items, item_deg, k, rng=None):
    """rank_fn(user, blocked) -> list of k item ids, training items excluded."""
    recalls, precisions, ndcgs, pops = [], [], [], []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for u, pos in test.items():
        pos_set = set(pos)
        if not pos_set:
            continue
        recs = rank_fn(u, set(train.get(u, [])))
        gains = np.array([1.0 if i in pos_set else 0.0 for i in recs])
        hits = gains.sum()
        recalls.append(hits / len(pos_set))
        precisions.append(hits / k)
        dcg = float((gains * discounts).sum())
        ideal = float(discounts[:min(k, len(pos_set))].sum())
        ndcgs.append(dcg / ideal if ideal > 0 else 0.0)
        pops.extend(item_deg[recs])
    return {
        "recall@%d" % k: float(np.mean(recalls)),
        "ndcg@%d" % k: float(np.mean(ndcgs)),
        "precision@%d" % k: float(np.mean(precisions)),
        "avgpop@%d" % k: float(np.mean(pops)),
    }


def run_dataset(ds, k, seeds=(2024, 2025, 2026)):
    train = read_lists(os.path.join(DATA, ds, "train.txt"))
    test = read_lists(os.path.join(DATA, ds, "test.txt"))
    n_items = 1 + max(max(v) for v in list(train.values()) + list(test.values()))
    item_deg = np.zeros(n_items)
    for items in train.values():
        for i in items:
            item_deg[i] += 1

    popular_order = np.argsort(-item_deg, kind="stable")

    def most_popular(_u, blocked):
        recs = []
        for i in popular_order:
            if i not in blocked:
                recs.append(int(i))
                if len(recs) == k:
                    break
        return recs

    out = {"MostPopular": evaluate(most_popular, train, test, n_items, item_deg, k)}

    # Random is averaged over seeds; it is the floor every method must clear.
    per_seed = []
    for seed in seeds:
        rng = np.random.default_rng(seed)

        def random_rank(_u, blocked):
            recs = []
            while len(recs) < k:
                cand = rng.integers(0, n_items, size=k * 4)
                for i in cand:
                    i = int(i)
                    if i not in blocked and i not in recs:
                        recs.append(i)
                        if len(recs) == k:
                            break
            return recs

        per_seed.append(evaluate(random_rank, train, test, n_items, item_deg, k))
    keys = per_seed[0].keys()
    out["Random"] = {
        "mean": {m: float(np.mean([s[m] for s in per_seed])) for m in keys},
        "std": {m: float(np.std([s[m] for s in per_seed], ddof=1)) for m in keys},
        "seeds": list(seeds),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["Gowalla", "Yahoo", "KuaiRec", "Coat"])
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    results = {}
    hdr = f"{'dataset':10s} {'method':12s} {'Recall':>10s} {'NDCG':>10s} {'Prec':>10s} {'AvgPop':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for ds in args.datasets:
        results[ds] = run_dataset(ds, args.k)
        mp = results[ds]["MostPopular"]
        rnd = results[ds]["Random"]["mean"]
        for label, m in (("MostPopular", mp), ("Random", rnd)):
            print(f"{ds:10s} {label:12s} {m['recall@%d' % args.k]:10.6f} "
                  f"{m['ndcg@%d' % args.k]:10.6f} {m['precision@%d' % args.k]:10.6f} "
                  f"{m['avgpop@%d' % args.k]:10.2f}")

    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "trivial_baselines.json")
    with open(path, "w") as f:
        json.dump({"k": args.k, "results": results}, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
