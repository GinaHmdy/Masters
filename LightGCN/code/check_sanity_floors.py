"""
The sanity floors from the remediation plan, as an executable check.

Two reference rankers bound what any real result may look like:

  * **Random** -- every model must beat it on every accuracy metric, on every dataset. A model
    below random is a broken harness, not a weak model. This is the check that would have caught
    the KuaiRec candidate-mask bug on day one: LightGCN was sitting 19x *below* random, and a
    weak model drifts toward random rather than falling an order of magnitude beneath it.
  * **MostPopular** -- losing to a non-personalized popularity ranker is not automatically a bug,
    but it must be investigated before the number is reported.

Random also wins AvgPop@20 on every dataset, which is the standing argument for why AvgPop
cannot carry a fairness claim by itself.

Exit code is non-zero if any hard floor is breached, so this can gate a results regeneration.

Usage:
    python check_sanity_floors.py
"""
import json
import os
import sys

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DATASETS = ["Gowalla", "Yahoo", "KuaiRec", "Coat"]
ACCURACY = ["recall@20", "ndcg@20", "precision@20"]
ORDER = ["LightGCN", "TSP", "PPD v3", "PPD v4", "PPD v6"]


def load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def phase1(ds):
    data = {}
    for suffix in ("", "_v6"):
        d = load(f"phase1_{ds.lower()}{suffix}.json")
        if d:
            data.update(d)
    return data


def main():
    triv = load("trivial_baselines.json")
    if not triv:
        raise SystemExit("results/trivial_baselines.json missing - run trivial_baselines.py")
    triv = triv["results"]

    hard_failures, investigate, avgpop_notes = [], [], []

    for ds in DATASETS:
        data, ref = phase1(ds), triv.get(ds)
        if not data or not ref:
            continue
        rand = ref["Random"]["mean"]
        mostpop = ref["MostPopular"]
        print(f"\n=== {ds} ===")
        print(f"  {'method':<10}" + "".join(f"{m:>16}" for m in ACCURACY) + f"{'vs Random':>12}{'vs MostPop':>12}")
        for m in ORDER:
            if m not in data:
                continue
            mean = data[m]["mean"]
            beats_rand = all(mean.get(k, 0) > rand[k] for k in ACCURACY)
            beats_pop = all(mean.get(k, 0) > mostpop[k] for k in ACCURACY)
            if not beats_rand:
                losing = [k for k in ACCURACY if mean.get(k, 0) <= rand[k]]
                hard_failures.append((ds, m, losing))
            if not beats_pop:
                losing = [k for k in ACCURACY if mean.get(k, 0) <= mostpop[k]]
                investigate.append((ds, m, losing))
            print(f"  {m:<10}" + "".join(f"{mean.get(k, float('nan')):>16.6f}" for k in ACCURACY)
                  + f"{'PASS' if beats_rand else 'FAIL':>12}{'pass' if beats_pop else 'LOSES':>12}")
        print(f"  {'Random':<10}" + "".join(f"{rand[k]:>16.6f}" for k in ACCURACY))
        print(f"  {'MostPopular':<10}" + "".join(f"{mostpop[k]:>16.6f}" for k in ACCURACY))

        # AvgPop@20: is the trivial ranker still the "fairest"?
        model_pops = {m: data[m]["mean"]["avgpop@20"] for m in ORDER
                      if m in data and "avgpop@20" in data[m]["mean"]}
        if model_pops:
            best_model, best_pop = min(model_pops.items(), key=lambda kv: kv[1])
            rand_pop = rand["avgpop@20"]
            if rand_pop < best_pop:
                avgpop_notes.append((ds, rand_pop, best_model, best_pop))
            print(f"  AvgPop@20: Random {rand_pop:.2f} vs best model {best_model} {best_pop:.2f}"
                  f"  -> {'Random wins' if rand_pop < best_pop else 'model wins'}")

    print("\n" + "=" * 72)
    if avgpop_notes:
        print("AvgPop@20 is won by Random on: "
              + ", ".join(f"{ds} ({rp:.1f} vs {bp:.1f})" for ds, rp, _, bp in avgpop_notes))
        print("  -> AvgPop@20 alone cannot carry a fairness claim. Report Coverage@20 and APLT@20.")
    if investigate:
        print("\nLOSES TO MostPopular (investigate before reporting):")
        for ds, m, ks in investigate:
            print(f"  {ds:<9} {m:<10} on {', '.join(ks)}")
    if hard_failures:
        print("\n!!!! HARD FLOOR BREACHED - below Random on an accuracy metric !!!!")
        for ds, m, ks in hard_failures:
            print(f"  {ds:<9} {m:<10} on {', '.join(ks)}")
        sys.exit(1)
    print("\nAll models clear the Random floor on every accuracy metric.")


if __name__ == "__main__":
    main()
