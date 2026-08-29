"""
Build the Phase 1 results tables from results/phase1_*.json plus paired significance tests.

Emits a markdown block ready to paste into RESULTS_CANONICAL.md and the external manuscript.
Every row carries mean +/- population std over seeds; every non-baseline row carries the
paired-test verdict against LightGCN.

Usage:
    python make_phase1_report.py [--out results/phase1_report.md]
"""
import argparse
import glob
import json
import os

from significance import compare

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PER_USER_DIR = os.path.join(RESULTS_DIR, "per_user")

ACC = ("recall@20", "ndcg@20", "precision@20")
COLS = [("recall@20", "Recall@20"), ("ndcg@20", "NDCG@20"), ("precision@20", "Precision@20"),
        ("avgpop@20", "AvgPop@20"), ("coverage@20", "Coverage@20"),
        ("aplt@20", "APLT@20"), ("gini@20", "Gini@20")]
TAG = {"LightGCN": "LightGCN", "TSP": "TSP", "PPD v3": "PPDv3", "PPD v4": "PPDv4",
       "PPD v6": "PPDv6"}


def fmt(mean, std, key):
    if key not in mean:
        return "--"
    m, s = mean[key], std.get(key, 0.0)
    prec = 2 if key == "avgpop@20" else 6
    return f"{m:.{prec}f} ± {s:.{prec}f}"


def load(dataset):
    data = {}
    for path in (os.path.join(RESULTS_DIR, f"phase1_{dataset}.json"),
                 os.path.join(RESULTS_DIR, f"phase1_{dataset}_v6.json")):
        if os.path.exists(path):
            with open(path) as f:
                data.update(json.load(f))
    return data


def sig_line(dataset_name, method):
    """Paired verdict vs LightGCN, or a reason it is unavailable."""
    base = os.path.join(PER_USER_DIR, f"{dataset_name}_LightGCN")
    cand = os.path.join(PER_USER_DIR, f"{dataset_name}_{TAG.get(method, method)}")
    if not glob.glob(f"{cand}_seed*.npz") or not glob.glob(f"{base}_seed*.npz"):
        return None
    try:
        return compare(base, cand)
    except SystemExit as exc:
        return {"error": str(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "phase1_report.md"))
    args = ap.parse_args()

    lines = ["# Phase 1 - re-baselined under the frozen protocol", "",
             "Mean ± population std over seeds. AvgPop@20 is computed over **test users**",
             "(Phase 0, finding L); Coverage/APLT/Gini accompany it because AvgPop alone is",
             "gameable. Significance is vs LightGCN, Holm-corrected across the three accuracy",
             "metrics, and reported at two levels: per-user (Wilcoxon, high power, fixed model)",
             "and across-seed (paired t-test, low power, answers 'does it survive retraining').",
             ""]

    for dataset in ("Gowalla", "Yahoo", "KuaiRec", "Coat"):
        data = load(dataset.lower())
        if not data:
            continue
        lines += [f"## {dataset}", "",
                  "| Method | " + " | ".join(c[1] for c in COLS) + " |",
                  "|" + "---|" * (len(COLS) + 1)]
        for method in ("LightGCN", "TSP", "PPD v3", "PPD v4", "PPD v6"):
            if method not in data:
                continue
            mean, std = data[method]["mean"], data[method]["std"]
            cells = [fmt(mean, std, k) for k, _ in COLS]
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
        lines.append("")

        # significance
        rows = []
        for method in ("TSP", "PPD v3", "PPD v4", "PPD v6"):
            if method not in data:
                continue
            rep = sig_line(dataset, method)
            if rep is None:
                rows.append(f"| {method} | per-user dumps missing | | |")
                continue
            if "error" in rep:
                rows.append(f"| {method} | {rep['error']} | | |")
                continue
            for metric in ACC:
                m = metric.split("@")[0]
                pu = rep["per_user"][m]
                ac = rep["across_seed"][m]
                p_pu = pu.get("p_holm", pu.get("p_value"))
                p_ac = ac.get("p_holm", ac.get("p_value"))
                sigma = ac.get("sigma_of_baseline")
                rows.append(
                    f"| {method} | {metric} | {ac['mean_diff']:+.6f} | "
                    f"{'n/a' if p_pu is None else f'{p_pu:.3g}'} | "
                    f"{'n/a' if p_ac is None else f'{p_ac:.3g}'} | "
                    f"{'--' if sigma is None else f'{sigma:+.2f}'} |")
        if rows:
            lines += ["**Significance vs LightGCN** "
                      f"(seeds: {', '.join(data['LightGCN']['seeds'])})", "",
                      "| Method | Metric | Δ mean | p (per-user) | p (across-seed) | σ of baseline |",
                      "|---|---|---|---|---|---|"] + rows + [""]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
