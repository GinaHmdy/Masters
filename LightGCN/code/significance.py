"""
Paired significance testing for method-vs-baseline comparisons.

Why this exists: the audit's finding F recorded that this repository contained no
significance test of any kind, while the write-up claimed "full mean and standard
deviation reporting". Differences were being declared wins at margins well inside the
seed spread -- the Yahoo Recall margin of +0.001115 sits at 0.44x the baseline's own
3-seed std of 0.002547.

Two levels of evidence are produced, and they answer different questions:

  * per-user paired test (Wilcoxon signed-rank, two-sided) -- "does this method beat the
    baseline on the same users?". High power because n = number of test users, but it
    holds the trained model fixed, so it says nothing about seed variance.
  * across-seed test (paired t-test over seed means) -- "does the margin survive
    retraining?". Low power (n = number of seeds) but this is the one that answers the
    question an examiner actually asks.

A result should clear BOTH. Reporting only the first is how a 0.44-sigma difference
becomes a "win".

Usage:
    python significance.py --baseline results/per_user/Gowalla_LightGCN \\
                           --candidate results/per_user/Gowalla_PPDv6 \\
                           --out results/significance_gowalla.json
"""
import argparse
import glob
import json
import os

import numpy as np

try:
    from scipy import stats as _stats
except ImportError:  # scipy is not in requirements.txt; degrade honestly
    _stats = None

METRICS = ("recall", "ndcg", "precision")


def _load_seed_files(prefix):
    """Load every <prefix>_seed*.npz, keyed by seed."""
    out = {}
    for path in sorted(glob.glob(f"{prefix}_seed*.npz")):
        seed = os.path.basename(path).rsplit("_seed", 1)[1].split(".")[0]
        out[seed] = np.load(path)
    if not out:
        raise SystemExit(f"no per-user dumps matching {prefix}_seed*.npz")
    return out


def _align(a, b):
    """Align two per-user dumps on their shared users, preserving order."""
    ua, ub = a["users"], b["users"]
    if len(ua) == len(ub) and np.array_equal(ua, ub):
        return slice(None), slice(None)
    common = np.intersect1d(ua, ub)
    if common.size == 0:
        raise SystemExit("baseline and candidate share no users")
    return np.searchsorted(ua, common, sorter=np.argsort(ua)), \
           np.searchsorted(ub, common, sorter=np.argsort(ub))


def holm(pvals):
    """Holm-Bonferroni step-down adjustment. Returns adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[name] = min(1.0, running)
    return adj


def compare(baseline_prefix, candidate_prefix):
    base = _load_seed_files(baseline_prefix)
    cand = _load_seed_files(candidate_prefix)
    shared = sorted(set(base) & set(cand))
    if not shared:
        raise SystemExit(f"no common seeds: {sorted(base)} vs {sorted(cand)}")

    report = {"seeds": shared, "per_user": {}, "across_seed": {}}

    # ---- per-user paired test, pooled over seeds -------------------------------
    raw_p = {}
    for metric in METRICS:
        diffs = []
        for seed in shared:
            b, c = base[seed], cand[seed]
            ia, ib = _align(b, c)
            diffs.append(np.asarray(c[metric])[ib] - np.asarray(b[metric])[ia])
        d = np.concatenate(diffs)
        entry = {
            "mean_diff": float(d.mean()),
            "n_users": int(d.size),
            "n_better": int((d > 0).sum()),
            "n_worse": int((d < 0).sum()),
            "n_tied": int((d == 0).sum()),
        }
        nz = d[d != 0]
        if _stats is None:
            entry["p_value"] = None
            entry["test"] = "scipy unavailable"
        elif nz.size == 0:
            entry["p_value"] = 1.0
            entry["test"] = "all differences exactly zero"
        else:
            stat, p = _stats.wilcoxon(nz, alternative="two-sided")
            entry["p_value"] = float(p)
            entry["test"] = "wilcoxon signed-rank (two-sided)"
            raw_p[metric] = float(p)
        report["per_user"][metric] = entry
    if raw_p:
        for metric, p_adj in holm(raw_p).items():
            report["per_user"][metric]["p_holm"] = p_adj

    # ---- across-seed paired test ----------------------------------------------
    raw_p = {}
    for metric in METRICS:
        b_means = np.array([np.asarray(base[s][metric]).mean() for s in shared])
        c_means = np.array([np.asarray(cand[s][metric]).mean() for s in shared])
        d = c_means - b_means
        entry = {
            "baseline_mean": float(b_means.mean()), "baseline_std": float(b_means.std(ddof=0)),
            "candidate_mean": float(c_means.mean()), "candidate_std": float(c_means.std(ddof=0)),
            "mean_diff": float(d.mean()), "n_seeds": len(shared),
        }
        # margin expressed in units of the baseline's own seed spread -- the number the
        # audit used to demolish the Yahoo claim
        entry["sigma_of_baseline"] = (
            float(d.mean() / b_means.std(ddof=0)) if b_means.std(ddof=0) > 0 else None)
        if _stats is not None and len(shared) >= 2 and np.any(d != 0):
            stat, p = _stats.ttest_rel(c_means, b_means)
            entry["p_value"] = float(p)
            entry["test"] = "paired t-test over seed means"
            raw_p[metric] = float(p)
            # A paired t-test on 3 seeds reports an extreme p whenever the per-seed
            # differences happen to be near-identical, because the t statistic divides by
            # their (tiny) spread. That is an artifact of n, not evidence. Flag it rather
            # than let a p=1e-30 on three points into the write-up.
            if len(shared) < 5:
                entry["low_n_warning"] = (
                    f"only {len(shared)} seeds; p-value is fragile and an extreme value "
                    f"reflects near-constant differences, not strong evidence")
            spread = float(np.std(d, ddof=0))
            entry["diff_std_across_seeds"] = spread
            if spread > 0 and abs(d.mean()) / spread > 50:
                entry["low_n_warning"] = (
                    entry.get("low_n_warning", "") +
                    " | differences are near-constant across seeds, so the t statistic is "
                    "inflated; treat the effect size, not the p-value, as the evidence")
        else:
            entry["p_value"] = None
            entry["test"] = "insufficient seeds" if len(shared) < 2 else "no difference"
        report["across_seed"][metric] = entry
    if raw_p:
        for metric, p_adj in holm(raw_p).items():
            report["across_seed"][metric]["p_holm"] = p_adj
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="prefix of baseline per-user dumps")
    ap.add_argument("--candidate", required=True, help="prefix of candidate per-user dumps")
    ap.add_argument("--out", default="")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    rep = compare(args.baseline, args.candidate)
    print(f"seeds compared: {', '.join(rep['seeds'])}")
    for level in ("per_user", "across_seed"):
        print(f"\n[{level}]")
        for metric, e in rep[level].items():
            p = e.get("p_holm", e.get("p_value"))
            mark = "" if p is None else ("  SIGNIFICANT" if p < args.alpha else "  n.s.")
            sig = e.get("sigma_of_baseline")
            extra = f"  ({sig:+.2f} sigma)" if sig is not None else ""
            pstr = "n/a" if p is None else f"{p:.3g}"
            print(f"  {metric:10s} diff={e['mean_diff']:+.6f}  p={pstr}{mark}{extra}")
            if e.get("low_n_warning"):
                print(f"             ^ CAUTION: {e['low_n_warning']}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
