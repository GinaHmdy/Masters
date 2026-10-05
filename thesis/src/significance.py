"""
Paired significance tests for a candidate method against a baseline (ported from
LightGCN/code/significance.py, generalised to any metric).

Two levels of evidence, answering different questions; a claimed win should clear both:
  * across seeds -- paired t-test over the per-seed metric values: does the margin survive
    retraining? Low power (n = seeds), but it is the question an examiner asks. Works for any
    metric, including catalogue-level ones (coverage, gini) that have no per-user value.
  * per user -- two-sided Wilcoxon signed-rank on per-user differences, pooled over seeds and
    aligned on user id: does the method beat the baseline on the same users? High power, but it
    holds the trained models fixed.
p-values are Holm-adjusted across the metrics tested at each level.

Inputs are dicts seed -> output of src.evaluate.evaluate (per-user vectors needed only for the
per-user level, i.e. evaluate(..., return_per_user=True)).
"""
import numpy as np
from scipy import stats


def holm(pvals):
    """Holm-Bonferroni step-down adjustment of {name: p}."""
    adj, running = {}, 0.0
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    for i, (name, p) in enumerate(items):
        running = max(running, (len(items) - i) * p)
        adj[name] = min(1.0, running)
    return adj


def across_seeds(base, cand, metric):
    seeds = sorted(set(base) & set(cand))
    b = np.array([base[s][metric] for s in seeds], dtype=np.float64)
    c = np.array([cand[s][metric] for s in seeds], dtype=np.float64)
    d = c - b
    e = {"n_seeds": len(seeds), "baseline_mean": float(b.mean()), "baseline_std": float(b.std()),
         "candidate_mean": float(c.mean()), "candidate_std": float(c.std()),
         "mean_diff": float(d.mean()), "diff_std_across_seeds": float(d.std()),
         # margin in units of the baseline's own seed spread
         "sigma_of_baseline": float(d.mean() / b.std()) if b.std() > 0 else None,
         "p_value": None}
    if len(seeds) >= 2 and np.any(d != 0):
        e["p_value"] = float(stats.ttest_rel(c, b).pvalue)
        e["test"] = "paired t-test over seeds"
        if len(seeds) < 5:
            e["warning"] = f"only {len(seeds)} seeds; the p-value is fragile"
        if d.std() > 0 and abs(d.mean()) / d.std() > 50:
            e["warning"] = ("differences are near-constant across seeds, so t is inflated; "
                            "report the effect size, not the p-value")
    else:
        e["test"] = "insufficient seeds" if len(seeds) < 2 else "no difference"
    return e


def per_user(base, cand, metric):
    diffs = []
    for s in sorted(set(base) & set(cand)):
        b, c = base[s]["per_user"], cand[s]["per_user"]
        common, ib, ic = np.intersect1d(b["users"], c["users"], return_indices=True)
        diffs.append(np.asarray(c[metric])[ic] - np.asarray(b[metric])[ib])
    d = np.concatenate(diffs)
    e = {"n_pairs": int(d.size), "mean_diff": float(d.mean()), "n_better": int((d > 0).sum()),
         "n_worse": int((d < 0).sum()), "n_tied": int((d == 0).sum()), "p_value": None}
    nz = d[d != 0]
    if nz.size == 0:
        e.update(p_value=1.0, test="all differences exactly zero")
    else:
        e.update(p_value=float(stats.wilcoxon(nz, alternative="two-sided").pvalue),
                 test="wilcoxon signed-rank (two-sided), pooled over seeds")
    return e


def compare(base, cand, metrics=("recall@20", "ndcg@20", "precision@20"), user_level=True):
    """Both tests for every metric; per-user only where both sides carry per-user vectors."""
    report = {"seeds": sorted(set(base) & set(cand)), "across_seed": {}, "per_user": {}}
    for m in metrics:
        report["across_seed"][m] = across_seeds(base, cand, m)
        if user_level and all("per_user" in r and m in r["per_user"]
                              for r in [*base.values(), *cand.values()]):
            report["per_user"][m] = per_user(base, cand, m)
    for level in ("across_seed", "per_user"):
        raw = {m: e["p_value"] for m, e in report[level].items() if e["p_value"] is not None
               and e.get("test", "").startswith(("paired", "wilcoxon"))}
        for m, p in holm(raw).items():
            report[level][m]["p_holm"] = p
    return report
