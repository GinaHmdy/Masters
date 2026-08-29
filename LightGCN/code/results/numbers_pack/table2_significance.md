# Table 2 - Paired significance, PPD v6 vs LightGCN

Two tests, and a result should clear both:

- **per-user**: Wilcoxon signed-rank over per-user metric vectors. High power
  (n = test users) but holds the trained model fixed, so it says nothing about seed variance.
- **across-seed**: paired t-test over seed means. Low power (n = 3) but this is the one
  that answers whether the margin survives retraining.

p-values are Holm-corrected across the three metrics. *sigma* is the mean difference
expressed in baseline seed standard deviations.

| Dataset | Metric | Δ mean | p (per-user) | p (across-seed) | σ | Verdict |
|---|---|---|---|---|---|---|
| Gowalla | recall@20 | +0.002303 | 0.00602 | 0.056 | +12.60 | per-user only |
| Gowalla | ndcg@20 | +0.005530 | 2.17e-64 | 0.0156 | +24.53 | **significant (both)** |
| Gowalla | precision@20 | +0.001336 | 2.03e-09 | 0.0265 | +47.59 | **significant (both)** |
| Yahoo | recall@20 | -0.002958 | 0.549 | 0.539 | -1.16 | n.s. |
| Yahoo | ndcg@20 | -0.001966 | 0.363 | 0.539 | -2.12 | n.s. |
| Yahoo | precision@20 | -0.000168 | 0.549 | 0.539 | -1.61 | n.s. |
| KuaiRec | recall@20 | +0.021896 | <1e-300 | 8.97e-07 | +51.02 | **significant (both)** |
| KuaiRec | ndcg@20 | +0.089973 | <1e-300 | 8.97e-07 | +73.25 | **significant (both)** |
| KuaiRec | precision@20 | +0.089177 | <1e-300 | 8.97e-07 | +89.09 | **significant (both)** |
| Coat | recall@20 | -0.009761 | 0.0577 | 0.0133 | -2.07 | n.s. |
| Coat | ndcg@20 | -0.006750 | 0.008 | 0.00335 | -5.62 | **significant LOSS** |
| Coat | precision@20 | -0.001423 | 0.0734 | 1.98e-30 | -2.42 | n.s. |

**Caution on n=3.** With three seeds a paired t-test has almost no power, and a
near-constant difference across seeds inflates the statistic rather than evidencing
the effect -- which is why Coat precision reports p ~ 1e-30 on a -2.4 sigma difference.
Read sigma as the evidence and the p-value as a sanity check, never the reverse.

*(`significance.py` emitted a low-n warning on at least one row above.)*
