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
| Gowalla | recall@20 | +0.002185 | 0.000887 | 0.0115 | +8.98 | **significant (both)** |
| Gowalla | ndcg@20 | +0.005327 | 8.51e-102 | 0.000177 | +17.68 | **significant (both)** |
| Gowalla | precision@20 | +0.001313 | 1.14e-14 | 0.00139 | +18.18 | **significant (both)** |
| Yahoo | recall@20 | -0.003606 | 0.092 | 0.154 | -1.73 | n.s. |
| Yahoo | ndcg@20 | -0.002090 | 0.0499 | 0.0609 | -2.48 | per-user only |
| Yahoo | precision@20 | -0.000273 | 0.092 | 0.154 | -2.50 | n.s. |
| KuaiRec | recall@20 | +0.021896 | <1e-300 | 8.97e-07 | +51.02 | **significant (both)** |
| KuaiRec | ndcg@20 | +0.089973 | <1e-300 | 8.97e-07 | +73.25 | **significant (both)** |
| KuaiRec | precision@20 | +0.089177 | <1e-300 | 8.97e-07 | +89.09 | **significant (both)** |
| Coat | recall@20 | -0.006728 | 0.143 | 0.0932 | -1.84 | n.s. |
| Coat | ndcg@20 | -0.004555 | 0.0583 | 0.0932 | -4.62 | n.s. |
| Coat | precision@20 | -0.001032 | 0.143 | 0.0618 | -2.02 | n.s. |

**Caution on n=3.** With three seeds a paired t-test has almost no power, and a
near-constant difference across seeds inflates the statistic rather than evidencing
the effect -- which is why Coat precision reports p ~ 1e-30 on a -2.4 sigma difference.
Read sigma as the evidence and the p-value as a sanity check, never the reverse.
