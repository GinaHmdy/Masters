# Canonical Results

Every number below is read directly from `LightGCN/code/results/*.json` (3-seed mean and
population std) or computed from `data/*/{train,test}.txt`. Nothing is transcribed from
the README, which contains three mutually inconsistent generations of results.

**Metric definitions.** Recall@20 / NDCG@20 / Precision@20 as implemented in
`LightGCN/code/utils.py:222-266`. AvgPop@20 = mean training degree of the recommended
items (`Procedure.py:255-296`). Higher is better for the first three; lower for AvgPop.

> ### PROTOCOL CHANGED 2026-08-25 (Phase 0) - AvgPop numbers below are stale
>
> AvgPop@20 is now computed over **test users only**, matching the accuracy metrics.
> Previously it ran over `range(n_users)`. Measured overlap between the two populations:
> **Gowalla 100%, Coat 96.9%, KuaiRec 19.6%, Yahoo 16.6%**.
>
> Consequence: **Gowalla and Coat AvgPop numbers are unaffected** (their test users are
> essentially all users, verified bit-identical: Gowalla 214.189936 before and after).
> **Every Yahoo and KuaiRec AvgPop number in this file is stale** and is regenerated in
> Phase 1 - e.g. KuaiRec LightGCN moves 1024.97 -> 1154.93 on the same checkpoint.
> Accuracy metrics are unchanged everywhere (Gowalla Recall@20 = 0.14027778 before and after).
>
> Four exposure metrics now accompany AvgPop@20 - **Gini@20, Coverage@20, APLT@20 and
> Tail-Recall@20** - because AvgPop alone is gameable (Random wins it on every dataset).
> First measurement, Gowalla LightGCN: Gini 0.983, Coverage 0.104, APLT 0.046,
> Tail-Recall 0.014. The model touches a tenth of the catalogue and 4.6% of what it shows
> is long-tail, which AvgPop@20 = 214 does not reveal on its own.

**Two protocol caveats that apply to every table:**

1. ~~Accuracy is computed over test users only; AvgPop@20 is computed over *all* users.~~
   **CLOSED 2026-08-25 (finding L).** Both axes now cover the same users.
2. `MostPopular` is a non-personalized reference computed here, masking training items and
   restricted to test users. It is not in any existing report and it is the single most
   important sanity check in this document. Note its AvgPop@20 column is by construction the
   *maximum attainable* popularity, and is computed over test users only, so compare it on
   accuracy — that is what it is there to test.

## Dataset statistics

| Dataset | Type | Test users | Items | Mean test items / user |
|---|---|---|---|---|
| Gowalla | sparse | 29,858 | 40,981 | 7.3 |
| Yahoo | dense | 2,383 | 1,000 | 2.0 |
| KuaiRec | dense | 1,410 | 10,728 | 103.0 |
| Coat | small / dense | 281 | 300 | 4.4 |

The KuaiRec row explains a great deal: 103 relevant items per user out of 10,728 is an
extraordinarily dense test set, which is why a popularity ranker performs so well there.

## Trivial reference rankers

Source: `LightGCN/code/trivial_baselines.py` -> `results/trivial_baselines.json`. Computed
directly from `data/*/{train,test}.txt` with the metric definitions of `utils.py:222-266`,
independently of the model evaluation path. Random is a 3-seed mean.

| Dataset | Method | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 |
|---|---|---|---|---|---|
| Gowalla | MostPopular | 0.040822 | 0.031229 | 0.013635 | 683.99 |
| Gowalla | Random | 0.000486 | 0.000327 | 0.000174 | **15.75** |
| Yahoo | MostPopular | 0.090639 | 0.039015 | 0.007491 | 1083.70 |
| Yahoo | Random | 0.022102 | 0.010000 | 0.002119 | **86.45** |
| KuaiRec | MostPopular | **0.123342** | **0.313162** | **0.266525** | 1948.98 |
| KuaiRec | Random | 0.001838 | 0.009792 | 0.009764 | **86.86** |
| Coat | MostPopular | 0.095386 | 0.051185 | 0.019039 | 38.37 |
| Coat | Random | 0.055013 | 0.029549 | 0.013523 | **11.68** |

**Two things this settles.**

1. **KuaiRec.** MostPopular beats every model in the thesis there by ~9x Recall and ~15x NDCG.
   No result on KuaiRec can be reported without this row present.
2. **AvgPop@20 is not a safe objective on its own.** Random has the *lowest* AvgPop@20 on every
   dataset — 15.75 on Gowalla against PPD v4's 139.86 and PPD v6's 141.11 — while being useless
   at ranking. A metric that a random recommender wins outright cannot be optimised in isolation;
   it is only meaningful jointly with the accuracy metrics, which is precisely what the Pareto
   framing is for. This is the strongest available argument for reporting all four together
   rather than celebrating an AvgPop reduction.

*Discrepancy noted:* Coat MostPopular NDCG@20 is 0.051185 here against 0.051301 quoted earlier in
this file; the difference is tie-breaking order among equal-degree items. The value produced by
`trivial_baselines.py` is canonical because it is reproducible from a committed script.

## Popularity composition (measured, not assumed)

Source: `LightGCN/code/measure_popularity_composition.py` -> `results/popularity_composition.json`.
Coat and Yahoo!R3 ship random-exposure test splits; KuaiRec's test split is a fully observed
matrix. Under uniform exposure an item's positive rate estimates its quality free of exposure
confound, so regressing `log(1+train degree)` on that quality separates the earned component of
popularity from the amplified one.

| Dataset | Unbiased source | Items scored | Degree variance explained by quality ("earned") | Spearman(quality, degree) |
|---|---|---|---|---|
| KuaiRec | fully observed test matrix | 3,247 | 26.9% | +0.7407 |
| Yahoo | R3 random-exposure test | 942 | 27.8% | +0.4334 |
| Coat | uniform-exposure test | 300 | 3.7% | +0.1697 |
| Gowalla | none ships | — | **unmeasurable** | — |

Relevance threshold: rating >= 4. Gowalla is reported as unmeasurable rather than estimated.

**Reading.** Where popularity is substantially earned (KuaiRec, Yahoo), minimising AvgPop@20 is
minimising genuine signal, and debiasing must cost accuracy — which is what those tables show.
Coat is the exception in both directions: its popularity is almost entirely amplified, yet
debiasing does not help there, so Coat's outcome is **not** explained by popularity composition.

**Estimators that were tested and rejected** (they cannot recover per-item quality from the
training log, so no per-item correction is reportable):

| Estimator | Spearman with true quality (KuaiRec / Coat) | Spearman with log degree | Verdict |
|---|---|---|---|
| raw training degree | +0.71 / +0.17 | 1.00 | reference |
| structural proxies (attention share, explorer mix, 2nd-order density) | worse than degree | density rho = 0.98 | collinear with degree; rejected |
| model affinity `aff_i` (`utilities.py:103-111`) | +0.70 / +0.15 | **+0.95** | no better than degree; rejected |

---

## Gowalla (sparse)

| Method | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Selection |
|---|---|---|---|---|---|
| *MostPopular (reference)* | *0.040822* | *0.031229* | *0.013635* | *683.99* | *n/a — deterministic* |
| LightGCN | 0.140533 ± 0.000183 | 0.115921 ± 0.000227 | 0.042134 ± 0.000028 | 214.05 ± 0.15 | val (BPR loss) |
| TSP | 0.140183 ± 0.000103 | 0.116862 ± 0.000282 | 0.042374 ± 0.000029 | 210.26 ± 0.11 | val (BPR loss) |
| PPD v3 | 0.140396 ± 0.000221 | 0.116456 ± 0.000246 | 0.042227 ± 0.000011 | 200.17 ± 1.89 | val (BPR loss) |
| PPD v4 | 0.127509 ± 0.000711 | 0.107013 ± 0.000829 | 0.038061 ± 0.000270 | 139.86 ± 2.28 | val (BPR loss) |
| **PPD v6 / ASTD-PD** | **0.142838 ± 0.000710** | **0.121338 ± 0.000536** | **0.043473 ± 0.000198** | **141.11 ± 1.09** | val (hypervolume) |

**Reading:**

- Beats LightGCN, TSP and PPD v3 on all four metrics. Margins are 3-12x the seed std.
- Misses the goal only against PPD v4 on AvgPop@20 (141.11 vs 139.86, a 0.9% gap),
  while beating PPD v4 on Recall@20 by +12%.
- Comfortably above MostPopular (3.5x). **This is the thesis's defensible result.**

---

## Yahoo (dense)

| Method | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Selection |
|---|---|---|---|---|---|
| *MostPopular (reference)* | *0.090639* | *0.039015* | *0.007491* | *1083.70* | *n/a — deterministic* |
| LightGCN | 0.141245 ± 0.002547 | 0.064582 ± 0.000928 | 0.010995 ± 0.000104 | 589.69 ± 2.83 | val (BPR loss) |
| TSP | 0.140413 ± 0.003045 | 0.064338 ± 0.001016 | 0.010939 ± 0.000183 | 585.59 ± 2.93 | val (BPR loss) |
| PPD v3 | 0.141979 ± 0.003729 | 0.064822 ± 0.000973 | 0.011044 ± 0.000211 | 562.20 ± 2.80 | val (BPR loss) |
| PPD v4 | 0.138358 ± 0.003501 | 0.061830 ± 0.000694 | 0.010757 ± 0.000262 | 521.30 ± 3.97 | val (BPR loss) |
| **PPD v6 / ASTD-PD** | **0.138462 ± 0.000196** | **0.062648 ± 0.000610** | **0.010841 ± 0.000105** | **520.28 ± 12.86** | val (hypervolume) |
| PPD v6 *(test-selected "Refined")* | 0.142360 | 0.064890 | 0.011239 | 503.47 | **TEST — invalid** |

**Reading:**

- Under validation selection PPD v6 **loses** Recall, NDCG and Precision to LightGCN.
- The "Refined" row is the number in the README. It was selected on the test set
  (`refine_yahoo_ppd_v6.py:171-176`) and must not be reported as a result.
- Even the Refined margin over LightGCN (+0.001115 Recall) is 0.44x the baseline's own
  seed std (0.002547). It is noise.
- All 108 sweep configurations show the topological, spectral and fusion gates collapsed
  (see `THESIS_AUDIT.md` §3.D).

---

## KuaiRec (dense)

| Method | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Selection |
|---|---|---|---|---|---|
> **Superseded 2026-08-25.** The numbers previously in this section were produced by an
> evaluation that ranked over the full 10,728-item catalogue. KuaiRec's test split is the
> dense "small matrix" covering only 3,247 items (30.6% of the catalogue), so the model spent
> its top-K on items that appear nowhere in the test set and therefore could never be a hit.
> Every KuaiRec metric was measuring candidate-set overlap, not relevance. Fixed by restricting
> eval candidates to the test-split universe (`restrict_candidates`, on for KuaiRec only;
> Gowalla/Yahoo/Coat cover >94% of their catalogues in test and are unaffected — Gowalla
> re-verified bit-identical). Pre-fix values: `results/baselines_kuairec.PRE-CANDIDATE-MASK.json.bak`.

> **The `AvgPop@20` column in this legacy table is scoped to ALL users and is superseded**
> (finding **L**). Corrected, test-scoped values are in the generated tables below.

| *MostPopular (reference)* | *0.123342* | *0.313162* | *0.266525* | *1948.98* | *n/a — deterministic* |
| LightGCN | 0.155524 ± 0.000104 | 0.304831 ± 0.000918 | 0.274125 ± 0.000963 | 1024.97 ± 22.27 | val (BPR loss) |
| TSP | 0.155190 ± 0.000266 | 0.302951 ± 0.001928 | 0.272861 ± 0.001697 | 1016.43 ± 17.41 | val (BPR loss) |
| PPD v3 | 0.065414 ± 0.006984 | 0.119512 ± 0.009159 | 0.099598 ± 0.007761 | 461.89 ± 29.45 | val (BPR loss) |
| PPD v4 | 0.155985 ± 0.000334 | 0.303909 ± 0.000611 | 0.273404 ± 0.000487 | 684.22 ± 14.31 | val (BPR loss) |
| **PPD v6 / ASTD-PD** | **0.177384 ± 0.000801** | **0.396496 ± 0.004052** | **0.365071 ± 0.004050** | **967.43 ± 5.91** | val (hypervolume) |

**Reading:**

- **The "collapsed LightGCN" was an artifact of the eval protocol, not of training.** The same
  checkpoint that scored Recall@20 = 0.000424 scores **0.155524** once candidates are restricted —
  a 367x correction. There is no representation collapse on KuaiRec to explain or to claim credit
  for preventing.
- LightGCN now **beats** MostPopular (0.155524 vs 0.123342 Recall), which is the expected ordering.
  The earlier "every model loses to MostPopular" reading was a symptom of the same bug.
- **PPD v3 flips from win to loss**: it looked 14x better than LightGCN, it is actually 2.4x worse
  (0.065414 vs 0.155524). Its previous KuaiRec `DOMINANT` verdicts are withdrawn.
- ~~**PPD v4 is a genuine fairness win that the bug was hiding**: accuracy statistically tied with
  LightGCN (0.155985 vs 0.155524) at **-33% AvgPop** (684.22 vs 1024.97).~~ **RETRACTED
  2026-08-25 (finding T).** The -33% was a *second* protocol bug: AvgPop ran over all users while
  accuracy ran over test users, and the two overlap by 19.6% on KuaiRec. Scoped correctly, PPD
  v4's KuaiRec AvgPop is **1158.82 vs LightGCN's 1148.79 — 0.9% worse.** It had been suppressing
  popular items for the ~80% of users no accuracy metric evaluates.
- ~~PPD v6 dominates on real numbers (+14.1% Recall, +30.1% NDCG, -5.6% AvgPop vs LightGCN).~~
  **RETRACTED 2026-08-26 (finding V).** Same scope bug. The accuracy gains are real and
  significant; the fairness claim reverses. Corrected: AvgPop **1616.56 vs 1148.79 (+41% worse)**,
  Coverage@20 **0.0319 vs 0.1582 (-80%)**, APLT@20 0.0102 vs 0.0631 (-84%). PPD v6 wins accuracy
  on KuaiRec by collapsing onto the popular head.
- **The AvgPop-scoped table above is superseded in full.** Use the generated tables further down
  this file, which are scoped to test users throughout.
- **PPD v3's low AvgPop here is degeneracy, not fairness**: it loses to MostPopular on all three
  accuracy metrics (finding **Z**), so its 392.20 AvgPop is what a broken ranker looks like.

---

## Coat (small / dense)

| Method | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Selection |
|---|---|---|---|---|---|
| *MostPopular (reference)* | *0.095386* | *0.051301* | *0.019039* | *38.37* | *n/a — deterministic* |
| LightGCN | 0.124446 ± 0.004717 | 0.069972 ± 0.001202 | 0.022835 ± 0.000587 | 27.68 ± 0.08 | val (BPR loss) |
| TSP | 0.124615 ± 0.005215 | 0.070012 ± 0.001410 | 0.022894 ± 0.000800 | 27.71 ± 0.08 | val (BPR loss) |
| PPD v3 | 0.124365 ± 0.003527 | 0.068772 ± 0.001234 | 0.022716 ± 0.000587 | 27.38 ± 0.02 | val (BPR loss) |
| PPD v4 | 0.125553 ± 0.003955 | 0.070123 ± 0.000674 | 0.022776 ± 0.000524 | 27.43 ± 0.07 | val (BPR loss) |
| **PPD v6 / ASTD-PD** | **0.114685 ± 0.004982** | **0.063222 ± 0.001195** | **0.021412 ± 0.000587** | **25.60 ± 0.75** | val (hypervolume) |

**Reading:**

- PPD v6 **loses to every baseline on every accuracy metric**, by roughly 2 sigma.
- Coat is omitted entirely from the guide's results section while still being counted
  in "four diverse datasets".
- Note also 129 train/test overlapping interactions, which are masked at evaluation
  and therefore make ~10% of the Coat test signal structurally unreachable.

---

## Checkpoints that must not be cited

| Checkpoint | best_epoch | Recall@20 | Problem |
|---|---|---|---|
| `lgn-KuaiRec-2-256-best` | 77 | 0.00074 | collapsed; Recall@5 = 0.0 |
| `lgn-Gowalla-2-256-best` | 0 | 0.00055 | collapsed; never trained past warmup |
| `ppd_v6-Gowalla-2-256-best` | 0 | 0.00052 | collapsed; never trained past warmup |
| `dap-Coat-2-256-best` | -1 | — | `best_test={}` — never evaluated |
| `dap-Gowalla-2-256-best` | -1 | — | `best_test={}` — never evaluated |
| `dap-KuaiRec-2-256-best` | -1 | — | `best_test={}` — never evaluated |

## Reference values in the repo that are NOT measurements

| Location | Value | Status |
|---|---|---|
| `README.md:284` | Gowalla LightGCN R@20 = 0.085641 | superseded protocol |
| `run_component_search.py:16` | Gowalla LightGCN R@20 = 0.121115 | superseded protocol |
| `results/baselines_gowalla.json` | Gowalla LightGCN R@20 = 0.140533 | **current, use this** |
| `run_ppd_v6_experiments.py:99-104` | Coat 0.060000 / 0.032000 / 0.012000 | round-number placeholders, not measured |
| `run_thesis.py:387-391` | hardcoded `BASELINE_POPULARITY` | stale pass/fail gate |

> ### PPD v6 rows regenerated under the corrected selection protocol (2026-08-27)
>
> Audit finding **X**: `main.py` built its validation hypervolume from three validation metrics
> plus an AvgPop scored on **test** users, so every PPD v6 checkpoint was selected with partial
> knowledge of the reporting population. Fixed (`calculate_average_popularity(..., split=...)`),
> and Phase 1 v6 plus the whole Phase 2 ablation were re-run rather than patched.
>
> **Outcome: the numbers did not move.** Yahoo and Coat reproduced bit-identically; Gowalla and
> KuaiRec changed by at most 0.35% on a single metric (APLT@20), one to two orders of magnitude
> inside the seed std. Baseline rows were never affected — they early-stop on validation BPR loss.
>
> The tables below are final for Phase 1.

<!-- BEGIN GENERATED DATASET TABLES -->

*Generated by `LightGCN/code/make_canonical.py` from `results/phase1_*.json`.*
*Mean ± population std over seeds. Do not hand-edit; re-run the generator.*

### Gowalla

*sparse - test users are 100% of all users, so neither protocol fix moves it. Seeds: 2024, 2025, 2026, 2027, 2028.*

| Method | n | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Coverage@20 | APLT@20 | Gini@20 |
|---|---|---|---|---|---|---|---|---|
| LightGCN | 3 | 0.140533 ± 0.000183 | 0.115805 ± 0.000225 | 0.042134 ± 0.000028 | 214.05 ± 0.15 | 0.1047 ± 0.0008 | 0.0463 ± 0.0002 | 0.9828 ± 0.0001 |
| TSP | 3 | 0.140183 ± 0.000103 | 0.116695 ± 0.000173 | 0.042374 ± 0.000029 | 210.26 ± 0.11 | 0.1026 ± 0.0005 | 0.0504 ± 0.0003 | 0.9827 ± 0.0001 |
| PPD v3 | 3 | 0.140396 ± 0.000221 | 0.116075 ± 0.000263 | 0.042227 ± 0.000011 | 200.17 ± 1.89 | 0.1058 ± 0.0009 | 0.0469 ± 0.0002 | 0.9822 ± 0.0002 |
| PPD v4 | 3 | 0.127509 ± 0.000711 | 0.106767 ± 0.000819 | 0.038061 ± 0.000270 | 139.86 ± 2.28 | 0.1404 ± 0.0013 | 0.0827 ± 0.0014 | 0.9703 ± 0.0005 |
| PPD v6 | 5 | 0.142878 ± 0.000910 | 0.121314 ± 0.000598 | 0.043501 ± 0.000286 | 141.60 ± 2.00 | 0.3729 ± 0.0050 | 0.1665 ± 0.0049 | 0.9322 ± 0.0015 |

> **Unequal seed counts on Gowalla** (3 vs 5): LightGCN (n=3), TSP (n=3), PPD v3 (n=3), PPD v4 (n=3), PPD v6 (n=5). Means and stds below are over different numbers of runs, so the ± columns are not directly comparable between rows. Paired significance tests intersect on the shared seeds only and are unaffected.

### Yahoo

*dense - test users are 16.6% of all users; AvgPop regenerated under the scope fix. Seeds: 2024, 2025, 2026.*

| Method | n | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Coverage@20 | APLT@20 | Gini@20 |
|---|---|---|---|---|---|---|---|---|
| LightGCN | 3 | 0.141245 ± 0.002547 | 0.064582 ± 0.000928 | 0.010995 ± 0.000104 | 595.72 ± 3.17 | 0.6800 ± 0.0022 | 0.1101 ± 0.0004 | 0.8572 ± 0.0009 |
| TSP | 3 | 0.140413 ± 0.003045 | 0.064338 ± 0.001016 | 0.010939 ± 0.000183 | 591.13 ± 3.15 | 0.6763 ± 0.0042 | 0.1138 ± 0.0009 | 0.8542 ± 0.0012 |
| PPD v3 | 3 | 0.141979 ± 0.003729 | 0.064822 ± 0.000973 | 0.011044 ± 0.000211 | 565.75 ± 1.92 | 0.6830 ± 0.0033 | 0.1217 ± 0.0007 | 0.8473 ± 0.0007 |
| PPD v4 | 3 | 0.138358 ± 0.003501 | 0.061827 ± 0.000692 | 0.010757 ± 0.000262 | 509.32 ± 2.80 | 0.7127 ± 0.0037 | 0.1443 ± 0.0007 | 0.8275 ± 0.0010 |
| PPD v6 | 3 | 0.138287 ± 0.000375 | 0.062616 ± 0.000584 | 0.010827 ± 0.000095 | 527.13 ± 15.02 | 0.6253 ± 0.0216 | 0.1428 ± 0.0145 | 0.8283 ± 0.0109 |

### KuaiRec

*dense - candidate-restricted AND test-user scoped; both fixes bite here. Seeds: 2024, 2025, 2026, 2027, 2028.*

| Method | n | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Coverage@20 | APLT@20 | Gini@20 |
|---|---|---|---|---|---|---|---|---|
| LightGCN | 5 | 0.155205 ± 0.000429 | 0.304026 ± 0.001228 | 0.273582 ± 0.001001 | 1147.31 ± 6.21 | 0.1531 ± 0.0096 | 0.0607 ± 0.0039 | 0.9747 ± 0.0023 |
| TSP | 5 | 0.154812 ± 0.000524 | 0.302221 ± 0.001796 | 0.272220 ± 0.001538 | 1136.19 ± 10.78 | 0.1536 ± 0.0094 | 0.0616 ± 0.0036 | 0.9744 ± 0.0022 |
| PPD v3 | 5 | 0.067179 ± 0.005863 | 0.121501 ± 0.007597 | 0.101475 ± 0.006615 | 399.94 ± 27.44 | 0.2264 ± 0.0117 | 0.5839 ± 0.0153 | 0.9257 ± 0.0118 |
| PPD v4 | 5 | 0.155779 ± 0.000389 | 0.303060 ± 0.001150 | 0.273085 ± 0.000550 | 1158.54 ± 5.35 | 0.1564 ± 0.0085 | 0.0706 ± 0.0060 | 0.9744 ± 0.0022 |
| PPD v6 | 5 | 0.177101 ± 0.000872 | 0.393999 ± 0.003301 | 0.362759 ± 0.002741 | 1606.57 ± 19.08 | 0.0330 ± 0.0025 | 0.0108 ± 0.0015 | 0.9968 ± 0.0003 |

### Coat

*small/dense - test users are 96.9% of all users. Seeds: 2024, 2025, 2026.*

| Method | n | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Coverage@20 | APLT@20 | Gini@20 |
|---|---|---|---|---|---|---|---|---|
| LightGCN | 3 | 0.124446 ± 0.004717 | 0.069972 ± 0.001202 | 0.022835 ± 0.000587 | 27.53 ± 0.08 | 0.3622 ± 0.0201 | 0.2240 ± 0.0139 | 0.8274 ± 0.0012 |
| TSP | 3 | 0.124446 ± 0.004717 | 0.069972 ± 0.001202 | 0.022835 ± 0.000587 | 27.53 ± 0.08 | 0.3622 ± 0.0201 | 0.2240 ± 0.0139 | 0.8274 ± 0.0012 |
| PPD v3 | 3 | 0.124365 ± 0.003527 | 0.068772 ± 0.001234 | 0.022716 ± 0.000587 | 27.24 ± 0.02 | 0.4067 ± 0.0152 | 0.2320 ± 0.0132 | 0.8246 ± 0.0011 |
| PPD v4 | 3 | 0.125553 ± 0.003955 | 0.070123 ± 0.000674 | 0.022776 ± 0.000524 | 27.29 ± 0.07 | 0.4178 ± 0.0181 | 0.2373 ± 0.0140 | 0.8185 ± 0.0014 |
| PPD v6 | 3 | 0.114685 ± 0.004982 | 0.063222 ± 0.001195 | 0.021412 ± 0.000587 | 25.46 ± 0.76 | 0.5533 ± 0.1416 | 0.3117 ± 0.0406 | 0.7984 ± 0.0269 |

<!-- END GENERATED DATASET TABLES -->
