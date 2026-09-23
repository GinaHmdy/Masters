# Canonical Results

Every number below is read directly from `LightGCN/code/results/*.json` (**5-seed** mean and
population std on all four datasets as of 2026-08-30) or computed from `data/*/{train,test}.txt`.
Nothing is transcribed from the README, which contains three mutually inconsistent generations
of results.

> **The authoritative per-dataset tables are the generated ones at the end of this file**,
> written by `make_canonical.py` straight from `results/phase1_*.json`. The four hand-written
> sections below have had their stale tables removed and are kept for the *readings* — the
> retractions and protocol history that the generated tables cannot carry.

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
   dataset — 15.75 on Gowalla against PPD v4's 138.44 and PPD v6's 141.60 — while being useless
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

> Numbers: see the generated **Gowalla** table at the end of this file (n = 5).
> Selection: baselines on validation BPR loss, PPD v6 on validation Pareto hypervolume.

**Reading (updated 2026-08-30, n = 5 seeds):**

- Beats LightGCN, TSP and PPD v3 on all four metrics, and the accuracy margins over LightGCN
  now clear **across-seed** significance, which they did not at n = 3:
  Recall +1.6% (p = 0.0115, was p = 0.056), NDCG +4.6% (p = 0.000177), Precision +3.1%
  (p = 0.00139). Per-user Wilcoxon clears all three as well.
- Against PPD v4 the picture is better than "3/4". v6 wins Recall by **+12.5%**
  (0.142878 vs 0.126957) and its AvgPop deficit (141.60 vs 138.44) is **not statistically
  distinguishable from zero** (paired t over 5 seeds, p = 0.214). It also more than doubles
  APLT@20 (0.1665 vs 0.0839) and improves Coverage@20 by 162% (0.3729 vs 0.1423, 2.6x).
- The remaining AvgPop gap **cannot be closed**: an inference-time popularity penalty trades
  Recall for AvgPop point for point, and at the crossover both margins are ties
  (`results/phase4_inference_penalty_gowalla.json`, audit finding **AG**). Validation
  hypervolume selects no penalty at all.
- Comfortably above MostPopular (3.5x). **This is the thesis's defensible result.**

---

## Yahoo (dense)

> Numbers: see the generated **Yahoo** table at the end of this file (n = 5). One row cannot
> be generated and is kept here as a warning:
>
> | Method | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Selection |
> |---|---|---|---|---|---|
> | PPD v6 *(test-selected "Refined")* | 0.142360 | 0.064890 | 0.011239 | 503.47 | **TEST — invalid** |

**Reading (updated 2026-08-30, n = 5 seeds):**

- Under validation selection PPD v6 **loses** Recall, NDCG and Precision to LightGCN — but at
  n = 5 **none of the three losses is statistically significant**: across-seed p = 0.154 /
  0.061 / 0.154, per-user p = 0.092 / 0.0499 / 0.092. The honest statement is *no measurable
  effect on Yahoo accuracy*, not *worse on Yahoo*.
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
  v4's KuaiRec AvgPop is **1158.54 vs LightGCN's 1147.31 — 1.0% worse.** It had been suppressing
  popular items for the ~80% of users no accuracy metric evaluates.
- ~~PPD v6 dominates on real numbers (+14.1% Recall, +30.1% NDCG, -5.6% AvgPop vs LightGCN).~~
  **RETRACTED 2026-08-26 (finding V).** Same scope bug. The accuracy gains are real and
  significant; the fairness claim reverses. Corrected: AvgPop **1606.57 vs 1147.31 (+40% worse)**,
  Coverage@20 **0.0330 vs 0.1531 (-78%)**, APLT@20 0.0108 vs 0.0607 (-82%). PPD v6 wins accuracy
  on KuaiRec by collapsing onto the popular head.
- **The AvgPop-scoped table above is superseded in full.** Use the generated tables further down
  this file, which are scoped to test users throughout.
- **PPD v3's low AvgPop here is degeneracy, not fairness**: it loses to MostPopular on all three
  accuracy metrics (finding **Z**), so its 399.94 AvgPop is what a broken ranker looks like.

---

## Coat (small / dense)

> Numbers: see the generated **Coat** table at the end of this file (n = 5).

**Reading (updated 2026-08-30, n = 5 seeds):**

- PPD v6 trails every baseline on every accuracy metric, but at n = 5 **the deficit is not
  significant**: across-seed p = 0.093 / 0.093 / 0.062, per-user p = 0.14 / 0.058 / 0.14.
  Meanwhile it improves APLT@20 from 0.2207 to 0.3208 (+45%) and Coverage@20 from 0.3720 to
  0.5733 (+54%).
  Coat is a *no-accuracy-cost exposure gain*, on a dataset small enough that nothing here
  is conclusive either way.
- Coat is omitted entirely from the guide's results section while still being counted
  in "four diverse datasets".
- Note also 129 train/test overlapping interactions, which are masked at evaluation
  and therefore make ~10% of the Coat test signal structurally unreachable.

---

## Mechanism ablation — what actually produces the effect

*Generated from `results/phase2_ablation_<dataset>.json`. Six arms x 3 seeds, all four
datasets complete. Gowalla shown; the pattern is the same on Yahoo, KuaiRec and Coat.*

| Arm | Recall@20 | AvgPop@20 | APLT@20 | alpha | theta |
|---|---|---|---|---|---|
| (a) full v6 | 0.142780 ± 0.000695 | 141.03 ± 1.15 | 0.1655 ± 0.0040 | 0.0841 | 0.2560 |
| (b) no spectral projection | 0.142700 ± 0.000610 | 140.92 ± 1.01 | 0.1655 ± 0.0035 | 0.0000 | 0.2559 |
| (c) no simplex stream | 0.149034 ± 0.000469 | 143.18 ± 0.50 | 0.1300 ± 0.0019 | 0.0847 | 0.0000 |
| (d) no tail reweighting | 0.146103 ± 0.000454 | 151.02 ± 1.55 | 0.0871 ± 0.0032 | 0.0838 | 0.2275 |
| (e) recipe only (a-d off) | 0.149980 ± 0.000504 | 149.64 ± 0.49 | 0.0863 ± 0.0011 | 0.0000 | 0.0000 |
| (f) (e) + uniform negatives | 0.150450 ± 0.000440 | 182.28 ± 0.59 | 0.0590 ± 0.0010 | 0.0000 | 0.0000 |

**Reading:**

- **The spectral projection is inert.** Arm (b) matches full v6 to 8e-05 Recall,
  0.11 AvgPop and 0.0000 APLT — inside the non-reproducibility floor (audit **AB**).
  This holds on all four datasets, Gowalla included, which was the last place it could have mattered.
- **The popularity-stratified negative sampler owns 79% of the AvgPop reduction** 
  (182.28 -> 149.64 of the 182.28 -> 141.03 total). On Yahoo the same share is 88-96%.
- **The named mechanisms own 74% of the tail-exposure gain** (0.0863 -> 0.1655 of
  0.0590 -> 0.1655), the tail reweighting contributing most of it.
- **Debiasing costs -0.0077 Recall** (-5.1%) against the no-mechanism arm. The trade is real.

---

## Gate transfer — do the mechanisms adapt, or just decay from where they start?

*Generated from `results/phase3_gate_transfer_<dataset>.json`. 3 seeds per arm.*

| Dataset | native | given alpha_base = 0.40 | given alpha_base = 0.10 |
|---|---|---|---|
| **Gowalla** | 0.0841 of 0.40 = **21.0%** | 0.0841 of 0.40 = **21.0%** | 0.0212 of 0.10 = **21.2%** |
| **Yahoo** | 0.0003 of 0.10 = 0.3% | 0.0019 of 0.40 = 0.5% | 0.0003 of 0.10 = 0.3% |
| **Coat** | 0.0001 of 0.10 = 0.1% | 0.0004 of 0.40 = 0.1% | 0.0001 of 0.10 = 0.1% |

**Reading:** the *retention ratio* is a property of the dataset and is invariant to where
alpha starts — Gowalla keeps ~21% of whatever it is given, Yahoo ~0.4%, Coat ~0.1%. Handed
Gowalla's four-times-larger start, Yahoo still collapses; handed Yahoo's smaller start, Gowalla
still retains. **The mechanisms are genuinely adaptive** (audit finding **AF**). Caveat: the
`astd_topo_gate_init` half of the swap is a dead flag while the gate is learnable, so this
tests `alpha_base` only.

---

## Inference-penalty frontier — the four-metric goal, tested and not met

*Generated from `results/phase4_inference_penalty_gowalla.json`. 5 seeds, Gowalla,
re-ranking the trained checkpoints (`--astd_inference_pop_penalty`, no retraining).*

| mu | validation HV | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | APLT@20 |
|---|---|---|---|---|---|---|
| 0.0 | 0.00049279 **<- selected** | 0.142843 | 0.121314 | 0.043491 | 141.61 | 0.1665 |
| 0.1 | 0.00049206 | 0.142740 | 0.121237 | 0.043458 | 141.42 | 0.1684 |
| 0.25 | 0.00049124 | 0.142581 | 0.121113 | 0.043410 | 141.14 | 0.1712 |
| 0.5 | 0.00048906 | 0.142246 | 0.120893 | 0.043306 | 140.65 | 0.1762 |
| 1.0 | 0.00048442 | 0.141768 | 0.120516 | 0.043149 | 139.64 | 0.1867 |
| 1.25 | 0.00048118 | 0.141437 | 0.120289 | 0.043048 | 139.12 | 0.1924 |
| 1.5 | 0.00047774 | 0.141099 | 0.120069 | 0.042954 | 138.55 | 0.1984 |
| 1.6 | 0.00047594 | 0.140930 | 0.119952 | 0.042907 | 138.33 | 0.2009 |
| 1.75 | 0.00047434 | 0.140721 | 0.119805 | 0.042847 | 137.98 | 0.2047 |
| 1.9 | 0.00047242 | 0.140544 | 0.119675 | 0.042786 | 137.64 | 0.2086 |
| 2.0 | 0.00047114 | 0.140415 | 0.119577 | 0.042746 | 137.40 | 0.2113 |
| 4.0 | 0.00044108 | 0.136654 | 0.116949 | 0.041664 | 131.68 | 0.2755 |

**Reading:** validation hypervolume — the criterion `main.py` already uses for checkpoint
selection — is maximised at **mu = 0** and falls monotonically, so the pre-registered rule keeps
the unpenalised model. The mu = 1.6-1.75 window whose point estimates clear every baseline on all
four metrics is a **statistical tie on both binding margins** (Recall vs LightGCN p = 0.68 and
0.96; AvgPop vs PPD v4 p = 0.96 and 0.85, paired t over 5 seeds). Reporting it as the goal
achieved would be test-set selection at 0.2 sigma. See audit finding **AG**.

---
## Hyperparameter selection — redone on validation (item 5.7)

*Generated from `results/phase5_hpo_*.json` and `results/phase5_final_*.json`. 96 configurations,
selection reading the `[Best Valid Results]` block only; each winner retrained at 5 seeds.*

| Dataset | configs | incumbent rank | winner's valid-HV gain | changed knobs |
|---|---|---|---|---|
| Gowalla | 12 | 5th | +0.39% | alpha 0.40→0.60, damp 0.05→0.10 |
| Yahoo | 36 | 20th | +0.78% | phi 1.0→1.5, alpha 0.10→0.05, damp 0.10→0.05 |
| KuaiRec | 12 | 7th | +0.12% | phi 1.5→1.0, alpha 0.35→0.20 |
| Coat | 36 | 7th | +4.61% | phi 1.5→2.0, alpha 0.10→0.05, damp 0.05→0.10 |

**Reading:**

- **The shipped configuration was never the validation winner on any dataset.** The leak is
  documented, not inferred: `refine_yahoo_ppd_v6.py`'s `score_fn` sorts by seed-2024 *test*
  metrics, and Yahoo's own recorded sweep names a different validation winner (0.15/0.05) than
  the config that shipped (0.10/0.10).
- **Re-selecting honestly changes nothing.** Retrained at 5 seeds, every winner lands within
  0.4% of the shipped configuration on test and no conclusion flips on any dataset. The
  differences that reach paired significance are 0.1-0.4% shifts detected because the seeds
  are matched, not because the effect is meaningful.
- **Gowalla's headline slightly strengthens under honest selection**: Recall@20 vs LightGCN
  moves from p = 0.0115 to **p = 0.00582** (+9.66 sigma), NDCG to 9.21e-05, Precision to
  0.000827 (`results/significance_gowalla_v6_valsel.json`).
- **The search independently reproduces the ablation.** Ranked by validation hypervolume the
  grids sort almost entirely by `astd_tail_phi`; `alpha_base` is flat (Gowalla's 12 configs
  split into a phi=1.5 block and a phi=1.0 block with alpha interleaved inside each; Coat's
  top five differ in alpha and agree on HV to seven decimals). The ablation reached "the
  spectral knob does nothing" by deletion; the HPO landscape reaches it by flatness.
- The shipped configurations stay in `experiment_configs.py`, so every earlier result keeps
  its meaning; the re-selected numbers live in `phase5_final_*.json` beside them.

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
| `results/baselines_gowalla.json` | Gowalla LightGCN R@20 = 0.140533 | **superseded** — pre-Phase-1, 3 seeds. Use `results/phase1_gowalla.json` (0.140693, 5 seeds) |
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
| LightGCN | 5 | 0.140693 ± 0.000243 | 0.115987 ± 0.000301 | 0.042188 ± 0.000072 | 212.99 ± 1.36 | 0.1058 ± 0.0016 | 0.0469 ± 0.0010 | 0.9826 ± 0.0002 |
| TSP | 5 | 0.140337 ± 0.000236 | 0.116880 ± 0.000293 | 0.042418 ± 0.000080 | 209.25 ± 1.29 | 0.1035 ± 0.0011 | 0.0510 ± 0.0011 | 0.9825 ± 0.0002 |
| PPD v3 | 5 | 0.140585 ± 0.000292 | 0.116295 ± 0.000411 | 0.042292 ± 0.000101 | 199.57 ± 1.65 | 0.1069 ± 0.0015 | 0.0474 ± 0.0010 | 0.9821 ± 0.0002 |
| PPD v4 | 5 | 0.126957 ± 0.000877 | 0.106169 ± 0.000991 | 0.037829 ± 0.000353 | 138.44 ± 2.79 | 0.1423 ± 0.0029 | 0.0839 ± 0.0020 | 0.9698 ± 0.0007 |
| PPD v6 | 5 | 0.142878 ± 0.000910 | 0.121314 ± 0.000598 | 0.043501 ± 0.000286 | 141.60 ± 2.00 | 0.3729 ± 0.0050 | 0.1665 ± 0.0049 | 0.9322 ± 0.0015 |

### Yahoo

*dense - test users are 16.6% of all users; AvgPop regenerated under the scope fix. Seeds: 2024, 2025, 2026, 2027, 2028.*

| Method | n | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Coverage@20 | APLT@20 | Gini@20 |
|---|---|---|---|---|---|---|---|---|
| LightGCN | 5 | 0.140833 ± 0.002087 | 0.064305 ± 0.000841 | 0.011041 ± 0.000109 | 593.93 ± 3.37 | 0.6816 ± 0.0041 | 0.1118 ± 0.0029 | 0.8557 ± 0.0021 |
| TSP | 5 | 0.140483 ± 0.002503 | 0.064260 ± 0.000855 | 0.011037 ± 0.000193 | 588.80 ± 3.86 | 0.6806 ± 0.0067 | 0.1159 ± 0.0034 | 0.8525 ± 0.0026 |
| PPD v3 | 5 | 0.141834 ± 0.003029 | 0.064761 ± 0.000861 | 0.011095 ± 0.000202 | 563.96 ± 2.79 | 0.6858 ± 0.0057 | 0.1235 ± 0.0028 | 0.8458 ± 0.0020 |
| PPD v4 | 5 | 0.137602 ± 0.002978 | 0.061553 ± 0.000691 | 0.010760 ± 0.000223 | 505.80 ± 5.06 | 0.7140 ± 0.0041 | 0.1463 ± 0.0034 | 0.8257 ± 0.0026 |
| PPD v6 | 5 | 0.137227 ± 0.003536 | 0.062214 ± 0.000897 | 0.010768 ± 0.000236 | 526.35 ± 11.68 | 0.6182 ± 0.0190 | 0.1430 ± 0.0112 | 0.8284 ± 0.0085 |

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

*small/dense - test users are 96.9% of all users. Seeds: 2024, 2025, 2026, 2027, 2028.*

| Method | n | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Coverage@20 | APLT@20 | Gini@20 |
|---|---|---|---|---|---|---|---|---|
| LightGCN | 5 | 0.124472 ± 0.003662 | 0.069713 ± 0.000987 | 0.022705 ± 0.000511 | 27.56 ± 0.08 | 0.3720 ± 0.0223 | 0.2207 ± 0.0115 | 0.8270 ± 0.0016 |
| TSP | 5 | 0.124472 ± 0.003662 | 0.069713 ± 0.000987 | 0.022705 ± 0.000511 | 27.56 ± 0.08 | 0.3720 ± 0.0223 | 0.2207 ± 0.0115 | 0.8270 ± 0.0016 |
| PPD v3 | 5 | 0.124636 ± 0.002856 | 0.068893 ± 0.000990 | 0.022598 ± 0.000491 | 27.24 ± 0.03 | 0.4127 ± 0.0210 | 0.2290 ± 0.0108 | 0.8242 ± 0.0019 |
| PPD v4 | 5 | 0.124267 ± 0.005312 | 0.069678 ± 0.000797 | 0.022705 ± 0.000430 | 27.32 ± 0.12 | 0.4267 ± 0.0183 | 0.2352 ± 0.0134 | 0.8189 ± 0.0020 |
| PPD v6 | 5 | 0.117744 ± 0.006399 | 0.065158 ± 0.003116 | 0.021673 ± 0.000715 | 25.31 ± 0.95 | 0.5733 ± 0.1322 | 0.3208 ± 0.0471 | 0.7943 ± 0.0294 |

<!-- END GENERATED DATASET TABLES -->
