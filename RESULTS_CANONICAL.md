# Canonical Results

Every number below is read directly from `LightGCN/code/results/*.json` (3-seed mean and
population std) or computed from `data/*/{train,test}.txt`. Nothing is transcribed from
the README, which contains three mutually inconsistent generations of results.

**Metric definitions.** Recall@20 / NDCG@20 / Precision@20 as implemented in
`LightGCN/code/utils.py:222-266`. AvgPop@20 = mean training degree of the recommended
items (`Procedure.py:255-296`). Higher is better for the first three; lower for AvgPop.

**Two protocol caveats that apply to every table:**

1. Accuracy is computed over test users only; AvgPop@20 is computed over *all* users
   (`Procedure.py:259` vs `:335`). On Yahoo and KuaiRec these populations overlap by only
   ~17-20%, so the two axes are not measured on the same users.
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
| *MostPopular (reference)* | *0.123342* | *0.313162* | *0.266525* | *1948.98* | *n/a — deterministic* |
| LightGCN | 0.000424 ± 0.000228 | 0.000245 ± 0.000139 | 0.000260 ± 0.000149 | 905.25 ± 21.56 | val (BPR loss) |
| TSP | 0.000372 ± 0.000259 | 0.000214 ± 0.000157 | 0.000225 ± 0.000167 | 897.95 ± 17.33 | val (BPR loss) |
| PPD v3 | 0.005990 ± 0.000879 | 0.005540 ± 0.000984 | 0.004988 ± 0.000848 | 364.56 ± 22.15 | val (BPR loss) |
| PPD v4 | 0.005462 ± 0.000283 | 0.004948 ± 0.000503 | 0.005154 ± 0.000395 | 486.81 ± 10.22 | val (BPR loss) |
| **PPD v6 / ASTD-PD** | **0.013903 ± 0.005774** | **0.021170 ± 0.004745** | **0.023440 ± 0.005743** | **763.47 ± 38.82** | val (hypervolume) |

**Reading:**

- **Every model here loses to MostPopular by ~9x Recall and ~15x NDCG.**
- The LightGCN and TSP rows are collapsed training runs — `lgn-KuaiRec-2-256-best.results.txt`
  records Recall@5 = exactly 0.0. They cannot serve as baselines, so the "32.8x gain"
  claim has no denominator.
- PPD v6's own seed spread is 42% of its mean.
- PPD v3/v4 achieve far lower AvgPop than v6 here, so v6 does not dominate them either.

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

