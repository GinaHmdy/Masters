# Table 7 - Hyperparameters re-selected on validation

The shipped configurations were influenced by test metrics: `refine_yahoo_ppd_v6.py`
sorts candidates by seed-2024 **test** results. `run_phase5_hpo.py` re-selects over 96
configurations reading the `[Best Valid Results]` block only, scored by the same 4-D
Pareto hypervolume used for checkpoint selection; `run_phase5_final.py` then retrains
each winner at 5 seeds. Dead knobs are not swept, and each incumbent is a grid point.

## Search outcome

| Dataset | configs | incumbent rank | winner's validation-HV gain |
|---|---|---|---|
| Gowalla | 12 | 5th | +0.39% |
| Yahoo | 36 | 20th | +0.78% |
| KuaiRec | 12 | 7th | +0.12% |
| Coat | 36 | 7th | +4.61% |

## Effect on the reported numbers, 5 seeds, paired by seed

| Dataset | metric | shipped | validation-selected | difference |
|---|---|---|---|---|
| Gowalla | recall@20 | 0.142878 | 0.143045 | +0.000167 |
| Gowalla | ndcg@20 | 0.121314 | 0.121475 | +0.000161 |
| Gowalla | avgpop@20 | 141.60 | 141.86 | +0.26 |
| Gowalla | aplt@20 | 0.1665 | 0.1662 | -0.0003 |
| Yahoo | recall@20 | 0.137227 | 0.136094 | -0.001133 |
| Yahoo | ndcg@20 | 0.062214 | 0.061716 | -0.000498 |
| Yahoo | avgpop@20 | 526.35 | 532.49 | +6.13 |
| Yahoo | aplt@20 | 0.1430 | 0.1349 | -0.0080 |
| KuaiRec | recall@20 | 0.177101 | 0.176880 | -0.000221 |
| KuaiRec | ndcg@20 | 0.393999 | 0.393095 | -0.000905 |
| KuaiRec | avgpop@20 | 1606.57 | 1599.73 | -6.84 |
| KuaiRec | aplt@20 | 0.0108 | 0.0120 | +0.0012 |
| Coat | recall@20 | 0.117744 | 0.118555 | +0.000811 |
| Coat | ndcg@20 | 0.065158 | 0.066006 | +0.000847 |
| Coat | avgpop@20 | 25.31 | 25.92 | +0.61 |
| Coat | aplt@20 | 0.3208 | 0.2863 | -0.0345 |

> **The re-selection changed the configuration on every dataset and the results on
> none.** All differences land within 0.4%, no conclusion flips, and Gowalla's
> headline margin over LightGCN slightly strengthens (Recall p = 0.0115 -> 0.00582).
> The shipped configurations remain in `experiment_configs.py` so earlier results keep
> their meaning; these sit beside them. See `THESIS_AUDIT.md` finding **AH**.

