# Table 6 - Inference-penalty frontier (Gowalla)

`--astd_inference_pop_penalty mu` subtracts `mu * p_i` at ranking time, so this sweep
re-ranks the five trained checkpoints rather than retraining anything. **mu is selected
on validation**, by the same 4-D Pareto hypervolume `main.py` uses for checkpoint
selection; the test columns are reported, never used to choose.

The bar a four-metric win must clear is the *strongest baseline on each axis*: recall@20 0.140693, ndcg@20 0.116880, precision@20 0.042418, avgpop@20 138.44.

| mu | validation HV | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | APLT@20 | clears all four? |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.00049279 **<- selected** | 0.142843 | 0.121314 | 0.043491 | 141.61 | 0.1665 | no |
| 0.1 | 0.00049206 | 0.142740 | 0.121237 | 0.043458 | 141.42 | 0.1684 | no |
| 0.25 | 0.00049124 | 0.142581 | 0.121113 | 0.043410 | 141.14 | 0.1712 | no |
| 0.5 | 0.00048906 | 0.142246 | 0.120893 | 0.043306 | 140.65 | 0.1762 | no |
| 1.0 | 0.00048442 | 0.141768 | 0.120516 | 0.043149 | 139.64 | 0.1867 | no |
| 1.25 | 0.00048118 | 0.141437 | 0.120289 | 0.043048 | 139.12 | 0.1924 | no |
| 1.5 | 0.00047774 | 0.141099 | 0.120069 | 0.042954 | 138.55 | 0.1984 | no |
| 1.6 | 0.00047594 | 0.140930 | 0.119952 | 0.042907 | 138.33 | 0.2009 | yes |
| 1.75 | 0.00047434 | 0.140721 | 0.119805 | 0.042847 | 137.98 | 0.2047 | yes |
| 1.9 | 0.00047242 | 0.140544 | 0.119675 | 0.042786 | 137.64 | 0.2086 | no |
| 2.0 | 0.00047114 | 0.140415 | 0.119577 | 0.042746 | 137.40 | 0.2113 | no |
| 4.0 | 0.00044108 | 0.136654 | 0.116949 | 0.041664 | 131.68 | 0.2755 | no |

> **The window that clears all four is a tie, not a win.** Across-seed paired t-tests
> at the crossover: Recall vs LightGCN p = 0.68 (mu=1.6) and 0.96 (mu=1.75); AvgPop vs
> PPD v4 p = 0.96 and 0.85. Validation hypervolume selects mu = 0 regardless. Quoting a
> row from that window as the goal achieved would be test-set selection at 0.2 sigma.
> See `THESIS_AUDIT.md` finding **AG**.

