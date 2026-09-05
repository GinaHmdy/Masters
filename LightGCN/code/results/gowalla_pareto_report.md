> # SUPERSEDED — do not cite
>
> This report was generated before the Phase 0 protocol fixes and the Phase 1 re-baseline.
> Its numbers predate: the AvgPop test-user scope fix (finding **L**), the KuaiRec candidate-set
> restriction (finding **P**), the validation-selection fix in checkpoint scoring (finding **X**),
> the extension to 5 seeds, and the validation-selected hyperparameters (finding **AH**).
>
> **Authoritative numbers:** `RESULTS_CANONICAL.md` and `LightGCN/code/results/numbers_pack/`,
> generated from `results/phase1_*.json`. This file is kept only because the audit's history
> cites it.


### Pareto Evaluation Report: Gowalla

| Method | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Dominates? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PPD v6 / ASTD-PD (Ours) (Ours)** | **0.142838** | **0.121338** | **0.043473** | **141.11** | - |
| LightGCN | 0.140533 | 0.115921 | 0.042134 | 214.05 | ✅ YES |
| TSP | 0.140183 | 0.116862 | 0.042374 | 210.26 | ✅ YES |
| PPD v3 | 0.140396 | 0.116456 | 0.042227 | 200.17 | ✅ YES |
| PPD v4 | 0.127509 | 0.107013 | 0.038061 | 139.86 | ❌ NO |
