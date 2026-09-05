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


### Pareto Evaluation Report: Coat

| Method | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Dominates? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PPD v6 / ASTD-PD (Ours) (Ours)** | **0.114685** | **0.063222** | **0.021412** | **25.60** | - |
| LightGCN | 0.124446 | 0.069972 | 0.022835 | 27.68 | ❌ NO |
| TSP | 0.124615 | 0.070012 | 0.022894 | 27.71 | ❌ NO |
| PPD v3 | 0.124365 | 0.068772 | 0.022716 | 27.38 | ❌ NO |
| PPD v4 | 0.125553 | 0.070123 | 0.022776 | 27.43 | ❌ NO |
