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


### Pareto Evaluation Report: KuaiRec

| Method | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Dominates? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PPD v6 / ASTD-PD (Ours) (Ours)** | **0.177384** | **0.396496** | **0.365071** | **967.43** | - |
| LightGCN | 0.155524 | 0.304831 | 0.274125 | 1024.97 | ✅ YES |
| TSP | 0.155190 | 0.302951 | 0.272861 | 1016.43 | ✅ YES |
| PPD v3 | 0.065414 | 0.119512 | 0.099598 | 461.89 | ❌ NO |
| PPD v4 | 0.155985 | 0.303909 | 0.273404 | 684.22 | ❌ NO |
