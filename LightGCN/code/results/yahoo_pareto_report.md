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


### Pareto Evaluation Report: Yahoo

| Method | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Dominates? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PPD v6 / ASTD-PD (Ours) (Ours)** | **0.138462** | **0.062648** | **0.010841** | **520.28** | - |
| LightGCN | 0.141245 | 0.064582 | 0.010995 | 589.69 | ❌ NO |
| TSP | 0.140413 | 0.064338 | 0.010939 | 585.59 | ❌ NO |
| PPD v3 | 0.141979 | 0.064822 | 0.011044 | 562.20 | ❌ NO |
| PPD v4 | 0.138358 | 0.061830 | 0.010757 | 521.30 | ❌ NO |
