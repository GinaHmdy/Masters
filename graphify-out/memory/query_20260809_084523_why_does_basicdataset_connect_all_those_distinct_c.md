---
type: "query"
date: "2026-08-09T08:45:23.350383+00:00"
question: "Why does BasicDataset connect all those distinct components across the data loading, model definition, utilities, and propagation layers?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["BasicDataset", "Loader", "LastFM", "LightGCN", "BasicModel", "PairWiseModel"]
---

# Q: Why does BasicDataset connect all those distinct components across the data loading, model definition, utilities, and propagation layers?

## Answer

Expanded from original query via vocab: [basic, dataset, data, loader, dataloader, model, utils, utilities, propagate, propagation]. Then traversed the graph via BFS and found that `BasicDataset` acts as the central data contract for the entire architecture.

- **Data Loading:** It defines the interface for specific dataset loaders (like `Loader` and `LastFM`) in `LightGCN/code/dataloader.py` and provides core methods to access user-item feedback (`getUserItemFeedback()`, `getUserPosItems()`).
- **Model Definition:** Models like `LightGCN`, `BasicModel`, and `PairWiseModel` in `LightGCN/code/model.py` and `DAP.py` directly depend on `BasicDataset` to retrieve the sparse adjacency matrix (`getSparseGraph()`) necessary for Graph Convolution.
- **Propagation Layers:** Advanced propagation mechanisms like `apply_TSP()` (Topological Simplex Propagation) and `inter_simplex_propagation_sparse()` (`LightGCN/code/run_thesis.py`) rely on the graph structure that `BasicDataset` initializes.
- **Utilities:** Metric trackers, debiasing tools, and utilities like `compute_pop_bias_scores()` and `plot_bias_score_histogram()` (`LightGCN/code/utilities.py`) query `BasicDataset` to calculate node degrees and popularity distributions.

## Outcome

- Signal: useful

## Source Nodes

- BasicDataset
- Loader
- LastFM
- LightGCN
- BasicModel
- PairWiseModel