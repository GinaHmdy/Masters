# Graph Report - .  (2026-08-09)

## Corpus Check
- Large corpus: 135 files · ~10,367,633 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 327 nodes · 435 edges · 85 communities (21 shown, 64 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 19 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Lightgcn Code Utils Components
- Lightgcn Code Model Components
- Lightgcn Code Run Components
- Lightgcn Code Utilities Components
- Lightgcn Dap Code Components
- Lightgcn Code Dataloader Components
- Lightgcn Code Model Components
- Lightgcn Code Dataloader Components
- Lightgcn Code Utils Components
- Sampling Lightgcn Code Components
- Sampling Lightgcn Code Components
- Lightgcn Imgs Amazon Components
- Lightgcn Imgs Torch Components
- Readme Dap Ppd Components
- Graphify Agents Rules Components
- Valid Data Kuairec Components
- Lightgcn Code Run Components
- Lightgcn Code Run Components
- Lightgcn Readme Gowalla Components
- Lightgcn Readme Requirements Components
- Coat Dist Data Components
- Data Coat Valid Components
- Data Epinions Dummy Components
- Data Epinions Dummy Components
- Data Epinions Test Components
- Data Epinions Train Components
- Data Gowalla Test Components
- Data Gowalla Topk Components
- Data Gowalla Topk Components
- Data Gowalla Train Components
- Data Gowalla Valid Components
- Data Kuairand Test Components
- Data Kuairand Train Components
- Kuairand Dist Data Components
- Data Kuairand Valid Components
- Data Kuairec Biased Components
- Data Kuairec Biased Components
- Valid Data Kuairec Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Data Kuairec Gen Components
- Train Data Kuairec Components
- Data Kuairec Test Components
- Data Kuairec Test Components
- Data Kuairec Test Components
- Data Kuairec Test Components
- Data Kuairec Test Components
- Data Kuairec Topk Components
- Data Kuairec Topk Components
- Data Kuairec Train Components
- Kuairec Data Train Components
- Kuairec Data Train Components
- Data Kuairec Unbiased Components
- Data Kuairec Valid Components
- Yahoo Data Train Components
- Yahoo Data Train Components
- Train Data Yahoo Components
- Canonical Reference Lightgcn Components
- Requirements Numpy Components

## God Nodes (most connected - your core abstractions)
1. `BasicDataset` - 28 edges
2. `LightGCN` - 24 edges
3. `LightGCN` - 22 edges
4. `Loader` - 19 edges
5. `LastFM` - 17 edges
6. `compute_pop_bias_scores()` - 13 edges
7. `timer` - 13 edges
8. `BasicModel` - 9 edges
9. `PairWiseModel` - 9 edges
10. `apply_TSP()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `LightGCN` --references--> `torch`  [INFERRED]
  LightGCN/README.md → requirements.txt
- `LightGCN` --uses--> `BasicDataset`  [INFERRED]
  LightGCN/code/DAP.py → LightGCN/code/dataloader.py
- `LightGCN` --uses--> `BasicModel`  [INFERRED]
  LightGCN/code/DAP.py → LightGCN/code/model.py
- `LightGCN` --uses--> `BasicDataset`  [INFERRED]
  LightGCN/code/model.py → LightGCN/code/dataloader.py
- `timer` --uses--> `BasicDataset`  [INFERRED]
  LightGCN/code/utils.py → LightGCN/code/dataloader.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Gowalla Dataset Splits** — data_gowalla_test_dataset, data_gowalla_train_dataset, data_gowalla_valid_dataset, data_gowalla_topk_predict_dap_dataset, data_gowalla_topk_predict_lgn_dataset [INFERRED 0.95]
- **KuaiRand Dataset Splits** — data_kuairand_test_dataset, data_kuairand_train_dataset, data_kuairand_valid_dataset [INFERRED 0.95]
- **KuaiRec Dataset Splits** — data_kuairec_test_dataset, data_kuairec_train_dataset, data_kuairec_test_bottom10_dataset, data_kuairec_test_top20_dataset [INFERRED 0.95]
- **KuaiRec_Gen fold 0 splits** — data_kuairec_gen_0_test_split, data_kuairec_gen_0_train_split, data_kuairec_gen_0_valid_split [INFERRED 0.85]
- **KuaiRec_Gen fold 1 splits** — data_kuairec_gen_1_test_split, data_kuairec_gen_1_train_split, data_kuairec_gen_1_valid_split [INFERRED 0.85]
- **KuaiRec_Gen fold 2 splits** — data_kuairec_gen_2_test_split, data_kuairec_gen_2_train_split, data_kuairec_gen_2_valid_split [INFERRED 0.85]

## Communities (85 total, 64 thin omitted)

### Community 0 - "Lightgcn Code Utils Components"
Cohesion: 0.10
Nodes (21): Created on Mar 1, 2020 Pytorch Implementation of LightGCN in Xiangnan He et al.…, Created on Mar 1, 2020 Pytorch Implementation of LightGCN in Xiangnan He et al.…, parse_args(), Created on Mar 1, 2020 Pytorch Implementation of LightGCN in Xiangnan He et al.…, Created on Mar 1, 2020 Pytorch Implementation of LightGCN in Xiangnan He et al.…, compute_pop_bias_scores(), project_and_remove(), Computes popularity bias score (PPD pipeline). Notes: - Adds detailed timing… (+13 more)

### Community 1 - "Lightgcn Code Model Components"
Cohesion: 0.08
Nodes (9): Dataset, BasicDataset, not necessary for large dataset it's stupid to return all neg items in super…, build a graph in torch.sparse.IntTensor. Details in NGCF's matrix form A = |I,…, BasicModel, PairWiseModel, PureMF, Parameters: users: users list pos: positive items for corresponding users neg:… (+1 more)

### Community 2 - "Lightgcn Code Run Components"
Cohesion: 0.16
Nodes (22): Graph, get_eval_embeddings(), predict_lgbm(), store_top_k_predicted_items(), Test(), test_one_batch(), train_lgbm_ranker(), Valid() (+14 more)

### Community 3 - "Lightgcn Code Utilities Components"
Cohesion: 0.11
Nodes (19): compute_bias_scores(), compute_global_popularity_score(), compute_personalized_preference_scores(), get_popular_item_ids(), init_influence_tracking_algorithm(), normalize_scores_global(), plot_bias_score_histogram(), prune_and_filter() (+11 more)

### Community 4 - "Lightgcn Dap Code Components"
Cohesion: 0.15
Nodes (4): cos_similarity(), LightGCN, Compute cosine similarity between two embedding tensors. Args: x: Tensor of…, Tensor

### Community 5 - "Lightgcn Code Dataloader Components"
Cohesion: 0.13
Nodes (5): Loader, Dataset type for pytorch \n Incldue graph information gowalla dataset, return: dict: {user: [items]}, return: dict: {user: [items]}, users: shape [-1] items: shape [-1] return: feedback [-1]

### Community 6 - "Lightgcn Code Model Components"
Cohesion: 0.15
Nodes (4): LightGCN, propagate methods for lightGCN, propagate methods for lightGCN, compute_d_pop_all_normalized()

### Community 7 - "Lightgcn Code Dataloader Components"
Cohesion: 0.11
Nodes (5): LastFM, return: dict: {user: [items]}, users: shape [-1] items: shape [-1] return: feedback [-1], change dataset mode to offer test data to dataloader, Dataset type for pytorch \n Incldue graph information LastFM dataset

### Community 8 - "Lightgcn Code Utils Components"
Cohesion: 0.20
Nodes (4): BPR_train_original(), minibatch(), shuffle(), timer

### Community 9 - "Sampling Lightgcn Code Components"
Cohesion: 0.29
Nodes (8): array_t, m, sampling, vector, PYBIND11_MODULE(), randint_(), sample_negative(), sample_negative_ByUser()

### Community 10 - "Sampling Lightgcn Code Components"
Cohesion: 0.29
Nodes (8): array_t, m, sampling, vector, PYBIND11_MODULE(), randint_(), sample_negative(), sample_negative_ByUser()

### Community 11 - "Lightgcn Imgs Amazon Components"
Cohesion: 0.40
Nodes (5): Amazon-book Dataset, Gowalla Dataset, Layer Count Impact on Performance, Performance Table by Layer Count, Yelp2018 Dataset

### Community 12 - "Lightgcn Imgs Torch Components"
Cohesion: 0.67
Nodes (3): Gowalla Dataset, LightGCN PyTorch Performance Table, Yelp2018 Dataset

### Community 13 - "Readme Dap Ppd Components"
Cohesion: 0.67
Nodes (3): Degree-Aware Propagation, Popularity Penalty/Preference Centroid, Topological Simplex Propagation

## Knowledge Gaps
- **76 isolated node(s):** `run.sh script`, `run_sweep.sh script`, `graphify rule`, `graphify workflow`, `Popularity Penalty/Preference Centroid` (+71 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **64 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BasicDataset` connect `Lightgcn Code Model Components` to `Lightgcn Code Utils Components`, `Lightgcn Dap Code Components`, `Lightgcn Code Dataloader Components`, `Lightgcn Code Model Components`, `Lightgcn Code Dataloader Components`, `Lightgcn Code Utils Components`?**
  _High betweenness centrality (0.153) - this node is a cross-community bridge._
- **Why does `LightGCN` connect `Lightgcn Code Model Components` to `Lightgcn Code Utils Components`, `Lightgcn Code Model Components`, `Lightgcn Code Utils Components`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Why does `Loader` connect `Lightgcn Code Dataloader Components` to `Lightgcn Code Utils Components`, `Lightgcn Code Model Components`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `BasicDataset` (e.g. with `LightGCN` and `BasicModel`) actually correct?**
  _`BasicDataset` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `LightGCN` (e.g. with `BasicDataset` and `BPRLoss`) actually correct?**
  _`LightGCN` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `LightGCN` (e.g. with `BasicDataset` and `BasicModel`) actually correct?**
  _`LightGCN` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `run.sh script`, `run_sweep.sh script`, `graphify rule` to the rest of the system?**
  _76 weakly-connected nodes found - possible documentation gaps or missing edges._