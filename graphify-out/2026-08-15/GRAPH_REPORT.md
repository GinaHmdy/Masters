# Graph Report - Masters  (2026-08-15)

## Corpus Check
- 42 files · ~10,396,367 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 416 nodes · 613 edges · 90 communities (25 shown, 65 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 36 edges (avg confidence: 0.55)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1fde1f56`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- utils.py
- BasicDataset
- run_thesis.py
- utilities.py
- LightGCN
- Loader
- LightGCN
- LastFM
- timer
- .rendered.sampling.cpp
- sampling.cpp
- Performance Table by Layer Count
- LightGCN PyTorch Performance Table
- Topological Simplex Propagation
- graphify rule
- KuaiRec unbiased-valid train split
- run.sh
- run_sweep.sh
- Gowalla Dataset
- LightGCN
- Coat Item Interaction Distribution
- Coat Valid Dataset
- Epinions-dummy Test Dataset
- Epinions-dummy Train Dataset
- Epinions Test Dataset
- Epinions Train Dataset
- Gowalla Test Dataset
- Gowalla TopK Predict DAP Dataset
- Gowalla TopK Predict LGN Dataset
- Gowalla Train Dataset
- Gowalla Valid Dataset
- KuaiRand Test Dataset
- KuaiRand Train Dataset
- KuaiRand Item Interaction Distribution
- KuaiRand Valid Dataset
- KuaiRec Biased Valid Test Dataset
- KuaiRec Biased Valid Train Dataset
- KuaiRec Biased Valid Valid Dataset
- KuaiRec_Gen 0 test split
- KuaiRec_Gen 0 train split
- KuaiRec_Gen 0 valid split
- KuaiRec_Gen 10 test split
- KuaiRec_Gen 10 train split
- KuaiRec_Gen 10 valid split
- KuaiRec_Gen 1 test split
- KuaiRec_Gen 1 train split
- KuaiRec_Gen 1 valid split
- KuaiRec_Gen 2 test split
- KuaiRec_Gen 2 train split
- KuaiRec_Gen 2 valid split
- KuaiRec_Gen 3 test split
- KuaiRec_Gen 3 train split
- KuaiRec_Gen 3 valid split
- KuaiRec_Gen 4 test split
- KuaiRec_Gen 4 train split
- KuaiRec_Gen 4 valid split
- KuaiRec_Gen 5 test split
- KuaiRec_Gen 5 train split
- KuaiRec_Gen 5 valid split
- KuaiRec_Gen 6 test split
- KuaiRec Gen 6 Train Split
- KuaiRec Test Bottom 10 Dataset
- KuaiRec Test Bottom 20 Dataset
- KuaiRec Test Bottom 80 Dataset
- KuaiRec Test Dataset
- KuaiRec Test Top 20 Dataset
- KuaiRec TopK Predict DAP Dataset
- KuaiRec TopK Predict LGN Dataset
- KuaiRec Train Dataset
- KuaiRec Head/Mid/Tail Distribution Chart
- KuaiRec Item Interaction Distribution Chart
- KuaiRec Unbiased Valid Test Dataset
- KuaiRec valid split
- Yahoo Head/Mid/Tail Distribution Chart
- Yahoo Item Interaction Distribution Chart
- Yahoo Train Split
- Canonical Reference Table
- numpy
- PPDv6ASTD
- Q: Why does BasicDataset connect all those distinct components across the data loading, model definition, utilities, and propagation layers?
- run_ppd_v6_experiments.py
- benchmark_yahoo_baselines.py
- yahoo_pareto_report.md

## God Nodes (most connected - your core abstractions)
1. `BasicDataset` - 33 edges
2. `LightGCN` - 28 edges
3. `LightGCN` - 22 edges
4. `Loader` - 22 edges
5. `PPDv6ASTD` - 20 edges
6. `LastFM` - 17 edges
7. `PopularityDensityStore` - 16 edges
8. `DummyDataset` - 14 edges
9. `compute_pop_bias_scores()` - 13 edges
10. `timer` - 13 edges

## Surprising Connections (you probably didn't know these)
- `LightGCN` --references--> `torch`  [INFERRED]
  LightGCN/README.md → requirements.txt
- `LightGCN` --uses--> `BasicDataset`  [INFERRED]
  LightGCN/code/DAP.py → LightGCN/code/dataloader.py
- `LightGCN` --uses--> `BasicModel`  [INFERRED]
  LightGCN/code/DAP.py → LightGCN/code/model.py
- `BasicModel` --uses--> `BasicDataset`  [INFERRED]
  LightGCN/code/model.py → LightGCN/code/dataloader.py
- `LightGCN` --uses--> `BasicDataset`  [INFERRED]
  LightGCN/code/model.py → LightGCN/code/dataloader.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Gowalla Dataset Splits** — data_gowalla_test_dataset, data_gowalla_train_dataset, data_gowalla_valid_dataset, data_gowalla_topk_predict_dap_dataset, data_gowalla_topk_predict_lgn_dataset [INFERRED 0.95]
- **KuaiRand Dataset Splits** — data_kuairand_test_dataset, data_kuairand_train_dataset, data_kuairand_valid_dataset [INFERRED 0.95]
- **KuaiRec Dataset Splits** — data_kuairec_test_dataset, data_kuairec_train_dataset, data_kuairec_test_bottom10_dataset, data_kuairec_test_top20_dataset [INFERRED 0.95]
- **KuaiRec_Gen fold 0 splits** — data_kuairec_gen_0_test_split, data_kuairec_gen_0_train_split, data_kuairec_gen_0_valid_split [INFERRED 0.85]
- **KuaiRec_Gen fold 1 splits** — data_kuairec_gen_1_test_split, data_kuairec_gen_1_train_split, data_kuairec_gen_1_valid_split [INFERRED 0.85]
- **KuaiRec_Gen fold 2 splits** — data_kuairec_gen_2_test_split, data_kuairec_gen_2_train_split, data_kuairec_gen_2_valid_split [INFERRED 0.85]

## Communities (90 total, 65 thin omitted)

### Community 0 - "utils.py"
Cohesion: 0.10
Nodes (16): Created on Mar 1, 2020 Pytorch Implementation of LightGCN in Xiangnan He et al.…, Created on Mar 1, 2020 Pytorch Implementation of LightGCN in Xiangnan He et al.…, parse_args(), Created on Mar 1, 2020 Pytorch Implementation of LightGCN in Xiangnan He et al.…, AUC(), NDCGatK_r(), Created on Mar 1, 2020 Pytorch Implementation of LightGCN in Xiangnan He et al.…, test_data should be a list? cause users may have different amount of pos items.… (+8 more)

### Community 1 - "BasicDataset"
Cohesion: 0.09
Nodes (5): Dataset, BasicDataset, not necessary for large dataset it's stupid to return all neg items in super…, build a graph in torch.sparse.IntTensor. Details in NGCF's matrix form A = |I,…, PureMF

### Community 2 - "run_thesis.py"
Cohesion: 0.13
Nodes (28): Graph, calculate_average_popularity(), get_eval_embeddings(), predict_lgbm(), no_grad, Created on Mar 1, 2020 Pytorch Implementation of LightGCN in Xiangnan He et al.…, PPD v6 / ASTD-PD training procedure with stratified negative sampling and…, Computes Average Popularity@K across all evaluated users. (+20 more)

### Community 3 - "utilities.py"
Cohesion: 0.10
Nodes (27): main(), parse_run_output(), run_ppd_v6_single(), compute_bias_scores(), compute_d_pop_all_normalized(), compute_global_popularity_score(), compute_personalized_preference_scores(), compute_pop_bias_scores() (+19 more)

### Community 4 - "LightGCN"
Cohesion: 0.14
Nodes (4): cos_similarity(), LightGCN, Tensor, Compute cosine similarity between two embedding tensors. Args: x: Tensor of…

### Community 5 - "Loader"
Cohesion: 0.10
Nodes (10): Loader, Dataset type for pytorch \n Incldue graph information gowalla dataset, return: dict: {user: [items]}, return: dict: {user: [items]}, users: shape [-1] items: shape [-1] return: feedback [-1], DCDLightGCN, evaluate_dcd(), main() (+2 more)

### Community 6 - "LightGCN"
Cohesion: 0.13
Nodes (5): LightGCN, propagate methods for lightGCN, propagate methods for lightGCN, Computes layer-aggregated dual-stream embeddings. Returns: user_A, item_A,…, Computes full recommendation scores for evaluation using Pareto Fusion.

### Community 7 - "LastFM"
Cohesion: 0.11
Nodes (5): LastFM, return: dict: {user: [items]}, users: shape [-1] items: shape [-1] return: feedback [-1], change dataset mode to offer test data to dataloader, Dataset type for pytorch \n Incldue graph information LastFM dataset

### Community 8 - "timer"
Cohesion: 0.13
Nodes (7): PairWiseModel, Parameters: users: users list pos: positive items for corresponding users neg:…, BPR_train_original(), BPRLoss, minibatch(), shuffle(), timer

### Community 9 - ".rendered.sampling.cpp"
Cohesion: 0.29
Nodes (8): array_t, m, sampling, vector, PYBIND11_MODULE(), randint_(), sample_negative(), sample_negative_ByUser()

### Community 10 - "sampling.cpp"
Cohesion: 0.29
Nodes (8): array_t, m, sampling, vector, PYBIND11_MODULE(), randint_(), sample_negative(), sample_negative_ByUser()

### Community 11 - "Performance Table by Layer Count"
Cohesion: 0.40
Nodes (5): Amazon-book Dataset, Gowalla Dataset, Layer Count Impact on Performance, Performance Table by Layer Count, Yelp2018 Dataset

### Community 12 - "LightGCN PyTorch Performance Table"
Cohesion: 0.67
Nodes (3): Gowalla Dataset, LightGCN PyTorch Performance Table, Yelp2018 Dataset

### Community 13 - "Topological Simplex Propagation"
Cohesion: 0.67
Nodes (3): Degree-Aware Propagation, Popularity Penalty/Preference Centroid, Topological Simplex Propagation

### Community 85 - "PPDv6ASTD"
Cohesion: 0.06
Nodes (23): PopularityDensityStore, Computes and caches popularity and density features for items and users.…, Stratified mixed negative sampler with logQ correction. Draws negatives from…, Samples negatives for a batch of (user_id, pos_item_id). Returns:…, StratifiedPopularityNegativeSampler, BasicModel, PPDv6ASTD, no_grad (+15 more)

### Community 86 - "Q: Why does BasicDataset connect all those distinct components across the data loading, model definition, utilities, and propagation layers?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Why does BasicDataset connect all those distinct components across the data loading, model definition, utilities, and propagation layers?, Source Nodes

### Community 87 - "run_ppd_v6_experiments.py"
Cohesion: 0.60
Nodes (4): main(), parse_output_metrics(), Parses recall, ndcg, precision, avgpop from main.py output., run_experiment()

### Community 88 - "benchmark_yahoo_baselines.py"
Cohesion: 0.90
Nodes (4): eval_run_thesis_baseline(), main(), parse_metrics_from_output(), train_and_eval_lightgcn()

## Knowledge Gaps
- **80 isolated node(s):** `run.sh script`, `run_sweep.sh script`, `Pareto Evaluation Report: Yahoo`, `Answer`, `Outcome` (+75 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **65 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BasicDataset` connect `BasicDataset` to `utils.py`, `LightGCN`, `Loader`, `LightGCN`, `LastFM`, `timer`, `PPDv6ASTD`?**
  _High betweenness centrality (0.158) - this node is a cross-community bridge._
- **Why does `Loader` connect `Loader` to `utils.py`, `BasicDataset`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **Why does `LightGCN` connect `LightGCN` to `utils.py`, `BasicDataset`, `PPDv6ASTD`, `timer`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `BasicDataset` (e.g. with `LightGCN` and `BasicModel`) actually correct?**
  _`BasicDataset` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `LightGCN` (e.g. with `BasicDataset` and `PopularityDensityStore`) actually correct?**
  _`LightGCN` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `LightGCN` (e.g. with `BasicDataset` and `BasicModel`) actually correct?**
  _`LightGCN` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `PPDv6ASTD` (e.g. with `BasicDataset` and `PopularityDensityStore`) actually correct?**
  _`PPDv6ASTD` has 4 INFERRED edges - model-reasoned connections that need verification._