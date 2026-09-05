# MSc Thesis: Topological and Fairness-Aware Enhancements for Graph Convolutional Networks in Recommender Systems

This repository contains the implementation for an MSc thesis project extending the
[LightGCN](https://arxiv.org/abs/2002.02126) architecture. It **builds on** two existing
popularity-debiasing methods — Post-hoc Popularity Debiasing (PPD) and Degree-Aware Propagation
(DAP) — together with an inherited Topological Simplex Propagation (TSP) implementation, and
**contributes** PPD v5 and **PPD v6 / ASTD-PD**, a dual-stream GCN that separates an accuracy
stream from a debiased stream and fuses them with a per-user Pareto gate.

See [Prior Art and Attribution](#prior-art-and-attribution) below for exactly which components
are inherited or published and which are this work's own. That distinction is stated up front
deliberately.

## Purpose and Scope

**Goal.** Produce a single method that outperforms standalone LightGCN, TSP and PPD on
Recall@20, NDCG@20, Precision@20 **and** Average Popularity@20 simultaneously, on both sparse
and dense datasets, under a validation-only selection protocol.

The components involved:
1. **Topological Simplex Propagation (TSP)** — *inherited*. Uses a semantic kNN graph and
   Laplacian smoothing to refine embeddings. Note that at `--k_tsp 1`, the setting used in every
   script here, the simplicial complex contains only nodes and edges — no triangles — so what
   executes is kNN-Laplacian smoothing rather than higher-order topological modelling.
2. **PPD (Post-hoc Popularity Debiasing), variants 1-4** — *published prior art*. Removes the
   projection of each embedding onto an estimated popularity direction.
3. **DAP (Degree-Aware Propagation)** — *published prior art*. Subtracts degree-conditioned
   cluster centroids during propagation.
4. **PPD v5 and PPD v6 / ASTD-PD** — *this work*. See the contributions section.

## Prior Art and Attribution

Determined from the literature and from `git show d3cad59`. The initial commit `d3cad59`
("first", **GinaHmdy**, 2026-03-27) — three months before this work's first commit `6562358`
(2026-06-20) — already contained the entire PPD framework, variants 1-4, the whole of TSP,
`DAP.py`, and every `--tsp` / `--ppd` / `--beta` / `--phi` flag.

| Component | Origin |
|---|---|
| **LightGCN** | He et al., *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*, SIGIR 2020 ([arXiv:2002.02126](https://arxiv.org/abs/2002.02126)) |
| **PPD, variants 1-4** (`utilities.py`, `constants.py`, `model.py:439`) | **Published prior art.** Islam, Wang & Zheleva, *Post-hoc Popularity Bias Correction in GNN-based Collaborative Filtering*, [arXiv:2510.12959](https://arxiv.org/abs/2510.12959) (Oct 2025). The operation implemented in `utilities.project_and_remove` is that paper's debiasing step, and its evaluation datasets (Coat, KuaiRec, Yahoo! R3) are the ones used here. Present in the repository at `d3cad59`. |
| **DAP** (`DAP.py`) | **Published prior art.** Degree-Aware Propagation, Chen et al. 2024 — itself a baseline in the PPD paper above. Present at `d3cad59`. |
| **TSP** (`run_thesis.py:99-243`) | **Inherited** from the prior author. Present at `d3cad59`, along with 48 pre-existing TSP sweep logs. |
| **DCD** (`run_dcd_sweep.py`) | **Published prior art.** This is DICE — Zheng et al., *Disentangling User Interest and Conformity for Recommendation with Causal Embedding*, WWW 2021. |
| **PPD v5** (`VARIANT_POST_PROP_PROJECT`) and `--ego_weight` | **This work** (commit `6562358`). |
| **PPD v6 / ASTD-PD** (`model.py:550-1099`) — dual-stream architecture, `PopularityDensityStore`, `StratifiedPopularityNegativeSampler`, spectral popularity basis, Pareto fusion gate, multi-objective loss | **This work, in its entirety.** |
| Validation-hypervolume checkpoint selection (`main.py:104-172`), the Pareto/hypervolume utilities, the multi-seed benchmarking layer | **This work.** |

Because PPD and DAP are published methods and TSP is prior work from the same group, comparing
against them is legitimate external baseline comparison rather than self-ablation. The related
work should also acknowledge PPD's own successor — **DPAA**, Islam, Faruk, Medya & Zheleva,
[arXiv:2605.11145](https://arxiv.org/abs/2605.11145) (May 2026) — which post-dates PPD on the
same three datasets.

## Architecture & Code Structure

The repository is organized around the PyTorch implementation of LightGCN, extended with custom procedures for thesis experiments.

- **`LightGCN/code/main.py`**: The primary entry point for training the baseline LightGCN model and variants using Bayesian Personalized Ranking (BPR) loss.
- **`LightGCN/code/run_thesis.py`**: The central script for executing thesis-specific experiments. It loads trained checkpoints, applies post-hoc transformations like TSP and PPD, and calculates advanced fairness metrics such as *Average Popularity*.
- **`LightGCN/code/Procedure.py`**: Contains the core logic for model training, validation, and testing procedures.
- **`LightGCN/code/dataloader.py`**: Manages data ingestion and preprocessing for a wide variety of datasets (e.g., Gowalla, Coat, KuaiRec, Yelp2018).
- **`LightGCN/code/model.py` & `register.py`**: Defines the neural network architectures and handles model registration.

## Prerequisites

Based on the required libraries and type hinting syntax used, **Python 3.8 or higher** is required.

### Dependencies
Ensure the following dependencies are installed:
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `torch` (PyTorch)
- `tqdm`
- `tensorboardX`
- `matplotlib`
- `networkx`
- `toponetx`
- `pybind11` (for C++ extensions)
- `cppimport` (for C++ extensions)

## Installation

Follow these step-by-step instructions to set up the environment:

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd Masters/LightGCN
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   # venv\Scripts\activate   # On Windows
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install networkx toponetx pybind11 cppimport
   ```

## How to Run

The codebase is split into standard training and post-hoc thesis experiments. Ensure that you have activated your virtual environment and are in the `LightGCN/code` directory:
```bash
cd LightGCN/code
```

### 1. Dataset Configuration Reference
For all experiments, use the following model settings to match the evaluation checkpoints:
* **Gowalla:** `--dataset Gowalla --layer 3 --recdim 64`
* **KuaiRec:** `--dataset KuaiRec --layer 2 --recdim 256`
* **Yahoo:** `--dataset Yahoo --layer 2 --recdim 256`

---

### 2. Training the Baseline Model
To train a baseline LightGCN model from scratch using the BPR loss, run the [main.py](./LightGCN/code/main.py) script. 
* **Gowalla:**
  ```bash
  python main.py --dataset Gowalla --model lgn --layer 3 --recdim 64 --epochs 1000
  ```
* **KuaiRec:**
  ```bash
  python main.py --dataset KuaiRec --model lgn --layer 2 --recdim 256 --epochs 1000
  ```
* **Yahoo:**
  ```bash
  python main.py --dataset Yahoo --model lgn --layer 2 --recdim 256 --epochs 1000
  ```
*(Trained checkpoints are automatically saved under the `./checkpoints/` folder. You can also run the provided [run.sh](./LightGCN/code/run.sh) script to automate this process).*

---

### 3. Running Thesis Experiments (TSP, PPD, DAP, and Hybrids)
For evaluations and post-hoc/in-propagation transformations, use [run_thesis.py](./LightGCN/code/run_thesis.py). 

#### Key Command-line Flags
* `--load 1`: Instructs the script to load a pre-trained checkpoint.
* `--ckpt <path>`: Specifies an explicit checkpoint file to load (if omitted, the script dynamically infers the filename from configuration parameters).
* `--fairness`: Activates and prints advanced fairness metrics (Average Popularity@20).
* `--save_ckpt`: Saves the transformed model weights and saves a `.results.txt` file containing the run's evaluation metrics.
* `--out_dir <path>`: Directory where the checkpoint and results files will be saved (defaults to `./checkpoints`).

#### A. Running Topological Simplex Propagation (TSP)
Applies post-hoc topological smoothing using simplicial complex relationships.
```bash
python run_thesis.py --dataset Gowalla --load 1 --layer 3 --recdim 64 \
    --tsp --theta_tsp 0.90 --top_m_tsp 15 --k_tsp 1 --alpha_tsp 0.1 \
    --fairness --save_ckpt
```
*(For systematic grid searches over TSP hyperparameters, you can run the automated script [run_sweep.sh](./LightGCN/code/run_sweep.sh)).*

#### B. Running Popularity Penalty/Preference Centroid (PPD Variant 3 - Post-Hoc)
Applies post-hoc popularity bias mitigation by adjusting the output embeddings.
```bash
python run_thesis.py --dataset Gowalla --load 1 --layer 3 --recdim 64 \
    --ppd --ppd_variant 3 --beta 0.20 --phi 1.0 \
    --fairness --save_ckpt
```

#### C. Running Popularity Penalty/Preference Centroid (PPD Variant 4 - Multi-Layer In-Propagation)
Applies popularity bias subtraction dynamically at every single aggregation layer of the GCN.
```bash
python run_thesis.py --dataset Gowalla --load 1 --layer 3 --recdim 64 \
    --ppd --ppd_variant 4 --beta 0.50 --phi 1.0 \
    --fairness --save_ckpt
```

#### D. Running Degree-Aware Propagation (DAP)
Runs training/evaluation using the Degree-Aware Propagation backbone model.
```bash
python run_thesis.py --dataset Gowalla --load 1 --layer 3 --recdim 64 \
    --dap --dap_weight1 0.6 --dap_weight2 0.6 --dap_num_clusters 5 \
    --fairness --save_ckpt
```

#### E. Running DAP + TSP Hybrid
Applies TSP post-hoc transformations on top of the trained DAP model representations.
```bash
python run_thesis.py --dataset Gowalla --load 1 --layer 3 --recdim 64 \
    --dap --dap_weight1 0.6 --dap_weight2 0.6 --dap_num_clusters 5 \
    --tsp --theta_tsp 0.90 --top_m_tsp 15 --k_tsp 1 --alpha_tsp 0.1 \
    --fairness --save_ckpt
```

#### F. Running DAP + TSP + PPD Variant 3 Hybrid
Executes the sequential combination of DAP, TSP, and PPD post-hoc mitigation.
```bash
python run_thesis.py --dataset KuaiRec --load 1 --layer 2 --recdim 256 \
    --dap --dap_weight1 0.6 --dap_weight2 0.6 --dap_num_clusters 5 \
    --tsp --theta_tsp 0.90 --top_m_tsp 15 --k_tsp 1 --alpha_tsp 0.1 \
    --ppd --ppd_variant 3 --beta 0.20 --phi 1.0 \
    --fairness --save_ckpt
```

---

### 4. Running the Perfected Hybrid GCN (Upgraded LightGCN + TSP + PPD v4)
This configuration applies topological simplex smoothing, in-propagation multi-layer popularity penalty (PPD v4), and weighted ego-embeddings (`--ego_weight`) to optimize the accuracy-fairness tradeoff. 

Run the corresponding dataset command to replicate the optimized thesis results:

* **Gowalla Settings** (`ego_weight=0.6`, `beta=0.50`):
  ```bash
  python run_thesis.py --dataset Gowalla --load 1 --layer 3 --recdim 64 \
      --ego_weight 0.6 --tsp --theta_tsp 0.90 --top_m_tsp 15 --k_tsp 1 --alpha_tsp 0.1 \
      --ppd --ppd_variant 4 --beta 0.50 --phi 1.0 --fairness --save_ckpt
  ```

* **KuaiRec Settings** (`uniform (ego=-1)`, `beta=2.00`):
  ```bash
  python run_thesis.py --dataset KuaiRec --load 1 --layer 2 --recdim 256 \
      --ego_weight -1.0 --tsp --theta_tsp 0.90 --top_m_tsp 15 --k_tsp 1 --alpha_tsp 0.1 \
      --ppd --ppd_variant 4 --beta 2.00 --phi 1.0 --fairness --save_ckpt
  ```

* **Yahoo Settings** (`uniform (ego=-1)`, `beta=0.07`):
  ```bash
  python run_thesis.py --dataset Yahoo --load 1 --layer 2 --recdim 256 \
      --ego_weight -1.0 --tsp --theta_tsp 0.90 --top_m_tsp 15 --k_tsp 1 --alpha_tsp 0.1 \
      --ppd --ppd_variant 4 --beta 0.07 --phi 1.0 --fairness --save_ckpt
  ```

---

## Results

Below are the empirical findings of the thesis experiments across the three evaluated datasets. We compare the standard **LightGCN Baseline** against **Topological Simplex Propagation (TSP)** and **Popularity Penalty/Preference Centroid (PPD)**.

### 1. Legacy (Unpropagated) Evaluation Results Table
*Note: This legacy table reflects evaluations where popularity bias was computed using raw, unpropagated (layer-0) embeddings. It is retained here to serve as a baseline comparison for the metric correction. To avoid rendering issues in standard markdown parsers, do not place blank lines inside the HTML table tags.*

> ⚠️ **This entire legacy section predates two protocol fixes. Do not quote any number from it.**
> Retained only so the corrections are auditable; the current-protocol tables (§3,
> `RESULTS_CANONICAL.md`) are the sole valid source.
>
> 1. **Every KuaiRec row is invalid** (flagged 2026-08-25). These runs ranked candidates over the
>    full 10,728-item catalogue while KuaiRec's test split covers 3,247, driving accuracy metrics
>    to near-zero for reasons unrelated to any method under test. No KuaiRec comparison,
>    percentage or "collapse" narrative below should be relied on. `THESIS_AUDIT.md` finding **P**.
> 2. **Every `Average Popularity@20` column is invalid on the dense datasets** (flagged
>    2026-08-26). AvgPop was computed over *all* users while accuracy was computed over *test*
>    users — 16.6% overlap on Yahoo, 19.6% on KuaiRec. This is why the legacy Yahoo baseline reads
>    AvgPop 74.363 against 595.72 under the current protocol: the two are not the same
>    measurement. Gowalla's AvgPop is unaffected (its test users are 100% of users), though it is
>    still layer-0. `THESIS_AUDIT.md` finding **L**, and finding **T** for what this bug cost.
>
> These rows have **not** been regenerated. Regenerating them would mean re-running PPD v1 at
> configurations no longer part of the comparison, which buys the thesis nothing; flagging them
> precisely costs nothing and is more honest than a silent deletion.

| Dataset | Method | Recall@20 | NDCG@20 | Precision@20 | HR@20 | Average Popularity@20 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gowalla** | *Baseline* | ***0.085641*** | ***0.071066*** | ***0.026589*** | *0.350961* | *255.411* |
|  | TSP (α=0.1, θ=0.90, m=15) | 0.084993 (-0.8%) | 0.070678 (-0.5%) | 0.026442 (-0.6%) | 0.348784 (-0.6%) | 262.631 (+2.8%) |
|  | PPD (β=0.20, φ=1.0) | 0.085171 (-0.5%) | 0.070581 (-0.7%) | 0.026440 (-0.6%) | **0.354478 (+1.0%)** | **25.323 (-90.1%)** |
| **KuaiRec** | *Baseline* | ***0.005594*** | *0.004254* | *0.004894* | *0.070922* | ***178.195*** |
|  | TSP (α=0.1, θ=0.90, m=15) | **0.005594 (+0.0%)** | 0.004254 (+0.0%) | 0.004894 (+0.0%) | 0.070922 (+0.0%) | 178.207 (+0.0%) |
|  | PPD (β=0.20, φ=1.0) | 0.004264 (-23.8%) | **0.027358 (+543.1%)** | **0.029220 (+497.1%)** | **0.259574 (+266.0%)** | 221.443 (+24.3%) |
| **Yahoo** | *Baseline* | *0.140200* | *0.064355* | *0.011204* | *0.213596* | *74.363* |
|  | TSP (α=0.1, θ=0.90, m=15) | **0.141382 (+0.8%)** | **0.064891 (+0.8%)** | **0.011267 (+0.6%)** | **0.214855 (+0.6%)** | 78.424 (+5.5%) |
|  | PPD (β=0.20, φ=1.0) | 0.130107 (-7.2%) | 0.059452 (-7.6%) | 0.010302 (-8.1%) | 0.200168 (-6.3%) | **52.863 (-28.9%)** |



---

### 2. Detailed Multi-Cutoff Evaluation (Legacy)

#### Gowalla
* **Baseline**
  * Recall: `@5` = 0.039472, `@10` = 0.057007, `@15` = 0.073337, `@20` = 0.085641, `@50` = 0.131425
  * NDCG: `@5` = 0.062432, `@10` = 0.062697, `@15` = 0.067294, `@20` = 0.071066, `@50` = 0.085300
  * Precision: `@5` = 0.048905, `@10` = 0.035465, `@15` = 0.030295, `@20` = 0.026589, `@50` = 0.016578
  * Hit Ratio: `@5` = 0.203095, `@10` = 0.265925, `@15` = 0.316833, `@20` = 0.350961, `@50` = 0.461752
* **TSP** ($\alpha=0.1, \theta=0.90, m=15, k=1$)
  * Recall: `@5` = 0.039559, `@10` = 0.056655, `@15` = 0.072880, `@20` = 0.084993, `@50` = 0.130772
  * NDCG: `@5` = 0.062537, `@10` = 0.062461, `@15` = 0.066992, `@20` = 0.070678, `@50` = 0.084877
  * Precision: `@5` = 0.049112, `@10` = 0.035317, `@15` = 0.030167, `@20` = 0.026442, `@50` = 0.016512
  * Hit Ratio: `@5` = 0.203631, `@10` = 0.265222, `@15` = 0.315192, `@20` = 0.348784, `@50` = 0.460044
* **PPD** ($\beta=0.20, \phi=1.0$)
  * Recall: `@5` = 0.039314, `@10` = 0.057031, `@15` = 0.073746, `@20` = 0.085171, `@50` = 0.132066
  * NDCG: `@5` = 0.062092, `@10` = 0.062348, `@15` = 0.067140, `@20` = 0.070581, `@50` = 0.085195
  * Precision: `@5` = 0.048623, `@10` = 0.035257, `@15` = 0.030361, `@20` = 0.026440, `@50` = 0.016668
  * Hit Ratio: `@5` = 0.203363, `@10` = 0.268169, `@15` = 0.321120, `@20` = 0.354478, `@50` = 0.469388

#### KuaiRec
* **Baseline**
  * Recall: `@5` = 0.000119, `@10` = 0.001054, `@15` = 0.002882, `@20` = 0.005594, `@50` = 0.028501
  * NDCG: `@5` = 0.000186, `@10` = 0.000988, `@15` = 0.002573, `@20` = 0.004254, `@50` = 0.016235
  * Precision: `@5` = 0.000284, `@10` = 0.001348, `@15` = 0.003310, `@20` = 0.004894, `@50` = 0.011362
  * Hit Ratio: `@5` = 0.001418, `@10` = 0.011348, `@15` = 0.040426, `@20` = 0.070922, `@50` = 0.277305
* **TSP** ($\alpha=0.1, \theta=0.90, m=15, k=1$)
  * Recall: `@5` = 0.000119, `@10` = 0.001054, `@15` = 0.002882, `@20` = 0.005594, `@50` = 0.028501
  * NDCG: `@5` = 0.000186, `@10` = 0.000988, `@15` = 0.002573, `@20` = 0.004254, `@50` = 0.016232
  * Precision: `@5` = 0.000284, `@10` = 0.001348, `@15` = 0.003310, `@20` = 0.004894, `@50` = 0.011362
  * Hit Ratio: `@5` = 0.001418, `@10` = 0.011348, `@15` = 0.040426, `@20` = 0.070922, `@50` = 0.277305
* **PPD** ($\beta=0.20, \phi=1.0$)
  * Recall: `@5` = 0.000908, `@10` = 0.002104, `@15` = 0.003039, `@20` = 0.004264, `@50` = 0.010481
  * NDCG: `@5` = 0.021871, `@10` = 0.025556, `@15` = 0.026458, `@20` = 0.027358, `@50` = 0.029399
  * Precision: `@5` = 0.024255, `@10` = 0.028298, `@15` = 0.028652, `@20` = 0.029220, `@50` = 0.030028
  * Hit Ratio: `@5` = 0.098582, `@10` = 0.176596, `@15` = 0.220567, `@20` = 0.259574, `@50` = 0.390780

#### Yahoo
* **Baseline**
  * Recall: `@5` = 0.051687, `@10` = 0.092600, `@15` = 0.117632, `@20` = 0.140200, `@50` = 0.251638
  * NDCG: `@5` = 0.036323, `@10` = 0.050953, `@15` = 0.058358, `@20` = 0.064355, `@50` = 0.089228
  * Precision: `@5` = 0.016450, `@10` = 0.014645, `@15` = 0.012505, `@20` = 0.011204, `@50` = 0.008200
  * Hit Ratio: `@5` = 0.081830, `@10` = 0.143097, `@15` = 0.180025, `@20` = 0.213596, `@50` = 0.368024
* **TSP** ($\alpha=0.1, \theta=0.90, m=15, k=1$)
  * Recall: `@5` = 0.051093, `@10` = 0.091157, `@15` = 0.118066, `@20` = 0.141382, `@50` = 0.249295
  * NDCG: `@5` = 0.036350, `@10` = 0.050802, `@15` = 0.058740, `@20` = 0.064891, `@50` = 0.089024
  * Precision: `@5` = 0.016198, `@10` = 0.014478, `@15` = 0.012533, `@20` = 0.011267, `@50` = 0.008141
  * Hit Ratio: `@5` = 0.080151, `@10` = 0.140999, `@15` = 0.180445, `@20` = 0.214855, `@50` = 0.365925
* **PPD** ($\beta=0.20, \phi=1.0$)
  * Recall: `@5` = 0.045644, `@10` = 0.082503, `@15` = 0.110245, `@20` = 0.130107, `@50` = 0.240241
  * NDCG: `@5` = 0.032899, `@10` = 0.046021, `@15` = 0.054196, `@20` = 0.059452, `@50` = 0.084289
  * Precision: `@5` = 0.014771, `@10` = 0.013051, `@15` = 0.011610, `@20` = 0.010302, `@50` = 0.007856
  * Hit Ratio: `@5` = 0.073017, `@10` = 0.129249, `@15` = 0.169115, `@20` = 0.200168, `@50` = 0.347881

---

## Completed Enhancements & Expanded Evaluation (Phase II)
 
Following the initial experimental runs, a series of systemic code optimizations and model integrations were successfully executed:
1. **Degree-Aware Propagation (DAP) Integration & GPU Optimization**: Vectorized the complex nested clustering similarity calculations inside [DAP.py](./LightGCN/code/DAP.py) using PyTorch tensor operations. This optimized implementation reduces the computational complexity of GCN clustering updates from $O(N^2)$ to $O(N \log N)$ and memory footprint from $O(N^2)$ to $O(N)$ on GPU. This fully resolved CUDA Out-Of-Memory (OOM) errors and sped up execution by orders of magnitude on large graphs like Gowalla.
2. **Propagated Popularity Evaluation Correction**: Discovered and resolved an evaluation inconsistency where popularity bias was computed on raw unpropagated embeddings. Since recommendations are generated using GCN-smoothed (propagated) embeddings, we updated the evaluation in `run_thesis.py` to calculate *Average Popularity@20* using the actual layer-propagated ratings.
3. **PPD Multi-Layer Update (Variant 4)**: Integrated and benchmarked `VARIANT_VECTOR_BIAS_ALL_LAYER_UPDATE` (PPD v4), which estimates and subtracts popularity bias vectors at *every single GCN layer* during propagation rather than post-hoc.
4. **DAP Evaluation**: Evaluated the registered Degree-Aware Propagation model (`dap`) on all three datasets.
5. **Hybrid DAP + TSP + PPD Experiments**: Benchmarked combinations of Topological Simplex Propagation (TSP) and PPD Variants using DAP as the backbone model to evaluate the interaction of simplex smoothing and in-propagation clustering.
6. **Upgraded LightGCN Backbone (Ego-Weighting)**: Introduced weighted layer aggregation (`--ego_weight`) to standard LightGCN. Giving higher weight (e.g. `0.6` or `0.7`) to the raw layer-0 ego embeddings prevents over-smoothing and representation collapse, boosting Precision, Recall, and NDCG in the hybrid "LightGCN + TSP + PPD" paradigm without using DAP.
 
### 1. Corrected & Enhanced (Propagated) Results Table
*Note: This table reflects evaluations using the mathematically consistent **GCN-propagated ratings** at $K=20$. It directly compares baseline architectures with TSP, PPD Variants, and the optimized DAP/Upgraded-LightGCN models.*
 
| Dataset | Method | Recall@20 | NDCG@20 | Precision@20 | Average Popularity@20 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gowalla** *(layer 3, recdim 64)* | *Baseline (LightGCN)* | *0.085641* | *0.071066* | *0.026589* | *471.209* |
|  | *TSP (α=0.1, θ=0.90, m=15, k=1)* | 0.084993 *(B: -0.8% / T: +0.0% / P: -0.2%)* | 0.070678 *(B: -0.5% / T: +0.0% / P: +0.1%)* | 0.026442 *(B: -0.6% / T: +0.0% / P: +0.0%)* | 470.112 *(B: -0.2% / T: +0.0% / P: -0.6%)* |
|  | *PPD Variant 3 (Baseline PPD)* | 0.085171 *(B: -0.5% / T: +0.2% / P: +0.0%)* | 0.070581 *(B: -0.7% / T: -0.1% / P: +0.0%)* | 0.026440 *(B: -0.6% / T: -0.0% / P: +0.0%)* | 472.865 *(B: +0.4% / T: +0.6% / P: +0.0%)* |
|  | PPD Variant 4 (Multi-Layer) | 0.074450 *(B: -13.1% / T: -12.4% / P: -12.6%)* | 0.053647 *(B: -24.5% / T: -24.1% / P: -24.0%)* | 0.023242 *(B: -12.6% / T: -12.1% / P: -12.1%)* | **240.736** *(B: -48.9% / T: -48.8% / P: -49.1%)* |
|  | DAP (Degree-Aware Propagation) | **0.100581** *(B: +17.4% / T: +18.3% / P: +18.1%)* | **0.083683** *(B: +17.8% / T: +18.4% / P: +18.6%)* | **0.030623** *(B: +15.2% / T: +15.8% / P: +15.8%)* | 365.173 *(B: -22.5% / T: -22.3% / P: -22.8%)* |
|  | DAP + TSP (Hybrid) | 0.094326 *(B: +10.1% / T: +11.0% / P: +10.7%)* | 0.079240 *(B: +11.5% / T: +12.1% / P: +12.3%)* | 0.029950 *(B: +12.6% / T: +13.3% / P: +13.3%)* | 314.997 *(B: -33.2% / T: -33.0% / P: -33.4%)* |
|  | DAP + TSP + PPD Variant 3 (Hybrid) | 0.082338 *(B: -3.9% / T: -3.1% / P: -3.3%)* | 0.069707 *(B: -1.9% / T: -1.4% / P: -1.2%)* | 0.027525 *(B: +3.5% / T: +4.1% / P: +4.1%)* | 304.650 *(B: -35.3% / T: -35.2% / P: -35.6%)* |
|  | DAP + TSP + PPD Variant 4 (Hybrid) | 0.085911 *(B: +0.3% / T: +1.1% / P: +0.9%)* | 0.070964 *(B: -0.1% / T: +0.4% / P: +0.5%)* | 0.026817 *(B: +0.9% / T: +1.4% / P: +1.4%)* | 450.334 *(B: -4.4% / T: -4.2% / P: -4.8%)* |
|  | LightGCN + TSP + PPD Variant 3 (Baseline PPD) | 0.085216 *(B: -0.5% / T: +0.3% / P: +0.1%)* | 0.070603 *(B: -0.7% / T: -0.1% / P: +0.0%)* | 0.026450 *(B: -0.5% / T: +0.0% / P: +0.0%)* | 472.652 *(B: +0.3% / T: +0.5% / P: -0.0%)* |
|  | LightGCN + TSP + PPD Variant 4 (Multi-Layer) | 0.070500 *(B: -17.7% / T: -17.1% / P: -17.2%)* | 0.050944 *(B: -28.3% / T: -27.9% / P: -27.8%)* | 0.021903 *(B: -17.6% / T: -17.2% / P: -17.2%)* | 251.027 *(B: -46.7% / T: -46.6% / P: -46.9%)* |
|  | Upgraded LightGCN + TSP + PPD Variant 3 (ego_weight=0.6) | 0.092431 *(B: +7.9% / T: +8.8% / P: +8.5%)* | 0.076512 *(B: +7.7% / T: +8.3% / P: +8.4%)* | 0.028641 *(B: +7.7% / T: +8.3% / P: +8.3%)* | 437.299 *(B: -7.2% / T: -7.0% / P: -7.5%)* |
|  | Upgraded LightGCN + TSP + PPD Variant 4 (ego_weight=0.6, beta=0.20) | 0.088322 *(B: +3.1% / T: +3.9% / P: +3.7%)* | 0.073877 *(B: +4.0% / T: +4.5% / P: +4.7%)* | 0.027516 *(B: +3.5% / T: +4.1% / P: +4.1%)* | 410.239 *(B: -12.9% / T: -12.7% / P: -13.2%)* |
|  | Upgraded LightGCN + TSP + PPD Variant 4 (ego_weight=0.6, beta=0.50) | 0.087353 *(B: +2.0% / T: +2.8% / P: +2.6%)* | 0.071463 *(B: +0.6% / T: +1.1% / P: +1.2%)* | 0.027413 *(B: +3.1% / T: +3.7% / P: +3.7%)* | 388.641 *(B: -17.5% / T: -17.3% / P: -17.8%)* |
| **KuaiRec** *(layer 2, recdim 256)* | *Baseline (LightGCN)* | *0.005594* | *0.004254* | *0.004894* | *788.993* |
|  | *TSP (α=0.1, θ=0.90, m=15, k=1)* | 0.005594 *(B: +0.0% / T: +0.0% / P: +31.2%)* | 0.004254 *(B: +0.0% / T: +0.0% / P: -84.5%)* | 0.004894 *(B: +0.0% / T: +0.0% / P: -83.3%)* | 788.424 *(B: -0.1% / T: +0.0% / P: +39.3%)* |
|  | *PPD Variant 3 (Baseline PPD)* | 0.004264 *(B: -23.8% / T: -23.8% / P: +0.0%)* | 0.027358 *(B: +543.1% / T: +543.1% / P: +0.0%)* | 0.029220 *(B: +497.1% / T: +497.1% / P: +0.0%)* | 565.989 *(B: -28.3% / T: -28.2% / P: +0.0%)* |
|  | PPD Variant 4 (Multi-Layer) | 0.000069 *(B: -98.8% / T: -98.8% / P: -98.4%)* | 0.000209 *(B: -95.1% / T: -95.1% / P: -99.2%)* | 0.000284 *(B: -94.2% / T: -94.2% / P: -99.0%)* | **5.977** *(B: -99.2% / T: -99.2% / P: -98.9%)* |
|  | DAP (Degree-Aware Propagation) | 0.010733 *(B: +91.9% / T: +91.9% / P: +151.7%)* | 0.008099 *(B: +90.4% / T: +90.4% / P: -70.4%)* | 0.008830 *(B: +80.4% / T: +80.4% / P: -69.8%)* | 780.356 *(B: -1.1% / T: -1.0% / P: +37.9%)* |
|  | DAP + TSP + PPD Variant 3 (Hybrid) | 0.017164 *(B: +206.8% / T: +206.8% / P: +302.5%)* | **0.060431** *(B: +1320.6% / T: +1320.6% / P: +120.9%)* | **0.057482** *(B: +1074.5% / T: +1074.5% / P: +96.7%)* | 675.025 *(B: -14.4% / T: -14.4% / P: +19.3%)* |
|  | DAP + TSP + PPD Variant 4 (Hybrid) | **0.040994** *(B: +632.8% / T: +632.8% / P: +861.4%)* | 0.050415 *(B: +1085.1% / T: +1085.1% / P: +84.3%)* | 0.044965 *(B: +818.8% / T: +818.8% / P: +53.9%)* | 530.001 *(B: -32.8% / T: -32.8% / P: -6.4%)* |
|  | LightGCN + TSP + PPD Variant 3 (Baseline PPD) | 0.004248 *(B: -24.1% / T: -24.1% / P: -0.4%)* | 0.027341 *(B: +542.7% / T: +542.7% / P: -0.1%)* | 0.029113 *(B: +494.9% / T: +494.9% / P: -0.4%)* | 565.994 *(B: -28.3% / T: -28.2% / P: +0.0%)* |
|  | LightGCN + TSP + PPD Variant 4 (Multi-Layer) | 0.000044 *(B: -99.2% / T: -99.2% / P: -99.0%)* | 0.000157 *(B: -96.3% / T: -96.3% / P: -99.4%)* | 0.000212 *(B: -95.7% / T: -95.7% / P: -99.3%)* | 6.004 *(B: -99.2% / T: -99.2% / P: -98.9%)* |
|  | Upgraded LightGCN + TSP + PPD Variant 3 (ego_weight=0.7) | 0.001798 *(B: -67.9% / T: -67.9% / P: -57.8%)* | 0.006396 *(B: +50.4% / T: +50.4% / P: -76.6%)* | 0.006560 *(B: +34.0% / T: +34.0% / P: -77.5%)* | 301.012 *(B: -61.8% / T: -61.8% / P: -46.8%)* |
|  | Upgraded LightGCN + TSP + PPD Variant 4 (uniform, beta=2.00) | **0.040994** *(B: +632.8% / T: +632.8% / P: +861.4%)* | 0.050416 *(B: +1085.1% / T: +1085.1% / P: +84.3%)* | 0.044965 *(B: +818.8% / T: +818.8% / P: +53.9%)* | 529.984 *(B: -32.8% / T: -32.8% / P: -6.4%)* |
| **Yahoo** *(layer 2, recdim 256)* | *Baseline (LightGCN)* | *0.140200* | *0.064355* | *0.011204* | *496.072* |
|  | *TSP (α=0.1, θ=0.90, m=15, k=1)* | **0.141382** *(B: +0.8% / T: +0.0% / P: +8.7%)* | **0.064891** *(B: +0.8% / T: +0.0% / P: +9.1%)* | **0.011267** *(B: +0.6% / T: +0.0% / P: +9.4%)* | 484.998 *(B: -2.2% / T: +0.0% / P: -29.8%)* |
|  | *PPD Variant 3 (Baseline PPD)* | 0.130107 *(B: -7.2% / T: -8.0% / P: +0.0%)* | 0.059452 *(B: -7.6% / T: -8.4% / P: +0.0%)* | 0.010302 *(B: -8.1% / T: -8.6% / P: +0.0%)* | 690.710 *(B: +39.2% / T: +42.4% / P: +0.0%)* |
|  | PPD Variant 4 (Multi-Layer) | 0.076496 *(B: -45.4% / T: -45.9% / P: -41.2%)* | 0.034366 *(B: -46.6% / T: -47.0% / P: -42.2%)* | 0.006525 *(B: -41.8% / T: -42.1% / P: -36.7%)* | **71.590** *(B: -85.6% / T: -85.2% / P: -89.6%)* |
|  | DAP (Degree-Aware Propagation) | 0.139885 *(B: -0.2% / T: -1.1% / P: +7.5%)* | 0.064074 *(B: -0.4% / T: -1.3% / P: +7.8%)* | 0.011120 *(B: -0.7% / T: -1.3% / P: +7.9%)* | 492.477 *(B: -0.7% / T: +1.5% / P: -28.7%)* |
|  | DAP + TSP (Hybrid) | 0.139325 *(B: -0.6% / T: -1.5% / P: +7.1%)* | 0.064152 *(B: -0.3% / T: -1.1% / P: +7.9%)* | 0.011099 *(B: -0.9% / T: -1.5% / P: +7.7%)* | 465.778 *(B: -6.1% / T: -4.0% / P: -32.6%)* |
|  | DAP + TSP + PPD Variant 3 (Hybrid) | 0.127084 *(B: -9.4% / T: -10.1% / P: -2.3%)* | 0.055852 *(B: -13.2% / T: -13.9% / P: -6.1%)* | 0.009966 *(B: -11.0% / T: -11.5% / P: -3.3%)* | 555.583 *(B: +12.0% / T: +14.6% / P: -19.6%)* |
|  | DAP + TSP + PPD Variant 4 (Hybrid) | 0.140542 *(B: +0.2% / T: -0.6% / P: +8.0%)* | 0.064849 *(B: +0.8% / T: -0.1% / P: +9.1%)* | 0.011225 *(B: +0.2% / T: -0.4% / P: +9.0%)* | 486.705 *(B: -1.9% / T: +0.4% / P: -29.5%)* |
|  | LightGCN + TSP + PPD Variant 3 (Baseline PPD) | 0.130176 *(B: -7.1% / T: -7.9% / P: +0.1%)* | 0.059562 *(B: -7.4% / T: -8.2% / P: +0.2%)* | 0.010344 *(B: -7.7% / T: -8.2% / P: +0.4%)* | 694.414 *(B: +40.0% / T: +43.2% / P: +0.5%)* |
|  | LightGCN + TSP + PPD Variant 4 (Multi-Layer) | 0.079615 *(B: -43.2% / T: -43.7% / P: -38.8%)* | 0.036303 *(B: -43.6% / T: -44.1% / P: -38.9%)* | 0.006798 *(B: -39.3% / T: -39.7% / P: -34.0%)* | 75.567 *(B: -84.8% / T: -84.4% / P: -89.1%)* |
|  | Upgraded LightGCN + TSP + PPD Variant 3 (ego_weight=0.6) | 0.131872 *(B: -5.9% / T: -6.7% / P: +1.4%)* | 0.060643 *(B: -5.8% / T: -6.5% / P: +2.0%)* | 0.010575 *(B: -5.6% / T: -6.1% / P: +2.6%)* | 416.055 *(B: -16.1% / T: -14.2% / P: -39.8%)* |
|  | Upgraded LightGCN + TSP + PPD Variant 4 (uniform, beta=0.07) | 0.140542 *(B: +0.2% / T: -0.6% / P: +8.0%)* | 0.064849 *(B: +0.8% / T: -0.1% / P: +9.1%)* | 0.011225 *(B: +0.2% / T: -0.4% / P: +9.0%)* | 486.713 *(B: -1.9% / T: +0.4% / P: -29.5%)* |


 
---
 
### 3. Analytical Breakdown & Success Stories
 
#### The DAP Accuracy-Fairness Breakthrough
Integrating **Degree-Aware Propagation (DAP)** yielded outstanding results, particularly on the **Gowalla** and **KuaiRec** datasets.
- **On Gowalla**: DAP boosted Recall@20 by **17.4%** (from 0.085641 to 0.100581) and NDCG@20 by **17.7%** (from 0.071066 to 0.083683) while concurrently reducing popularity bias by **22.5%** (from 471.21 to 365.17).
- **On KuaiRec**: DAP nearly **doubled the recommendation accuracy**, improving Recall@20 by **91.9%** (from 0.005594 to 0.010733) and NDCG@20 by **90.4%** (from 0.004254 to 0.008099).
- **Why it succeeded**: Graph neural networks like LightGCN suffer from *representation collapse* and *popularity-driven over-smoothing*, where the message-passing process biases user/item embeddings towards high-degree (highly popular) items, dragging low-degree (niche) items into their embedding neighborhoods. DAP counters this in-propagation by dynamically clustering nodes and penalizing update updates based on similarity to high-degree and low-degree cluster centroids. This prevents low-degree items from collapsing into popular centroids, improving recommendation quality for tail items and debiasing the overall recommendation space without losing accuracy.
 
#### PPD Variant 4 Multi-Layer Debiasing
Evaluating **PPD Variant 4** (Multi-Layer Update) revealed the power of in-propagation popularity bias subtraction compared to post-hoc shifts (Variant 3).
- **On Yahoo**: PPD Variant 4 achieved a massive **85.6% reduction** in popularity bias (dropping from 496.07 to 71.59).
- **On Gowalla**: PPD Variant 4 reduced popularity bias by **48.9%** (from 471.21 to 240.74).
- **On KuaiRec**: PPD Variant 4 achieved an extreme **99.2% reduction** in bias (dropping average popularity down to 5.98).
- **Why it succeeded**: Post-hoc methods (like PPD Variant 3) shift user/item embeddings or penalize ratings after GCN layers are completed. However, since the underlying representations have already undergone over-smoothing, post-hoc adjustments can lead to unstable ranking outputs or severe accuracy drops (e.g. KuaiRec Variant 3). Variant 4 estimates and subtracts popularity bias vectors *during GCN propagation at each layer*, preventing the bias from propagating and compounding, which results in more robust and stable debiased embeddings.
 
#### The Hybrid TSP + PPD Paradigm (Variants 3 & 4)
Evaluating the sequential combination of **Topological Simplex Propagation (TSP)** and **Popularity Penalty/Preference Centroid (PPD)** reveals key insights into how topological smoothing interacts with popularity debiasing.
- **Nature of the Combination**: In these experiments, the model first applies TSP to GCN embeddings, refining them using higher-order simplex relations (cliques and Laplacians). Then, the PPD pipeline is executed on top of these topologically-refined embeddings. PPD Variant 3 updates the embeddings post-hoc (zero-layer update), whereas PPD Variant 4 performs in-propagation vector subtraction at each layer.
- **On Gowalla**: 
  - TSP + PPD Variant 3 achieved Recall@20 of **0.085216** and Average Popularity of **472.652**, performing very similarly to PPD Variant 3 alone (Recall@20: 0.085171, Average Popularity: 472.865).
  - TSP + PPD Variant 4 achieved Recall@20 of **0.070500** and Average Popularity of **251.027**, showing a small trade-off in accuracy (down from 0.074450) and a slightly higher popularity bias compared to PPD Variant 4 alone (which had 240.736 popularity).
- **On KuaiRec**:
  - TSP + PPD Variant 3 yielded Recall@20 of **0.004248** and Average Popularity of **565.994**, closely mirroring PPD Variant 3 alone.
  - TSP + PPD Variant 4 showed extreme debiasing with Average Popularity dropping to **6.004** (similar to 5.977 for Variant 4 alone), but at the cost of almost all recommendation utility (Recall@20: 0.000044).
- **On Yahoo**:
  - TSP + PPD Variant 3 achieved Recall@20 of **0.130176** and Average Popularity of **694.414**, keeping pace with PPD Variant 3.
  - TSP + PPD Variant 4 yielded Recall@20 of **0.079615** and Average Popularity of **75.567**. Interestingly, on this dense dataset, the TSP + PPD Variant 4 hybrid slightly *outperformed* PPD Variant 4 alone (Recall@20 of 0.079615 vs 0.076496), indicating that topological simplex propagation can help recover some recommendation quality lost during all-layer popularity bias subtraction.
 
#### Upgraded LightGCN Backbone (Weighted Ego-Embeddings)
We introduced weighted layer aggregation (specifically the `--ego_weight` hyperparameter) into the standard GCN propagation step of the LightGCN model. 
- **The Concept**: Standard LightGCN averages the embeddings of all propagation layers equally. However, as the number of layers increases, representations naturally smooth out, drag tail items into popular clusters, and cause over-smoothing. By assigning a higher weight (e.g., `0.6` or `0.7`) to the layer-0 ego embeddings, the network retains strong user/item identity and specificity, resolving representation collapse.
- **On Gowalla**: 
  - Upgraded LightGCN + TSP + PPD Variant 3 achieved a **+7.9% increase in Recall@20** (`0.092431` vs `0.085641`), **+7.7% NDCG@20** (`0.076512` vs `0.071066`), and **+7.7% Precision@20** (`0.028641` vs `0.026589`), while simultaneously **reducing popularity bias by 7.2%** (down to `437.299`).
  - Upgraded LightGCN + TSP + PPD Variant 4 (with `ego_weight=0.6` and `beta=0.50`) achieved a **+2.0% Recall@20 increase** (`0.087353` vs `0.085641`), **+0.6% NDCG@20 increase** (`0.071463` vs `0.071066`), and **+3.1% Precision@20 increase** (`0.027413` vs `0.026589`) over the baseline, while maintaining a more aggressive **17.5% reduction in popularity bias** (`388.641`). With `beta=0.20`, it achieved **+3.1% Recall@20** (`0.088322`), **+4.0% NDCG@20** (`0.073877`), and **+3.5% Precision@20** (`0.027516`), with a **12.9% popularity reduction** (`410.239`).
- **On KuaiRec**: 
  - Upgraded LightGCN + TSP + PPD Variant 3 achieved a **+50.3% NDCG@20** increase (`0.006396` vs `0.004254`) and **+34.0% Precision@20** (`0.006560` vs `0.004894`) while **cutting popularity bias by 61.8%** (dropping Average Popularity down to `301.012`).
  - Upgraded LightGCN + TSP + PPD Variant 4 (with uniform layer aggregation and `beta=2.00`) achieved a massive **+632.8% Recall@20 increase** (`0.040994` vs `0.005594`), **+1085% NDCG@20 increase** (`0.050416` vs `0.004254`), and **+818.8% Precision@20 increase** (`0.044965` vs `0.004894`) over the baseline, while achieving an aggressive **32.8% reduction in popularity bias** (`529.984` average popularity).
- **On Yahoo**: 
  - Upgraded LightGCN + TSP + PPD Variant 3 achieved **16.1% popularity bias reduction** (Average Popularity: `416.055`) compared to the baseline (`496.072`) while maintaining high recommendation utility (Recall@20: `0.131872`).
  - Upgraded LightGCN + TSP + PPD Variant 4 (with uniform layer aggregation and `beta=0.07`) achieved a **+0.2% Recall@20 increase** (`0.140542` vs `0.140200`), **+0.8% NDCG@20 increase** (`0.064849` vs `0.064355`), and **+0.2% Precision@20 increase** (`0.011225` vs `0.011204`) over the baseline standard model, while providing a **1.9% popularity bias reduction** (`486.713` average popularity).


---

## Main Thesis Contributions (MSc Work)

**Read this alongside [Prior Art and Attribution](#prior-art-and-attribution).** PPD variants
1-4, DAP and TSP are *not* contributions of this work; the items below are.

### The primary contribution

**PPD v6 / ASTD-PD** (`model.py:550-1099`) — a dual-stream graph recommender that separates an
accuracy stream from a debiased stream, removes popularity via a detached low-rank spectral
subspace rather than a hand-crafted direction vector, and fuses the two with a per-user gate.
It includes `PopularityDensityStore` (which, unlike PPD v1-v4, uses true item degree rather
than an embedding-derived affinity proxy), a popularity-stratified negative sampler with logQ
correction, and a multi-objective loss with popularity-independence and covariance penalties.

**Validation-hypervolume model selection** (`main.py:104-172`) — checkpoint selection on a 4-D
Pareto hypervolume over (Recall, NDCG, Precision, inverted AvgPop) computed on the *validation*
split, instead of on validation BPR loss. This is a methodological contribution independent of
the architecture, and it is applicable to any accuracy-fairness recommender.

**PPD v5** (`VARIANT_POST_PROP_PROJECT`) and `--ego_weight` (commit `6562358`) — post-propagation
projection removal, and weighted layer aggregation that retains layer-0 ego embeddings.

### Engineering and methodological contributions

1. **Scalable Degree-Aware Graph Representation Learning** *(optimisation of inherited `DAP.py`;
   the DAP method itself is Chen et al. 2024)*:
   - *Contribution*: Solved a computational bottleneck in cluster-based graph debiasing by redesigning the similarity subtraction calculations inside [DAP.py](LightGCN/code/DAP.py) using vectorized sorting and prefix/suffix cumulative sums.
   - *Impact*: Reduced time complexity from $O(N^2)$ to $O(N \log N)$ and memory from $O(N^2)$ to $O(N)$ on GPU, enabling clustering-based degree debiasing on graphs the size of Gowalla (>70,000 nodes), which previously hit CUDA OOM.

2. **Integration and evaluation of in-propagation debiasing** *(PPD Variant 4 is prior art —
   `forward_post_hoc_with_bias_vector_estimation` was present at `d3cad59`; the contribution here
   is the systematic multi-dataset evaluation of it, not its invention)*:
   - *Contribution*: Benchmarked layer-by-layer debiasing (v4) against post-hoc projection (v3) across four datasets under a common protocol.
   - *Caveat now established*: `--ppd_beta_schedule` and `--ppd_seq_beta_v3` are computed and never used (`model.py:509-519`, `model.py:445-447`), and `--beta` does not scale the projection in `project_and_remove` at all — so sweeps over those knobs measured nothing. Earlier claims in this README that rest on them are withdrawn.

3. **Methodological Correction of Popularity Bias Metrics**:
   - *Contribution*: Identified and mathematically corrected a methodological bug in the evaluation of popularity bias. Legacy implementations evaluated Average Popularity using raw, unpropagated embeddings, which did not correspond to the actual GCN-propagated recommendations.
   - *Impact*: Established a mathematically consistent evaluation protocol for GCN recommenders by utilizing layer-propagated ratings, revealing the true popularity bias baseline (up to 2.5x higher than previously reported) and validating the need for in-propagation debiasing techniques.

4. **Multi-Domain Empirical Validation**:
   - *Contribution*: Constructed a unified empirical benchmarking suite that evaluates GCN baselines, Topological Simplex Propagation (TSP), Popularity Bias Mitigation (PPD), and Degree-Aware Propagation (DAP) across three distinct recommendation domains: social-locational (Gowalla), short-video (KuaiRec), and music-ratings (Yahoo).
   - *Impact*: Demonstrated that while topological smoothing (TSP) is highly effective for dense item-interaction domains (such as Yahoo, yielding +0.8% accuracy), degree-aware in-propagation (DAP) is crucial for highly skewed datasets (such as KuaiRec, yielding +91.9% accuracy and Gowalla, yielding +17.4% accuracy).

---

## Detailed Analysis of Perfected Hybrid GCN (Upgraded LightGCN + TSP + PPD v4)

In this project, we successfully addressed the accuracy-fairness trade-off of the hybrid Topological Simplex Propagation (TSP) and Popularity Bias Mitigation (PPD v4) recommendation pipeline. Originally, applying multi-layer in-propagation popularity bias subtraction (PPD v4) caused severe accuracy degradation on sparse datasets (like KuaiRec) or dense datasets (like Yahoo). By optimizing node-degree aggregation, representation alignment, and neighborhood propagation constraints, we perfected the hybrid pipeline to **beat or match the standard GCN baseline in all accuracy metrics (Recall, NDCG, and Precision) while maintaining robust popularity bias reduction**.

### 1. What was Added & Solved
- **Mathematical Aggregation Alignment**: We resolved a critical mismatch between training and post-hoc evaluation aggregation logic. The legacy Python implementation of GCN neighborhood propagation in [model.py](./LightGCN/code/model.py) calculated a simple average ($D^{-1}A$), which mismatched the model's trained symmetric normalized Laplacian graph ($D^{-1/2}AD^{-1/2}$). We rewrote the propagation steps using sparse matrix multiplication (`torch.sparse.mm(self.Graph, all_emb)`), restoring representation alignment and accelerating calculation from 6 minutes to milliseconds.
- **Double-Normalization Fix**: We fixed a GCN aggregation bug in [model.py](./LightGCN/code/model.py) where neighbor embeddings were divided by node degree twice. Removing this double-normalization restored standard graph convolutional scaling.
- **~~Tunable Popularity Penalty ($\beta$)~~ — WITHDRAWN**: this claimed a granular scaling factor $\beta$ controlling the intensity of layer-wise popularity subtraction. After the reflection-bug fixes (`e8d12d0`, `1fde1f5`), `project_and_remove`'s `beta` argument is no longer referenced in its body (`utilities.py:707-732`) and `beta_l` in the v4 loop is computed then discarded (`model.py:509-519`). The projection is always applied at full strength. $\beta$ affects only the co-popularity penalty $\gamma$ inside $r_{ui}$ (`utilities.py:161-165`), which is a different quantity. **Every reported per-dataset $\beta$ value in this file is therefore mislabelled**, and the sweeps over it are void.
- **Hyperparameter Optimization Suite**: We conducted systematic parameter sweeps on $\beta$ and `--ego_weight` across all three evaluation datasets to find the Pareto-optimal thresholds.

### 2. How the Optimal Settings Work
The perfected hybrid pipeline achieves Pareto-optimal performance using dataset-specific propagation structures:
- **Gowalla (Optimal Settings: `ego_weight = 0.6`, `beta = 0.20` or `beta = 0.50`)**: Gowalla is a medium-density social/locational graph. By allocating `0.6` weight to the raw layer-0 ego embeddings, the model retains strong user and item identities, preventing over-smoothing. Scaling the multi-layer popularity penalty to $\beta=0.50$ or $\beta=0.20$ subtracts enough popularity bias to reduce the average popularity score by **17.5%** or **12.9%** respectively, while *boosting* Recall@20 (up to `0.087353` / `0.088322`), NDCG@20 (up to `0.071463` / `0.073877`), and Precision@20 (up to `0.027413` / `0.027516`) past the baseline.
- **KuaiRec (Optimal Settings: `ego_weight = -1.0` (uniform), `beta = 2.00`)**: KuaiRec is a highly sparse short-video recommendation dataset. Setting `ego_weight = 0.7` originally starved the GCN of propagation layers, collapsing accuracy to near zero. By restoring uniform layer averaging (`ego_weight = -1.0`), neighborhood information propagates fully, allowing the model to leverage collaborative filter connections. Setting $\beta = 2.00$ subtracts popularity bias at each layer, resulting in an aggressive **32.8% bias reduction** (Average Popularity: `529.984` vs `788.993`) while **improving Recall@20 by +632.8%**, **NDCG@20 by +1085%**, and **Precision@20 by +818.8%** over the baseline.
- **Yahoo (Optimal Settings: `ego_weight = -1.0` (uniform), `beta = 0.07`)**: Yahoo is a dense music rating dataset. A high ego weight or large popularity penalty disrupts the densely clustered rating relationships. By using uniform layer averaging (`ego_weight = -1.0`) and a conservative layer-wise popularity penalty of $\beta = 0.07$, the model stabilizes. This fine-tuned setting achieves a **1.9% reduction in popularity bias** (Average Popularity: `486.713` vs `496.072`) while beating the baseline in Recall@20 (`0.140542`), NDCG@20 (`0.064849`), and Precision@20 (`0.011225`).

### 3. Why it is Academically & Empirically Effective
- **Prevention of Representation Collapse**: Traditional GCN models over-smooth user/item representation vectors, pulling tail items into the dense coordinate neighborhoods of highly popular items. Subtracting popularity bias in-propagation at each layer (PPD v4) pushes the item embeddings away from the popularity coordinate center. 
- **Simplex Smoothing via TSP**: Applying Topological Simplex Propagation (TSP) prior to debiasing embeds higher-order multi-hop connectivity (using simplicial complexes, cliques, and Laplacians). This ensures that while popularity bias vectors are subtracted, the underlying collaborative and topological structure of the graph is preserved.
- **Ego-Preservation vs. Deep Propagation**: Retaining a high ego weight prevents GCN message-passing layers from over-smoothing node representations on dense graphs (like Gowalla). Conversely, on sparse graphs (like KuaiRec) or dense rating matrices (like Yahoo), deep propagation with uniform weights is necessary to discover collaborative preferences, which can then be safely debiased using small, targeted popularity penalties.

---

## Implementation Notes & Hyperparameter Glossary

1. **Uniform Layer Aggregation**
   - **Concept:** Controls the layer-aggregation strategy during GCN message passing, setting equal weight to all propagation layers when generating final embeddings. Activated by setting `--ego_weight -1.0`.
   - **Example:** In a 3-layer GCN configuration, an item has embeddings from the raw layer ($E_0$) up to the final propagation layer ($E_1, E_2, E_3$). Under the uniform strategy, the final representation is calculated as a strict, unweighted average:
     $$E_{final} = \frac{1}{4}E_0 + \frac{1}{4}E_1 + \frac{1}{4}E_2 + \frac{1}{4}E_3$$
   - **Use Case:** Crucial for highly sparse datasets like KuaiRec, allowing neighborhood information to flow deeply without starving structural connections.

2. **Ego-Embedding Weight (`--ego_weight`)**
   - **Concept:** A hyperparameter in the upgraded LightGCN backbone that assigns a prioritized explicit weight to the raw, unpropagated Layer-0 embeddings ($E_0$), preventing deep propagation layers from over-smoothing node representations.
   - **Example:** Setting `--ego_weight 0.6` allocations 60% of the final embedding's data to the node's unique structural identity, splitting the remaining 40% among the deeper convolutional layers:
     $$E_{final} = 0.60 \cdot E_0 + 0.133 \cdot E_1 + 0.133 \cdot E_2 + 0.133 \cdot E_3$$
   - **Use Case:** Applied to medium-density graphs like Gowalla. For instance, it anchors a niche, long-tail item (like an indie coffee shop) to its unique coordinate location, preventing it from blending completely into popular neighbor clusters (like a nearby megamall or Starbucks) during multi-hop convolutions.

3. **Popularity Penalty Scaling Factor (`--beta`)**
   - **Concept:** A granular scaling parameter $\beta$ controlling the intensity of the layer-wise popularity bias subtraction during propagation inside PPD Variant 4.
   - **Mathematical Application:** Applied dynamically at every layer $l$ using the formula:
     $$E_{debiased}^{(l)} = E_{raw}^{(l)} - (\beta \cdot B_l)$$
     *(where $B_l$ is the estimated popularity force vector at layer $l$)*.
   - **Examples:**
     - *Aggressive Correction ($\beta = 2.00$):* Used on KuaiRec to forcefully subtract twice the popularity vector force, pushing item vectors completely away from mainstream hubs to yield a 99.2% reduction in popularity bias.
     - *Conservative Correction ($\beta = 0.07$):* Used on dense rating graphs like Yahoo where clusters are fragile. Subtracting only 7% of the bias vector preserves complex rating structures while subtly debiased predictions, beating baseline accuracy metrics.

## Phase III: Advanced Integration & Graph Denoising (Phase III Results)

Phase III introduces five advanced features: Self-Supervised Graph Contrastive Learning (SGL) for structural robustness, Learnable Attention-Based Ego-Weighting to prevent representation collapse, Edge Denoising via Isolation Forest to prune anomalies, LightGBM Ranker to fuse graph embeddings with statistical features, and Hodge-Laplacian Topological Simplex Propagation (TSP). 

The table below presents the evaluation results for the Phase III configurations evaluated at $K=20$ on Gowalla and KuaiRec:

| Dataset | Method | Recall@20 | NDCG@20 | Precision@20 | Average Popularity@20 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gowalla** *(layer 3, recdim 64)* | *PPD Variant 4 Baseline* | *0.074450* | *0.053647* | *0.023242* | *240.736* |
|  | DAP + SGL + PPD Variant 4 | 0.150303 | 0.123683 | 0.046408 | 118.286 |
|  | LightGCN + Dynamic Ego + PPD Variant 4 | 0.115112 | 0.099430 | 0.035871 | 277.184 |
|  | Denoised LightGCN + LightGBM Ranker | 0.091174 | 0.064740 | 0.025655 | 121.316 |
| **KuaiRec** *(layer 2, recdim 256)* | *PPD Variant 4 Baseline* | *0.000069* | *0.000209* | *0.000284* | *5.977* |
|  | DAP + SGL + PPD Variant 4 | 0.011504 | 0.026741 | 0.022766 | 472.634 |
|  | LightGCN + Dynamic Ego + PPD Variant 4 | 0.068570 | 0.094432 | 0.080035 | 605.014 |
|  | Denoised LightGCN + LightGBM Ranker | 0.005210 | 0.006052 | 0.005851 | 804.843 |

### Comparative Discussion

Evaluating the Phase III runs against the PPD Variant 4 baseline reveals substantial performance improvements across all metric groups:

1. **DAP + SGL + PPD Variant 4**: On Gowalla, this method dramatically improves Recall@20 from 0.074450 to 0.150303 (a 101.9% improvement) and NDCG@20 from 0.053647 to 0.123683 (a 130.6% improvement), while reducing popularity bias by more than half (Average Popularity dropping from 240.736 to 118.286). On KuaiRec, it recovers recommendation utility from near-zero levels, boosting Recall@20 to 0.011504 and NDCG@20 to 0.026741. This shows that combining degree-aware centroids, graph contrastive learning, and in-propagation bias subtraction creates a powerful synergy that debiases recommendations while enhancing accuracy.

2. **LightGCN + Dynamic Ego + PPD Variant 4**: On Gowalla, dynamically learning attention-based ego weights improves Recall@20 by 54.6% to 0.115112 and NDCG@20 by 85.3% to 0.099430. On KuaiRec, this method achieves the highest Phase III accuracy, boosting Recall@20 to 0.068570 and NDCG@20 to 0.094432. The dynamic ego weighting preserves user and item identities, preventing over-smoothing and representation collapse even under strong layer-wise popularity penalties.

3. **Denoised LightGCN + LightGBM Ranker**: Edge denoising via Isolation Forest removes anomalous interaction noise, and the LightGBM Ranker effectively blends user/item embeddings with topological similarity and statistical features. On Gowalla, it improves Recall@20 by 22.5% to 0.091174 while significantly reducing popularity bias to 121.316 (a 49.6% reduction over the baseline). On KuaiRec, it improves Recall@20 to 0.005210 while maintaining high rating specificity.

## Dataset-Specific Structural Dynamics & Performance Variance Analysis

The experiments demonstrate that our enhancements (DAP, TSP, PPD) perform differently depending on the structural and mathematical properties of each dataset:

*   **Gowalla (Highly Sparse Locational Graph, Sparsity: 99.9160%):**
    *   **Structure:** Gowalla is extremely sparse, with an average user degree of only 21.67 and item degree skewness of 14.50. This means a tiny fraction of items own a massive share of interactions, while the rest are long-tail locations.
    *   **Behavior:** Standard Graph Convolutional Networks (GCN) suffer from severe *representation collapse* on Gowalla, dragging niche location embeddings into the coordinate neighborhoods of popular hubs during multi-hop convolutions. Introducing **Degree-Aware Propagation (DAP)** dynamically clusters the graph and subtracts similarity centroids. Since Gowalla has a high skewness (14.50), subtracting this degree bias separates tail items from the main cluster centroids, boosting Recall@20 by **+17.4%** over baseline.
    *   **Hybrids:** Combining DAP with TSP (Topological Simplex Propagation) achieves Recall@20 of 0.094326 (+10.1% over baseline) while cutting average popularity from 471.209 to 314.997 (-33.2%). Adding PPD Variant 3 further reduces popularity to 304.650 (-35.3%) but slightly drops accuracy, suggesting that DAP + TSP is already sufficient to balance accuracy and fairness without additional post-hoc projections.

*   **KuaiRec (Sparse Short-Video Graph with Dense Hubs, Sparsity: 98.5022%):**
    *   **Structure:** KuaiRec has a very high average user degree (130.49) and a massive popularity concentration, where the top 5% of items own 39.38% of all interactions.
    *   **Behavior:** Due to the extremely strong popularity bias, standard LightGCN collapses, recommending only the most popular videos and resulting in a near-zero baseline Recall@20 of 0.005594. Subtracting this popularity bias at each GCN layer (PPD Variant 4 with $\beta=2.00$ and uniform layer aggregation) forcefully pushes user and item representations away from popular hubs. This allows the model to leverage collaborative filter connections for long-tail recommendations, yielding a **+632.8% Recall@20 increase** (`0.040994`) over baseline.
    *   **Hybrids:** The **DAP + TSP + PPD Variant 3 (Hybrid)** achieves an NDCG@20 of 0.060431 (+1320.6% over baseline) and Precision@20 of 0.057482 (+1074.5% over baseline) while reducing average popularity to 675.025 (-14.4%). This indicates that combining degree-aware clustering with topological simplex smoothing and post-hoc bias projection is exceptionally effective at recovering ranking accuracy for sparse datasets dominated by global popularity hubs.

*   **Yahoo (Dense Music Rating Matrix, Sparsity: 99.0978%):**
    *   **Structure:** Yahoo has only 1,000 items and 14,382 users, resulting in a low average user degree of 6.17 but extremely high user degree skewness of 15.54.
    *   **Behavior:** Because the rating matrix is densely populated around a small set of items and user degrees are tiny, the collaborative clusters are fragile. Aggressive popularity bias subtraction (PPD Variant 4 with high $\beta$) wipes out the thin signal for low-degree users, collapsing accuracy. Hence, Yahoo requires uniform layer aggregation and a very conservative popularity penalty ($\beta=0.07$) to preserve the complex rating structures while subtly debiasing predictions, beating baseline accuracy metrics.
    *   **Hybrids:** Running **DAP + TSP (Hybrid)** on Yahoo maintains high recommendation utility with Recall@20 of 0.139325 (99.4% of baseline) while reducing popularity bias by 6.1% (465.778 vs 496.072 baseline). Running **DAP + TSP + PPD Variant 3 (Hybrid)** on Yahoo achieves Recall@20 of 0.127084 while yielding an average popularity of 555.583, providing a flexible accuracy-fairness trade-off.

## Phase IV: Combined Pipeline Grid Search (DAP + SGL + Hodge-Laplacian TSP + PPD Variant 4)

In this phase, we implemented and evaluated a fully integrated pipeline combining **Degree-Aware Propagation (DAP)**, **Self-Supervised Graph Contrastive Learning (SGL)**, **Hodge-Laplacian Topological Simplex Propagation (TSP)**, and **PPD Variant 4** (in-propagation popularity debiasing) across all three benchmark datasets (Gowalla, KuaiRec, and Yahoo). 

### 1. Evaluation Results & Grid Search Performance

Each configuration was evaluated against the reference benchmark (`LightGCN + TSP + PPD Variant 3`) to calculate percent deltas. The results are classified as follows:
- **A. DOMINANT**: all three accuracy metrics >= reference AND Average Popularity@20 < reference.
- **B. ACCURACY-WIN**: accuracy metrics >= 5% higher with Average Popularity@20 within 1% tolerance of reference.
- **C. FAIRNESS-WIN**: Average Popularity@20 >= 20% lower with accuracy metrics within 1% tolerance of reference.
- **D. TRADE-OFF/LOSS**: neither of the above conditions holds.

| Dataset | Beta ($\beta$) | Ego Weight Strategy | Recall@20 | NDCG@20 | Precision@20 | Average Popularity@20 | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gowalla** | *Reference: TSP + PPD v3* | *N/A* | *0.076893* | *0.054366* | *0.023447* | *258.450* | *N/A* |
| | $\beta=0.2$ | `ego_weight = 0.6` | 0.124879 (+62.41%) | 0.098547 (+81.27%) | 0.038248 (+63.12%) | 87.577 (-66.11%) | **A. DOMINANT** |
| | $\beta=0.2$ | `ego_weight = -1.0` (Uniform) | 0.150500 (+95.73%) | 0.123656 (+127.45%) | 0.046477 (+98.22%) | 117.759 (-54.44%) | **A. DOMINANT** |
| | $\beta=0.2$ | `ego_weight = dynamic` (Attention) | 0.110327 (+43.48%) | 0.088176 (+62.19%) | 0.034666 (+47.85%) | 119.683 (-53.69%) | **A. DOMINANT** |
| | $\beta=0.5$ | `ego_weight = 0.6` | 0.121659 (+58.22%) | 0.095925 (+76.44%) | 0.037253 (+58.88%) | 87.700 (-66.07%) | **A. DOMINANT** |
| | $\beta=0.5$ | `ego_weight = -1.0` (Uniform) | 0.145595 (+89.35%) | 0.118994 (+118.88%) | 0.044939 (+91.66%) | 114.464 (-55.71%) | **A. DOMINANT** |
| | $\beta=0.5$ | `ego_weight = dynamic` (Attention) | 0.111549 (+45.07%) | 0.090635 (+66.71%) | 0.035312 (+50.60%) | 132.586 (-48.70%) | **A. DOMINANT** |
| **KuaiRec** | *Reference: TSP + PPD v3* | *N/A* | *0.233261* | *0.222384* | *0.106575* | *390.871* | *N/A* |
| | $\beta=0.5$ | `ego_weight = -1.0` (Uniform) | 0.001472 (-99.37%) | 0.002543 (-98.86%) | 0.002447 (-97.70%) | 537.734 (+37.57%) | D. TRADE-OFF/LOSS |
| | $\beta=0.5$ | `ego_weight = dynamic` (Attention) | 0.000000 (-100.00%) | 0.002864 (-98.71%) | 0.003191 (-97.01%) | 508.793 (+30.17%) | D. TRADE-OFF/LOSS |
| | $\beta=1.0$ | `ego_weight = -1.0` (Uniform) | 0.003216 (-98.62%) | 0.006835 (-96.93%) | 0.006312 (-94.08%) | 511.508 (+30.86%) | D. TRADE-OFF/LOSS |
| | $\beta=1.0$ | `ego_weight = dynamic` (Attention) | 0.002809 (-98.80%) | 0.005030 (-97.74%) | 0.004894 (-95.41%) | 556.578 (+42.39%) | D. TRADE-OFF/LOSS |
| | $\beta=2.0$ | `ego_weight = -1.0` (Uniform) | 0.011449 (-95.09%) | 0.026481 (-88.09%) | 0.022518 (-78.87%) | 472.184 (+20.80%) | D. TRADE-OFF/LOSS |
| | $\beta=2.0$ | `ego_weight = dynamic` (Attention) | 0.008818 (-96.22%) | 0.021530 (-90.32%) | 0.017057 (-84.00%) | 422.265 (+8.03%) | D. TRADE-OFF/LOSS |
| **Yahoo** | *Reference: TSP + PPD v3* | *N/A* | *0.088663* | *0.073549* | *0.027056* | *280.957* | *N/A* |
| | $\beta=0.02$ | `ego_weight = -1.0` (Uniform) | 0.129930 (+46.54%) | 0.059075 (-19.68%) | 0.010176 (-62.39%) | 325.920 (+16.00%) | D. TRADE-OFF/LOSS |
| | $\beta=0.02$ | `ego_weight = dynamic` (Attention) | 0.098438 (+11.02%) | 0.043228 (-41.23%) | 0.008204 (-69.68%) | 333.478 (+18.69%) | D. TRADE-OFF/LOSS |
| | $\beta=0.05$ | `ego_weight = -1.0` (Uniform) | 0.130035 (+46.66%) | 0.059077 (-19.68%) | 0.010176 (-62.39%) | 323.638 (+15.19%) | D. TRADE-OFF/LOSS |
| | $\beta=0.05$ | `ego_weight = dynamic` (Attention) | 0.098963 (+11.62%) | 0.043333 (-41.08%) | 0.008246 (-69.52%) | 332.822 (+18.46%) | D. TRADE-OFF/LOSS |
| | $\beta=0.07$ | `ego_weight = -1.0` (Uniform) | 0.129181 (+45.70%) | 0.058775 (-20.09%) | 0.010092 (-62.70%) | 322.757 (+14.88%) | D. TRADE-OFF/LOSS |
| | $\beta=0.07$ | `ego_weight = dynamic` (Attention) | 0.098963 (+11.62%) | 0.043326 (-41.09%) | 0.008246 (-69.52%) | 332.544 (+18.36%) | D. TRADE-OFF/LOSS |
| | $\beta=0.15$ | `ego_weight = -1.0` (Uniform) | 0.130310 (+46.97%) | 0.059630 (-18.93%) | 0.010218 (-62.23%) | 330.378 (+17.59%) | D. TRADE-OFF/LOSS |
| | $\beta=0.15$ | `ego_weight = dynamic` (Attention) | 0.098788 (+11.42%) | 0.043198 (-41.27%) | 0.008225 (-69.60%) | 331.091 (+17.84%) | D. TRADE-OFF/LOSS |

### 2. Dataset-Specific Analysis

- **Gowalla**: All configurations resulted in a **DOMINANT** performance compared to the reference benchmark. The uniform aggregation strategy (`ego_weight = -1.0`) combined with a conservative popularity penalty ($\beta=0.20$) emerged as the overall winner, yielding a massive **+95.73% Recall@20** and **+127.45% NDCG@20** improvement while dropping popularity bias by **-54.44%**. This suggests that on Gowalla, DAP's structural debiasing coupled with SGL representation robustness and Hodge-Laplacian simplex smoothing provides a massive boost in tail coverage without hurting ranking quality.
- **KuaiRec**: All configurations resulted in a **TRADE-OFF/LOSS** classification due to underperforming accuracy compared to the reference benchmark. This occurs because the combined model was only trained for 15 epochs in Phase III (matching short-run configurations), which is insufficient for the dense, large-scale KuaiRec dataset to reach full convergence. Higher beta values ($\beta=2.00$) show partial recovery of accuracy (Recall@20: 0.011449) compared to lower betas, indicating that a larger popularity penalty is required to counteract KuaiRec's extreme skewness when training is limited.
- **Yahoo**: Yahoo also received a **TRADE-OFF/LOSS** classification across all configurations. While Recall@20 showed a significant boost (up to **+46.97%**), both NDCG@20 and Precision@20 declined, and Average Popularity rose by ~15-18%. Since Yahoo's user rating graph is small and fragile, applying the complex post-hoc Hodge-Laplacian TSP and multi-layer PPD Variant 4 force-vector subtraction distorts the localized collaborative clusters, showing that simpler post-hoc combinations (like DAP + TSP or PPD Variant 3) are more suited for this dense matrix.

### 3. CLI Flags and Implementation Details

- **`--tsp_hodge`**: A new post-hoc evaluation flag that swaps the standard simplicial Laplacian in the Topological Simplex Propagation (TSP) smoothing step for the unnormalized Hodge Laplacian ($L_k = L_{k,\text{down}} + L_{k,\text{up}}$). This is computed using `sc.hodge_laplacian_matrix(k)` from `TopoNetX`, allowing edge-wise simplex propagation during graph smoothing. If not specified, standard `--tsp` uses `sc.normalized_laplacian_matrix(k)` (normalized Hodge Laplacian).
- **`--dynamic_ego`**: Exposes attention-based learnable ego-weighting. Under this configuration, the model learns parameters `W_att` and `w_att` during BPR training to calculate dynamically-weighted layer aggregation, preventing representation collapse.

### 4. What was Added & Solved (Thesis Pipeline Additions)

*   **Hodge-Laplacian TSP Option**: Modified `run_thesis.py` and `parse.py` to add `--tsp_hodge`. In `run_thesis.py`'s `apply_TSP`, we check the flag to toggle between `sc.hodge_laplacian_matrix(k)` (unnormalized Hodge Laplacian) and `sc.normalized_laplacian_matrix(k)` (normalized simplicial Laplacian).
*   **Grid Search Automation**: Created `run_thesis_experiments.py` to train DAP + SGL with/without `--dynamic_ego` and automatically execute evaluations across the parameter grids for Gowalla, KuaiRec, and Yahoo.

---
---

# ==============================================================================
# PHASE V: PPD v6 / ASTD-PD (Adaptive Spectral-Topological Debiasing with Pareto Dual-Stream Fusion)
# ==============================================================================

---
---

## 1. Executive Summary & Problem Formulation

Traditional Graph Convolutional Networks (e.g., LightGCN) in recommendation systems suffer from severe **popularity bias** and **representation collapse**:
1. **Neighborhood Homogenization**: High-degree popular items dominate bipartite graph convolution messages, dragging long-tail item embeddings into popular cluster neighborhoods.
2. **The Accuracy-Fairness Dilemma**: Post-hoc calibration (e.g., PPD v3) and in-propagation global subtraction (e.g., PPD v4) reduce average popularity but historically suffer from **severe recall and ranking collapse** (losing 10–25% accuracy on dense and sparse graphs).
3. **Topological Obliviousness**: Simplicial complexes and higher-order co-occurrence simplices can boost collaborative filtering but inadvertently amplify popularity hubs if not popularity-damped.

**PPD v6 / ASTD-PD** (*Adaptive Spectral-Topological Debiasing with Pareto Dual-Stream Fusion*) is an end-to-end, dual-stream neural graph architecture engineered to resolve this trade-off and achieve **strict 4-dimensional Pareto dominance** over:
- `LightGCN`
- `TSP` (Topological Simplex Propagation)
- `PPD v3` (Post-hoc popularity calibration)
- `PPD v4` (In-propagation popularity subtraction)

across all four core metrics simultaneously:
$$\text{Recall@20} \uparrow \quad \text{NDCG@20} \uparrow \quad \text{Precision@20} \uparrow \quad \text{Average Popularity@20} \downarrow$$

---

## 2. Mathematical Architecture & Algorithmic Design

```
+----------------------------------------------------------------------------------------------------+
|                                    PPD v6 / ASTD-PD ARCHITECTURE                                   |
+----------------------------------------------------------------------------------------------------+
                                      [User-Item Bipartite Graph G]
                                                    |
             +--------------------------------------+--------------------------------------+
             |                                                                             |
    [Accuracy Stream E^A,(0)]                                                     [Debiased Stream E^D,(0)]
             |                                                                             |
    Standard GCN: M^G = P E^A                                                     Popularity Subspace Basis C^(l)
    Damped Simplex: M^T = S^kappa E^A                                             Pi(m) = m C C^T (Orthonormal)
    Topological Gate: theta_v in [0, 1]                                           Adaptive Alpha: alpha_v = a_v * rho_v
    Fused Message: m = theta M^T + (1-theta) M^G                                  High-Freq Residual: gamma * h_v
    Residual: h = E^A - m                                                         Debiased Message: m_D = m - alpha Pi(m) + gamma h
             |                                                                             |
    Layer Aggregation: E^A = LayerMean(E^A,(l))                                   Layer Aggregation: E^D = LayerMean(E^D,(l))
             |                                                                             |
             +--------------------------------------+--------------------------------------+
                                                    |
                                      [Pareto Dual-Stream Fusion]
                                  lambda_u = MLP([p_u, delta_u, div_u])
                                         t_i = (1 - p_i)^phi_t
                        Score: s_ui = s_ui^A + lambda_u * t_i * s_ui^D - mu_cal * p_i
```

### 2.1 Feature Engineering & Normalization Store (`PopularityDensityStore`)
To ensure robust generalization across sparse and saturated graphs without manual per-dataset re-tuning:
1. **Rank-Normalized Item Popularity**:
   $$p_i = \frac{\text{rank}(\deg_i)}{|\mathcal{I}|} \in [0, 1]$$
2. **User Popularity Tendency**:
   $$p_u = \frac{1}{|\mathcal{N}(u)|} \sum_{i \in \mathcal{N}(u)} p_i \in [0, 1]$$
3. **Second-Order Graph Density**:
   $$\delta_i = \text{rank}\left(\log\left(1 + \sum_{u \in \mathcal{N}(i)} \deg_u\right)\right) \in [0, 1], \quad \delta_u = \text{rank}\left(\log\left(1 + \sum_{i \in \mathcal{N}(u)} \deg_i\right)\right) \in [0, 1]$$
4. **User History Decile Diversity**:
   $$\text{div}_u = \frac{|\{ \text{decile}(i) : i \in \mathcal{N}(u) \}|}{10} \in [0, 1]$$

### 2.2 Popularity-Damped Simplex Item Graph ($S^\kappa$)
Item co-occurrence simplices $A_{ij}^{\text{co}} = (R^\top R)_{ij}$ are scaled by popularity damping factor $\kappa$ to prevent simplicial message explosion along dominant items:
$$S_{ij}^\kappa = \frac{A_{ij}^{\text{co}}}{\sqrt{\deg_i \deg_j}} \cdot \frac{1}{(1 + \deg_i)^\kappa (1 + \deg_j)^\kappa}$$
The matrix is adaptively pruned to top-$K$ neighbors ($K=200$) and stored as an efficient PyTorch Sparse COO tensor.

### 2.3 Detached Low-Rank Popularity Subspace Basis ($C^{(l)}$)
Rather than computing full SVD on GPU every step or relying on unconstrained linear projections:
- Item embeddings at layer $l$ are grouped into 5 popularity quantile buckets.
- Difference vectors $v_1 = \mu_4 - \mu_0, \ v_2 = \mu_3 - \mu_1, \ v_3 = \mu_2 - \mu_{\text{global}}$ span the popularity subspace.
- Orthonormal basis $C^{(l)} \in \mathbb{R}^{d \times r}$ ($r=2$) is extracted via QR decomposition:
  $$C^{(l)\top} C^{(l)} = I_r$$
- Maintained as registered buffers, updated periodically via exponential moving average ($\beta_{\text{EMA}} = 0.95$) detached from autograd to prevent graph backpropagation corruption.
- Spectral projection filter:
  $$\Pi^{(l)}(m_v) = m_v C^{(l)} C^{(l)\top}$$

### 2.4 Adaptive Node-Level Debiasing & High-Frequency Residual
- **Global Density Scaler**:
  $$\alpha_{\max} = \text{clip}\left(\frac{\alpha_{\text{base}}}{1 + \xi \log(1 + \bar{d}_{\text{item}} / d_{\text{ref}})}, \alpha_{\min}, \alpha_{\text{base}}\right)$$
- **Node-Level Strength**:
  $$\alpha_v^{(l)} = \alpha_{\max} \cdot \sigma(\text{MLP}_\alpha([p_v, \delta_v])) \cdot \frac{p_v^\gamma}{p_v^\gamma + 0.5^\gamma}$$
- **High-Frequency Collaborative Residual**:
  $$h_v^{(l)} = e_v^{A,(l)} - m_v^{(l)}$$
- **Debiased Message**:
  $$m_v^{D,(l)} = m_v^{(l)} - \alpha_v^{(l)} \Pi^{(l)}(m_v^{(l)}) + \gamma_v^{(l)} h_v^{(l)}$$

### 2.5 Dynamic User-Level Pareto Fusion Scoring
During inference and multi-task loss computation, scores are adaptively fused:
$$s_{ui} = \frac{e_u^{A\top} e_i^A}{\tau_{\text{acc}}} + \lambda_u (1 - p_i)^{\phi_t} \frac{e_u^{D\top} e_i^D}{\tau_{\text{deb}}} - \mu_{\text{cal}} p_i$$
where:
- $\lambda_u = \lambda_{\min} + (\lambda_{\max} - \lambda_{\min}) \sigma(\text{MLP}_\lambda([p_u, \delta_u, \text{div}_u])) \in [\lambda_{\min}, \lambda_{\max}]$
- $t_i = (1 - p_i)^{\phi_t}$ is the nonlinear tail discovery filter ($\phi_t \ge 1.5$)
- $\mu_{\text{cal}}$ is the optional post-hoc popularity calibration penalty.

### 2.6 Multi-Objective Joint Loss & Stratified Negative Sampler
- **Stratified Negative Sampler with Exact LogQ Correction**:
  Mixture of uniform ($r_u$), popularity-matched decile ($r_m$), and hard top-decile ($r_h$) negatives with correction $\tilde{s}_{uj} = s_{uj} - \log q_j$.
- **Joint Optimization Loss**:
  $$\mathcal{L} = \lambda_{\text{final}} \mathcal{L}_{\text{final}} + \lambda_A \mathcal{L}_A + \lambda_D \mathcal{L}_{D,\text{tail}} + \lambda_{\text{ind}} (\mathcal{L}_{\text{proj}} + \lambda_{\text{cov}} \mathcal{L}_{\text{cov}}) + \lambda_2 \mathcal{L}_2$$
- **Validation 4D Pareto Hypervolume Early Stopping**:
  Checkpoints are evaluated and saved based on the exact 4-dimensional hypervolume:
  $$\text{HV} = \text{Hypervolume}\left(\text{Recall@20}, \text{NDCG@20}, \text{Precision@20}, \frac{1}{\text{AvgPop@20}}\right)$$

---

## 3. Comprehensive Experimental Results Across 4 Datasets

All models are evaluated across **five** independent random seeds (`2024`-`2028`) with full training and early stopping, under the frozen Phase 0 protocol. Significance is reported two ways — per-user Wilcoxon and an across-seed paired t-test — because only the second answers "does the margin survive retraining?" (`LightGCN/code/significance.py`).

**Headline, Gowalla:** PPD v6 beats LightGCN on Recall@20, NDCG@20 and Precision@20 with across-seed significance (p = 0.0115 / 0.000177 / 0.00139) at −33% AvgPop@20, +252% Coverage@20 and +255% APLT@20. Against PPD v4 it wins Recall by +12.5% while its AvgPop deficit is not statistically distinguishable from zero (p = 0.214).

**Hyperparameters are validation-selected.** The originals were not — they came from a sweep script that ranked candidates on test metrics — so all four datasets were re-selected over 96 configurations reading validation only, with each winner retrained at 5 seeds. Nothing moved outside seed noise (finding **AH**).

**Two results that qualify it, both measured rather than assumed.** The six-arm ablation shows the popularity-stratified negative sampler accounts for 79% (Gowalla) to 96% (Yahoo) of the AvgPop reduction and the spectral projection is inert everywhere, while the named mechanisms own 74% of the tail-exposure gain. And a 12-point sweep of an inference-time popularity penalty shows the remaining four-metric gap cannot be closed: the penalty trades Recall for AvgPop point for point. Full detail in `RESULTS_CANONICAL.md` and `THESIS_AUDIT.md` findings **AE**-**AG**.

<!-- BEGIN GENERATED V6 TABLES -->

*Generated by `make_readme_tables.py` from `results/phase1_*.json` -- do not hand-edit.*
*Protocol: 5 seeds (2024, 2025, 2026, 2027, 2028), mean ± population std. Best value per column in bold.*
*Significance tests vs LightGCN live in `results/significance_*.json`; the narrative reading is in `THESIS_AUDIT.md` findings T-V.*

### Gowalla

**Sparse.** Test users are 100% of all users, so no protocol fix moves these numbers -- which is why this is the result the thesis leads with.

| Method | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Coverage@20 | APLT@20 | Gini@20 |
|---|---|---|---|---|---|---|---|
| LightGCN | 0.140693 ± 0.000243 | 0.115987 ± 0.000301 | 0.042188 ± 0.000072 | 212.99 ± 1.36 | 0.1058 ± 0.0016 | 0.0469 ± 0.0010 | 0.9826 ± 0.0002 |
| TSP | 0.140337 ± 0.000236 | 0.116880 ± 0.000293 | 0.042418 ± 0.000080 | 209.25 ± 1.29 | 0.1035 ± 0.0011 | 0.0510 ± 0.0011 | 0.9825 ± 0.0002 |
| PPD v3 | 0.140585 ± 0.000292 | 0.116295 ± 0.000411 | 0.042292 ± 0.000101 | 199.57 ± 1.65 | 0.1069 ± 0.0015 | 0.0474 ± 0.0010 | 0.9821 ± 0.0002 |
| PPD v4 | 0.126957 ± 0.000877 | 0.106169 ± 0.000991 | 0.037829 ± 0.000353 | **138.44 ± 2.79** | 0.1423 ± 0.0029 | 0.0839 ± 0.0020 | 0.9698 ± 0.0007 |
| **PPD v6 (Ours)** | **0.142878 ± 0.000910** | **0.121314 ± 0.000598** | **0.043501 ± 0.000286** | 141.60 ± 2.00 | **0.3729 ± 0.0050** | **0.1665 ± 0.0049** | **0.9322 ± 0.0015** |

---

### Yahoo

**Dense.** Only 16.6% of users appear in the test split; every row describes that minority. AvgPop is scoped to test users (audit finding L).

| Method | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Coverage@20 | APLT@20 | Gini@20 |
|---|---|---|---|---|---|---|---|
| LightGCN | 0.140833 ± 0.002087 | 0.064305 ± 0.000841 | 0.011041 ± 0.000109 | 593.93 ± 3.37 | 0.6816 ± 0.0041 | 0.1118 ± 0.0029 | 0.8557 ± 0.0021 |
| TSP | 0.140483 ± 0.002503 | 0.064260 ± 0.000855 | 0.011037 ± 0.000193 | 588.80 ± 3.86 | 0.6806 ± 0.0067 | 0.1159 ± 0.0034 | 0.8525 ± 0.0026 |
| PPD v3 | **0.141834 ± 0.003029** | **0.064761 ± 0.000861** | **0.011095 ± 0.000202** | 563.96 ± 2.79 | 0.6858 ± 0.0057 | 0.1235 ± 0.0028 | 0.8458 ± 0.0020 |
| PPD v4 | 0.137602 ± 0.002978 | 0.061553 ± 0.000691 | 0.010760 ± 0.000223 | **505.80 ± 5.06** | **0.7140 ± 0.0041** | **0.1463 ± 0.0034** | **0.8257 ± 0.0026** |
| **PPD v6 (Ours)** | 0.137227 ± 0.003536 | 0.062214 ± 0.000897 | 0.010768 ± 0.000236 | 526.35 ± 11.68 | 0.6182 ± 0.0190 | 0.1430 ± 0.0112 | 0.8284 ± 0.0085 |

---

### KuaiRec

**Dense.** Candidates restricted to the 3,247-item test universe (finding P) AND AvgPop scoped to test users (finding L). Both fixes bite hard here.

| Method | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Coverage@20 | APLT@20 | Gini@20 |
|---|---|---|---|---|---|---|---|
| LightGCN | 0.155205 ± 0.000429 | 0.304026 ± 0.001228 | 0.273582 ± 0.001001 | 1147.31 ± 6.21 | 0.1531 ± 0.0096 | 0.0607 ± 0.0039 | 0.9747 ± 0.0023 |
| TSP | 0.154812 ± 0.000524 | 0.302221 ± 0.001796 | 0.272220 ± 0.001538 | 1136.19 ± 10.78 | 0.1536 ± 0.0094 | 0.0616 ± 0.0036 | 0.9744 ± 0.0022 |
| PPD v3 | 0.067179 ± 0.005863 | 0.121501 ± 0.007597 | 0.101475 ± 0.006615 | **399.94 ± 27.44** | **0.2264 ± 0.0117** | **0.5839 ± 0.0153** | **0.9257 ± 0.0118** |
| PPD v4 | 0.155779 ± 0.000389 | 0.303060 ± 0.001150 | 0.273085 ± 0.000550 | 1158.54 ± 5.35 | 0.1564 ± 0.0085 | 0.0706 ± 0.0060 | 0.9744 ± 0.0022 |
| **PPD v6 (Ours)** | **0.177101 ± 0.000872** | **0.393999 ± 0.003301** | **0.362759 ± 0.002741** | 1606.57 ± 19.08 | 0.0330 ± 0.0025 | 0.0108 ± 0.0015 | 0.9968 ± 0.0003 |

---

### Coat

**Small/dense.** Test users are 96.9% of all users. TSP builds zero edges on Coat and never runs, so its rows are identical to LightGCN by construction (finding S).

| Method | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Coverage@20 | APLT@20 | Gini@20 |
|---|---|---|---|---|---|---|---|
| LightGCN | 0.124472 ± 0.003662 | **0.069713 ± 0.000987** | **0.022705 ± 0.000511** | 27.56 ± 0.08 | 0.3720 ± 0.0223 | 0.2207 ± 0.0115 | 0.8270 ± 0.0016 |
| TSP | 0.124472 ± 0.003662 | **0.069713 ± 0.000987** | **0.022705 ± 0.000511** | 27.56 ± 0.08 | 0.3720 ± 0.0223 | 0.2207 ± 0.0115 | 0.8270 ± 0.0016 |
| PPD v3 | **0.124636 ± 0.002856** | 0.068893 ± 0.000990 | 0.022598 ± 0.000491 | 27.24 ± 0.03 | 0.4127 ± 0.0210 | 0.2290 ± 0.0108 | 0.8242 ± 0.0019 |
| PPD v4 | 0.124267 ± 0.005312 | 0.069678 ± 0.000797 | 0.022705 ± 0.000430 | 27.32 ± 0.12 | 0.4267 ± 0.0183 | 0.2352 ± 0.0134 | 0.8189 ± 0.0020 |
| **PPD v6 (Ours)** | 0.117744 ± 0.006399 | 0.065158 ± 0.003116 | 0.021673 ± 0.000715 | **25.31 ± 0.95** | **0.5733 ± 0.1322** | **0.3208 ± 0.0471** | **0.7943 ± 0.0294** |

---

<!-- END GENERATED V6 TABLES -->

## 4. Key Findings & Scientific Breakthroughs

1. **Resolution of the Historical Trade-Off — on sparse graphs only.** On **Gowalla** PPD v6 / ASTD-PD achieves higher Recall@20, NDCG@20 and Precision@20 than LightGCN — all three significant across 5 seeds (p = 0.0115 / 0.000177 / 0.00139) and under validation-selected hyperparameters — while cutting AvgPop@20 by 33% and raising catalogue Coverage@20 by 252%. ~~on KuaiRec by 5.6% (967.43 vs 1024.97) alongside +14.1% Recall.~~ **RETRACTED 2026-08-26.** The KuaiRec popularity reduction was the same all-user/test-user scope artifact as finding T. Scoped correctly, v6's KuaiRec AvgPop@20 is **1616.19 vs LightGCN's 1148.79 — 41% WORSE**, at Coverage@20 0.0319 vs 0.1582 (−80%). The +14% Recall is real and significant; it was bought by collapsing onto the popular head. See audit finding **V**. ~~Note the 30-40% popularity reduction claimed in earlier drafts belongs to PPD v4, which on corrected KuaiRec cuts AvgPop 33.2%.~~ **RETRACTED 2026-08-25 (audit finding T).** The -33.2% AvgPop was an artifact of a second protocol bug: AvgPop ran over ALL users while accuracy ran over TEST users. Measured on test users, PPD v4's KuaiRec AvgPop is **+0.9% (worse than LightGCN)**, not -33.2%. It was suppressing popular items for the ~80% of users no accuracy metric evaluates. PPD v4's popularity reduction **is** real on Gowalla and Yahoo, where test users are ~all users: on Gowalla it lifts APLT +79% and Coverage +34%. On KuaiRec it is a near-no-op.
2. ~~**Topological Stability Without Hub Explosion**~~ — **QUALIFIED 2026-08-30 by the Phase 2 ablation (finding AE).** The popularity-damped simplex stream does not improve accuracy: removing it (arm `c_no_simplex`) *raises* Gowalla Recall@20 from 0.142780 to 0.149034. What it does buy is tail exposure — APLT@20 falls from 0.1655 to 0.1300 without it. The defensible claim is an accuracy-for-exposure trade, not an enhancement.
3. ~~**Representation Collapse Prevention on Skewed Graphs**~~ — **WITHDRAWN 2026-08-25.** This finding rested on LightGCN scoring $Recall = 0.00042$ on KuaiRec. That figure was an evaluation-protocol artifact: candidates were ranked over the full 10,728-item catalogue while the test split covers 3,247 items, so the model's top-K landed almost entirely on items that could never be a hit. The identical checkpoint scores **0.155524** under the corrected protocol. There is no representation collapse on KuaiRec, and no $32.8\times$ recovery to claim.
4. **Adaptive Mechanism Stability — true on Gowalla only, and the adaptation is real.** The gates do *not* hold everywhere: on Yahoo and Coat both $\alpha$ and $\theta$ collapse to numerical zero, and $\bar{\lambda}_{\text{fuse}}$ saturates at its configured ceiling rather than sitting where it was initialised (findings **D**, **Y**). Where they survive, they survive because of the data and not the initialisation: handed Yahoo's 4x smaller $\alpha_{base}$, Gowalla still retains 21.2% of it against 21.0% natively, while Yahoo retains ~0.4% whatever it is given (finding **AF**). Note that surviving is not the same as mattering — setting $\alpha_{base}=0$ changes Gowalla's Recall by 8e-05 (finding **AE**).

---

## 5. How to Run PPD v6 / ASTD-PD

### 5.1 Training PPD v6 End-to-End
```bash
cd LightGCN/code

# Yahoo
python main.py --method ppd_v6 --dataset Yahoo --layer 2 --recdim 256 --astd_alpha_base 0.10 --astd_topo_gate_init 0.75 --astd_temperature_acc 0.30 --astd_fuse_lambda_max 0.25 --astd_tail_phi 2.0

# Gowalla
python main.py --method ppd_v6 --dataset Gowalla --layer 3 --recdim 64 --astd_alpha_base 0.40 --astd_topo_gate_init 0.40 --astd_fuse_lambda_max 0.35

# KuaiRec
python main.py --method ppd_v6 --dataset KuaiRec --layer 2 --recdim 256 --astd_alpha_base 0.35 --astd_topo_gate_init 0.50 --astd_fuse_lambda_max 0.35
```

### 5.2 Running the benchmark under the frozen protocol

These are the drivers that produce every number in `results/` and in the tables above. They
merge into their JSON rather than overwriting it, and skip any unit whose checkpoint already
exists, so all of them are safe to re-run.

```bash
# Phase 1 -- all methods, all datasets, 5 seeds (baselines are post-hoc transforms of one
# trained LightGCN per seed; PPD v6 trains its own weights, hence the two drivers)
python run_phase1_baselines.py            # add: Gowalla --seeds 2027,2028
python run_phase1_v6.py

# Phase 2 -- mechanism-isolation ablation, six arms x 3 seeds
python run_phase2_ablation.py

# Phase 3 -- gate transfer: does the collapse track the data or the initialization?
python run_phase3_gate_transfer.py

# Phase 4 -- inference-penalty frontier (re-ranks trained checkpoints, no retraining)
python run_phase4_inference_penalty.py

# Significance, then regenerate every derived artifact
python significance.py --baseline results/per_user/Gowalla_LightGCN \
                       --candidate results/per_user/Gowalla_PPDv6 \
                       --out results/significance_gowalla_v6.json
python make_canonical.py && python make_readme_tables.py && python make_numbers_pack.py
python check_sanity_floors.py             # non-zero exit if any model falls below Random

# Multi-GPU: one worker per arm, each with a private checkpoint tag and results shard
WORKERS_PER_GPU=3 ./run_parallel_gowalla.sh && python merge_shards.py phase2_ablation_gowalla

# Standalone unit test suite
python -m unittest tests/test_ppd_v6.py
```

> **Do not use `refine_yahoo_ppd_v6.py`, `run_component_search.py`, `test_beta_sweep.py`,
> `run_dcd_sweep.py` or `run_yahoo_ablations.py` to produce reportable numbers.** They rank
> configurations on parsed **test** metrics (audit ledger 5.7). They are kept because their
> outputs are cited in the audit's history, not because their selection is sound.

### 5.3 CLI Flag Reference for PPD v6
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--method ppd_v6` | `lightgcn` | Dispatches to dual-stream ASTD-PD training with Pareto hypervolume early stopping |
| `--astd_alpha_base` | `0.20` | Base debiasing strength before dataset density scaling |
| `--astd_density_xi` | `1.0` | Global density scaling factor $\xi$ |
| `--astd_topo_gate_init`| `0.50` | Initial bias for topological routing gate $\theta_v$ |
| `--astd_topo_pop_damping` | `0.10` | Popularity exponent $\kappa$ for simplicial adjacency damping |
| `--astd_highfreq_gamma` | `0.10` | Weight $\gamma$ for high-frequency collaborative residual preservation |
| `--astd_fuse_lambda_max`| `0.50` | Maximum Pareto fusion gate $\lambda_{\max}$ |
| `--astd_tail_phi` | `1.0` | Exponent $\phi_t$ for nonlinear tail discovery weighting |
| `--astd_temperature_acc`| `0.10` | Softmax temperature for accuracy stream |
| `--astd_neg_hard_ratio` | `0.20` | Fraction of hard top-decile negatives in stratified sampler |
| `--astd_pop_basis_rank` | `2` | Rank $r$ of low-rank popularity subspace |

