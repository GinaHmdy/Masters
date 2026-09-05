# MSc Thesis — Contribution Audit

**Scope.** Independent audit of the contribution's integrity and validity, measured against
the work's own stated goal. Every claim below cites a file:line or a re-runnable command.

## Remediation ledger

Findings are struck from §3 only when closed by a commit or a re-run. Status as of 2026-08-30:

| # | Item | Status | Closed by |
|---|---|---|---|
| 5.1 | Commit the v6 code | **CLOSED** | `3e3d60c` — code, tests, `results/*.json` and `.results.txt` now tracked. The ~773 MB of `.pth.tar` binaries were deliberately excluded; if they are needed for reproduction they should go to a release asset or DVC, not to git history. |
| 5.2 | Correct the attribution | **CLOSED** | `README.md` § Prior Art and Attribution; `MSC_THESIS_CONTRIBUTION_GUIDE.md` attribution notice + Phase-arc labels; `docs/THE_STORY.md` Part 3. **Upgraded from the audit's original finding:** PPD is not merely inherited, it is a *published method* — Islam, Wang & Zheleva, arXiv:2510.12959 (Oct 2025). DAP is Chen et al. 2024 (a baseline in that same paper) and DCD is DICE (WWW 2021). This strengthens the thesis rather than weakening it: the baselines are external, not self-ablations. |
| — | Framing: popularity is not always bias | **NEW, ADOPTED** | `MSC_THESIS_CONTRIBUTION_GUIDE.md` §2.0, `docs/THE_STORY.md` §2.5. See finding **O** below. |
| 5.3 | MostPopular + Random in the harness | **CLOSED** | `trivial_baselines.py` runs both reference rankers on every dataset under the same protocol, candidate mask included, and `check_sanity_floors.py` turns them into an executable gate (non-zero exit if any model falls below Random). Result: all models clear Random everywhere, but on KuaiRec **every method except PPD v6 loses to MostPopular on NDCG@20** — see finding **Z**. Random wins AvgPop@20 on all four datasets, which is why AvgPop alone cannot carry the fairness claim (see 5.9). Both reference rankers now appear as rows in the numbers-pack tables rather than in a footnote. |
| 5.4 | Re-run Gowalla targeting the AvgPop gap | **CLOSED — negative result** | Phase 4. `run_phase4_inference_penalty.py` sweeps the inference-time popularity penalty over the five trained checkpoints (no retraining: the penalty is applied at ranking time, `model.py:1035`), 12 values x 5 seeds, selected on validation. The thesis's own hypervolume criterion selects **mu = 0**, and the mu = 1.6-1.75 window whose point estimates do clear all four baselines is a **statistical tie on both binding margins** (Recall vs LightGCN p = 0.68; AvgPop vs PPD v4 p = 0.96). The four-metric goal is not attainable on Gowalla. See finding **AG** — which also shows the *better* claim it licenses: v6's AvgPop deficit vs PPD v4 is itself not significant (p = 0.214) while it beats PPD v4 on Recall by +12.5%. |
| 5.5 | Mechanism-isolation ablation | **CLOSED** | Phase 2, six arms x 3 seeds on **all four datasets**. `run_phase2_ablation.py`. The verdict is uniform: the popularity-stratified negative sampler accounts for 79% (Gowalla) to 96% (Yahoo) of the AvgPop reduction, the tail reweighting and simplex stream own the tail-exposure gain (74% of APLT on Gowalla), and the **spectral projection is inert on every dataset including Gowalla**, the one where its parameter survives training (finding **AE**). `run_phase3_gate_transfer.py` resolved finding **Y**'s adaptation-vs-initialization confound: retention tracks the dataset, not the starting value (finding **AF**). |
| 5.6 | Fix/remove dead flags | **CLOSED** | Phase 0. `beta` removed from `project_and_remove` (it was accepted and never referenced, so every PPD v3 `--beta` sweep since the reflection fix measured nothing). `--ppd_beta_schedule`, `--ppd_seq_beta_v3` and `--astd_topo_gate_init` now print an explicit `[WARNING] ... has NO EFFECT` at startup rather than silently no-opping. |
| 5.7 | Re-select hyperparameters on validation | **CLOSED** | Phase 5. `run_phase5_hpo.py` re-selects over **96 configurations across all four datasets, reading the `[Best Valid Results]` block only**; `run_phase5_final.py` retrains each winner at 5 seeds. The leak was real — the shipped config was never the validation winner on any dataset (Yahoo shipped 0.10/0.10 while its own sweep's validation winner was 0.15/0.05; `refine_yahoo_ppd_v6.py`'s test-metric `score_fn` chose the former) — **and it changed nothing**: every re-selected configuration lands within 0.4% of the shipped one on test, no conclusion flips, and Gowalla's headline margin slightly strengthens (Recall p = 0.0115 → **0.00582**). The sweep also reproduces finding **AE** independently: validation HV sorts almost entirely by `astd_tail_phi` and is flat along `alpha_base`. See finding **AH**. |
| 5.8 | Fix or drop collapsed baselines | **CLOSED** | The KuaiRec baselines were never collapsed — the *evaluation* was. Candidates were ranked over all 10,728 items while KuaiRec's test split (the dense "small matrix") covers 3,247; >99% of every top-K was unhittable by construction. Fixed by `restrict_candidates` (`dataloader.py` `candidateItems`, `Procedure.py` `_apply_candidate_mask`). Same checkpoint: Recall@20 0.000424 → **0.155524**. Baselines and the 27-cell KuaiRec component search re-run; see finding **P**. |
| — | Eval protocol: KuaiRec candidate set | **NEW, CLOSED** | See finding **P** below. |
| 5.9 | Gini / coverage / APLT; AvgPop over test users | **CLOSED** | Phase 0. `utils.py` gained `gini_at_k`, `coverage_at_k`, `APLT_ATk`, `TailRecallATk` + `long_tail_mask`, all computed from the top-K matrix `Test()` already builds. AvgPop@20 now runs over `testDict` users (finding L). |
| 5.10 | Evaluate on unbiased splits | **CLOSED (as stated) / RESCOPED** | Phase 4. The default `Coat/` and `KuaiRec/` directories are byte-identical to their unbiased variants across train, valid **and** test — so every result already reported is a randomized-exposure evaluation. The audit's own narrowing ("only validation differs") was also wrong: `KuaiRec/biased-valid/` differs in all three files, with 5x the validation users. What remains is the narrower selection-protocol question, now folded into Phase 3. See finding **W**. |
| 5.11 | 5 seeds + paired significance test | **CLOSED** | Phase 1. **All four datasets now report every method at n = 5** with mean +/- population std, plus per-user Wilcoxon and across-seed paired t-tests (`significance.py`). The extension changed the headline verdict: Gowalla's Recall@20 margin over LightGCN was p = 0.056 at n = 3 and is **p = 0.0115 at n = 5** (+8.98 sigma), so v6 now clears across-seed significance on all three accuracy metrics — NDCG p = 0.000177, Precision p = 0.00139. It also softened two negatives: v6's accuracy deficit is **not** significant on Yahoo (p = 0.154 / 0.061 / 0.154) or Coat (p = 0.093 / 0.093 / 0.062). |

**Stated goal.** Build a new pipeline that outperforms standalone **LightGCN**, **TSP** and
**PPD** on *all four* metrics simultaneously — Recall@20, NDCG@20, Precision@20, Average
Popularity@20 — on **both sparse and dense** datasets.

**Verdict.** Three things, in order of importance.

1. **The measured outcome.** The goal is achieved on one sparse dataset (Gowalla) against
   three of the four baselines, and is not achieved on any dense dataset under a clean
   protocol. The Gowalla result is real, statistically significant, and defensible. The
   "absolute 4-dimensional Pareto win across four datasets" claimed in
   `MSC_THESIS_CONTRIBUTION_GUIDE.md:17` is contradicted by the repository's own reports.

2. **The mechanism claim does not hold.** The repo's own diagnostics
   (`results/ppd_v6_yahoo_sweep.json`) show that in **all 108** Yahoo configurations the
   topological gate collapses to theta = 0.0007, the spectral debiasing strength to
   alpha = 0.0003, and the "personalized" fusion gate sits at its initialization
   lambda = 0.4998. All three headline mechanisms are inert. Whatever effect v6 has comes
   from the tail-weighted loss and the `(1 - p_i)^phi` inference reweighting — simple
   degree-based reweightings — plus a changed training recipe. See §3.D.

3. **Attribution must be corrected.** ~~TSP, DAP and PPD variants 1-4 were inherited from a
   prior author, not created in this work. The README currently claims otherwise.~~
   **CLOSED 2026-08-24, and the finding was stronger than first stated:** PPD v1-4 and DAP are
   not merely inherited, they are **published methods** (PPD = Islam, Wang & Zheleva,
   arXiv:2510.12959, Oct 2025; DAP = Chen et al. 2024, a baseline *within* that paper; DCD =
   DICE, WWW 2021). Corrected in `README.md`, `MSC_THESIS_CONTRIBUTION_GUIDE.md` and
   `docs/THE_STORY.md`. Net effect on the thesis is **positive**: the baselines are external
   published work, so beating them is standard comparison rather than self-ablation. See §2.

**The constructive read.** Finding 2 is not fatal to the thesis; it is a better thesis. "I
built an elaborate spectral-topological debiaser, instrumented it, and found that the simple
tail reweighting inside it accounts for the entire effect" is an honest, interesting and
defensible result. It requires the ablation in §5.5 to state properly.

---

## 1. Scorecard against the goal

From the repo's own outputs — `LightGCN/code/results/phase1_*.json` via `make_canonical.py`,
with the verdicts taken from `results/significance_*_v6.json` rather than from point estimates:

**Updated 2026-08-30, n = 5 seeds on every dataset, under the frozen Phase 0 protocol.**

| Dataset | Density | vs LightGCN | vs TSP | vs PPD v3 | vs PPD v4 | Goal met? |
|---|---|---|---|---|---|---|
| **Gowalla** | sparse | YES | YES | YES | 3/4 — AvgPop 141.61 vs 138.44 (n.s., p = 0.214) | **Near-miss (3/4)** |
| **Yahoo** | dense | NO (accuracy −, n.s.) | NO | NO | NO | NO |
| **KuaiRec** | dense | 3/4 — accuracy wins, AvgPop far worse | 3/4 | NO | 3/4 | NO |
| **Coat** | dense | NO (accuracy −, n.s.) | NO | NO | NO | NO |

**Gowalla remains one metric away, and Phase 4 established that the gap cannot be closed.** An
inference-time popularity penalty moves AvgPop below PPD v4's only by giving up exactly enough
Recall to fall back to LightGCN's; at the crossover both margins are statistically zero, and the
thesis's own validation criterion selects no penalty at all (finding **AG**).

**The defensible claim is a different one.** Against PPD v4 on Gowalla, v6's AvgPop deficit is
itself **not significant** (−3.17, p = 0.214) while its Recall advantage is large and significant
(+12.5%, 0.142878 vs 0.126957), and it more than doubles APLT@20 (0.1665 vs 0.0839). Against
LightGCN it wins all three accuracy metrics with across-seed significance (Recall p = 0.0115,
NDCG p = 0.000177, Precision p = 0.00139) at −33% AvgPop, +252% Coverage and +255% APLT. That is
a dominance claim over PPD v4 and an accuracy-plus-exposure claim over LightGCN — narrower than
"all four metrics against all baselines", and it survives the significance tests.

On the dense datasets the honest reading is *no effect rather than a loss*: v6's accuracy deficit
clears no significance threshold on Yahoo (p = 0.154 / 0.061 / 0.154) or Coat
(p = 0.093 / 0.093 / 0.062), and on KuaiRec it wins accuracy decisively while becoming markedly
*more* popularity-concentrated (finding **Z**), which is a failure of the debiasing goal, not of
the accuracy goal.

---

## 2. Provenance — what was inherited vs. authored here

Determined from git history. Re-runnable: `git show d3cad59:LightGCN/code/constants.py`.

The initial commit `d3cad59` ("first", **GinaHmdy**, 2026-03-27) — three months before this
work's first commit `6562358` (MohabMostafa, 2026-06-20) — **already contained**:

- The **entire PPD framework**: `compute_global_popularity_score`,
  `compute_personalized_preference_scores`, `compute_bias_scores`,
  `compute_d_pop_all_normalized`, `project_and_remove`, `compute_pop_bias_scores`.
- **PPD variants 1, 2, 3 and 4** — all four constants present verbatim in `constants.py`.
- `forward_post_hoc_with_bias_vector_estimation` (the v4 in-propagation loop) in `model.py`.
- The **entire TSP implementation** in `run_thesis.py`, plus 48 pre-existing TSP sweep logs.
- **`DAP.py`** (Degree-Aware Propagation) in full.
- The `--tsp`, `--ppd`, `--beta`, `--phi`, `--fairness` and all `*_tsp` flags; the
  train/valid/test loader.

**Authored by this work:**

- **PPD v5** (`VARIANT_POST_PROP_PROJECT`) and **`--ego_weight`** — both introduced in `6562358`.
- **PPD v6 / ASTD-PD in its entirety** — `PPDv6ASTD` (`model.py:550-1099`),
  `PopularityDensityStore`, `StratifiedPopularityNegativeSampler`, the dual-stream
  architecture, the spectral popularity basis, the Pareto fusion gate, the multi-objective
  loss, the unit tests (`tests/test_ppd_v6.py`).
- The genuine bug fixes: the beta-reflection bug (`e8d12d0`, `1fde1f5`), orthogonal projection
  scaling and non-negative clamping (`20abc71`, `c3151e0`).
- The Pareto/hypervolume utilities and validation-hypervolume early stopping (`main.py:104-172`).
- The experiment-runner layer.

### Two consequences

**This strengthens the framing.** Because TSP and PPD v1–v4 are prior art *from the group*,
comparing against them is legitimate baseline comparison — not self-ablations dressed up as
competitors. The goal statement ("beat LightGCN, TSP, PPD") is exactly right. Lean into it.

**But it must be attributed.** `README.md:3` currently reads *"This project introduces novel
post-hoc methods … Topological Simplex Propagation (TSP) and … Popularity Bias (PPD)"*,
claiming authorship of inherited code. `MSC_THESIS_CONTRIBUTION_GUIDE.md` presents PPD v3/v4
as Phase III contributions of this thesis. Whatever the working arrangement with the prior
author, **the thesis must state plainly which components were inherited and which are the
candidate's own.** This is the highest-priority correction in this audit — higher than any
statistical issue, because it is an authorship claim. `LightGCN/README.md:12-14` already
credits the SIGIR 2020 LightGCN paper correctly; TSP/PPD/DAP need the same treatment.

**Operational risk:** `PPDv6ASTD` does not exist at HEAD. The central contribution is
uncommitted working-tree state. Commit it before anything else.

---

## 3. Findings

### DISQUALIFYING

#### A. KuaiRec is falsified by a trivial baseline
MostPopular computed directly from `data/*/{train,test}.txt` — train items masked, test users
only, same metric definitions as `utils.py:222-266`:

| Dataset | MostPop R@20 | MostPop NDCG@20 | MostPop P@20 | Best reported model R@20 |
|---|---|---|---|---|
| Gowalla | 0.040822 | 0.031229 | 0.013635 | 0.142838 (ok) |
| Yahoo | 0.090639 | 0.039015 | 0.007491 | 0.142360 (ok) |
| Coat | 0.095386 | 0.051301 | 0.019039 | 0.125553 (ok) |
| ~~**KuaiRec**~~ | **0.123342** | **0.313162** | **0.266525** | ~~0.013903 (loses)~~ → **0.177384 (wins, 1.4x)** — see finding **P** |

> **Superseded 2026-08-25 by finding P.** The KuaiRec row above is wrong, and so is the
> diagnosis under it. `lgn-KuaiRec-2-256-best` is not a collapsed run: it scores Recall@20 =
> **0.155524** once eval candidates are restricted to the test-split universe, and it *beats*
> MostPopular (0.123342). The `Recall@5 = 0.0` symptom was the eval bug, not training. Corrected
> KuaiRec row: **MostPop 0.123342 | best model 0.177384 (PPD v6, wins)**.

~~Every model in the thesis loses to a non-personalized most-popular ranker on KuaiRec by ~9x
Recall and ~14x NDCG.~~ The "32.8x recall breakthrough"
(`MSC_THESIS_CONTRIBUTION_GUIDE.md:176`) was measured against `lgn-KuaiRec-2-256-best` at
Recall@20 = 0.00074 — a number produced by a broken candidate set, not by a collapsed model.
The breakthrough claim is withdrawn either way; the real PPD v6 margin over LightGCN is 1.14x.

**Constructive reading.** KuaiRec has ~103 test items per user over a dense, heavily skewed
catalogue. Popularity genuinely *is* the signal there, so debiasing must cost accuracy. That
is a legitimate negative result and a good thesis chapter — but only with the MostPopular row
in the table.

#### B. The Yahoo win comes from selecting on the test set
`refine_yahoo_ppd_v6.py:1-11` names the exact test numbers to beat in its docstring.
`refine_yahoo_ppd_v6.py:171-176` ranks candidates by
`m["recall@20"] - m["avgpop@20"]/10000`, parsed from the **final test** block, explicitly
discarding the `val_hypervolume` it also parsed.

- Validation-selected run → `results/yahoo_pareto_report.md`: **fails all four baselines.**
- Test-selected run → `results/yahoo_refined_pareto_report.md`: the version in the README.

Same pattern in `run_component_search.py:58-60`, `test_beta_sweep.py:53-74` (the source of the
per-dataset beta constants), `run_dcd_sweep.py`, `run_yahoo_ablations.py`,
`evaluate_canonical.py` — all select through `run_thesis.py`, which has no validation path
at all (`run_thesis.py:362`).

#### C. The dominance claim contradicts the repo's own reports
Coat fails on all four and **is silently omitted** from the guide's §5 tables while still
being counted in "four diverse datasets"; PPD v6 loses every Coat accuracy metric by ~2 sigma
(0.114685 vs 0.124446). Yahoo (validation-selected) fails all four. KuaiRec fails vs PPD
v3/v4. Gowalla fails vs PPD v4.

### DISQUALIFYING (continued)

#### D. All three headline mechanisms are inert on Yahoo — proven by the repo's own diagnostics
`results/ppd_v6_yahoo_sweep.json` logs per-epoch diagnostics for all 108 sweep configurations.
Across **every single one of them**:

| Diagnostic | What it gates | Final value (max / median over 108 configs) |
|---|---|---|
| `mean_theta_final` | the popularity-damped simplex stream S^kappa | **0.0007 / 0.0007** |
| `mean_alpha_final` | the spectral popularity projection Pi(m) | **0.0151 / 0.0003** |
| `mean_lambda_final` | the personalized Pareto fusion gate | **0.4998 / 0.4998** |

Read against the architecture:

- **theta collapses to ~0.** `m = theta*M_T + (1-theta)*M_G` (`model.py:887`). At theta = 0.0007
  the topological stream contributes ~0.07% of the message. **The simplex topology pillar is
  switched off by training in every configuration.** The theta trajectory in the JSON falls
  0.4845 -> 0.0007 within ~20 epochs.
- **alpha collapses to ~0.** `m_D = m - alpha*Pi(m) + gamma*h` (`model.py:911`). At alpha = 0.0003
  the spectral popularity subspace removes essentially nothing. **The spectral debiasing
  pillar is switched off.**
- **lambda is pinned at 0.4998 in all 108 configs**, i.e. `fuse_mlp_lambda` sits at its
  sigmoid initialization of 0.5 and never learns. **The fusion gate is a constant, not
  personalized.**

This directly contradicts `MSC_THESIS_CONTRIBUTION_GUIDE.md` §6.4, which lists
*"Diagnostic tracking verifying non-collapse of neural gates (alpha-bar, theta-bar,
lambda-fuse-bar)"* as a pillar of experimental rigour. The diagnostics were collected and
they show precisely the collapse the guide claims to have ruled out.

**What is actually producing the popularity reduction**, then, is the part of the design that
does *not* depend on alpha or theta: the tail-weighted loss `tail_weights = 1.0 + t_pos`
(`model.py:1050`) and the inference-time tail reweighting `t_i = (1 - p_i)^phi_t`
(`model.py:977-980`). Both are simple degree-based reweightings. This is a legitimate and
genuinely useful finding — *the simple tail reweighting does the work; the spectral and
topological machinery does not* — but it is the opposite of what the thesis currently claims.

#### D2. `--astd_topo_gate_init` is a dead flag, and one third of the sweep compute was wasted
`model.py:881-884`:
```python
if self.topo_gate_learnable:
    theta = self.gate_mlp_theta(feat_3)
else:
    theta = torch.full(..., self.topo_gate_init, device=dev)
```
`--astd_topo_gate_learnable` defaults to `1` (`parse.py:103`) and is **never set to 0 by any
script in the repo**, so the `else` branch is dead code and `topo_gate_init` is never read.

This is visible in the sweep output: the 108 configurations are 36 distinct settings crossed
with `astd_topo_gate_init in {0.6, 0.75, 0.9}`, and the three blocks are **byte-identical** —
the same test Recall@20 triplet (0.13521, 0.13750, 0.13819) repeats verbatim for each gate
value. Two thirds of that sweep measured nothing.

### SERIOUS

#### E. The v6 comparison is confounded — the most important missing experiment
PPD v6 changes the loss (sampled softmax + logQ correction + temperature 0.1–0.3,
`model.py:1039-1051`), the negative sampler (popularity-stratified), **and** the parameter
count (4 embedding tables vs 2) on top of the debiasing machinery. Baselines use BPR with
uniform negatives (`Procedure.py:135-168`). No ablation isolates the debiasing from the
training recipe, so the gains cannot yet be attributed to the stated contribution. The
KuaiRec and Gowalla gains are at least as plausibly explained by logQ + temperature.

#### F. Statistical claims are unsupported
No significance test anywhere in the repo. The guide claims "full mean and standard deviation
reporting"; `format_pareto_report` (`utilities.py:814-845`) prints means only.

- Yahoo Recall margin +0.001115 vs baseline seed std 0.002547 → **0.44 sigma**.
- `is_pareto_dominant`'s `eps_acc=1e-4` (`utilities.py:739`) is 25x smaller than that std, so
  every "YES" on Yahoo is within noise.
- KuaiRec PPD v6 seeds {0.01844, 0.00576, 0.01751} → 42% relative std.
- **Gowalla is the exception**: baseline Recall std 0.000183 vs a +0.002305 margin. Real.

#### G. TSP is not topological at the settings used *(SUPERSEDED by finding S -- it is worse than this)*
With `--k_tsp 1` (every README command; hardcoded at `run_component_search.py:107`),
`build_simplicial_complex_from_graph` (`run_thesis.py:148-158`) adds only nodes and edges —
no triangles, no higher-order simplices. What runs is Laplacian smoothing over a kNN graph
built from **the model's own embeddings** (`run_thesis.py:108-146`), which is circular: the
"topology" encodes what the model already believes. This is inherited code, so not the
candidate's error — but the thesis describes TSP as higher-order topological modelling, and
at K=1 that is not what executes. It explains why TSP is ~0.0% in every table.

#### H. Dead hyperparameters presented as tuned *(CLOSED 2026-08-25, Phase 0)*
After the reflection-bug fixes, `project_and_remove`'s `beta` argument is never referenced in
the body (`utilities.py:707-732`). **`--beta` has no effect on PPD v3's projection strength**,
despite per-dataset values (0.20 / 2.00 / 0.07) being swept and reported throughout.
`--ppd_beta_schedule` and `--ppd_seq_beta_v3` are likewise computed and unused
(`model.py:509-519`, `model.py:445-447`). The fixes were correct; the sweeps over these knobs
since then are void.

#### I. `--dap --ppd_variant 4` silently drops DAP
`run_thesis.py:355-356` sets `posthoc=1`, so `computer_DAP` never runs. Visible in the
README's own numbers: Yahoo "DAP+TSP+PPD v4" and "Upgraded LightGCN+TSP+PPD v4" are both
0.140542 / 0.064849 / 0.011225 — identical to 6 dp. Same on KuaiRec (0.040994 twice). Those
"hybrid" rows are plain PPD v4.

#### O. The whole framework treats all popularity as bias (NEW — 2026-08-24)
Every debiasing mechanism in the repository removes popularity *uniformly*, and PPD v6 removes
**more** the **more popular** an item is: `alpha_node` is modulated by
`rho = p^eta / (p^eta + 0.5^eta)` (`model.py:895-897`), and `t_i = (1 - p_i)^phi`
(`model.py:977-980`) boosts every tail item regardless of merit.

But popularity decomposes into a **benign** part (item quality — genuine collaborative signal)
and a **harmful** part (exposure amplification / conformity). See Zhao et al., IEEE TKDE 2022,
arXiv:2109.07946, and PDA, Zhang et al., SIGIR 2021, arXiv:2105.06067.

Removing benign popularity costs accuracy ~1:1; removing harmful popularity is close to free.
So a uniform remover can only *slide along* the accuracy/popularity trade-off, never move it —
which is precisely the observed behaviour, and precisely what the stated goal forbids.

**This finding subsumes several others.** It gives a single causal explanation for the Coat
failure (mild bias, little harmful component, so uniform removal is pure loss), the KuaiRec
failure (fully-observed dense matrix, popularity largely benign, so minimising AvgPop is
minimising signal), and the Gowalla success (sparse, strong feedback loop, mostly harmful).
It also means finding **J** is more serious than stated: AvgPop@20 is not merely incomplete, it
is *mis-specified* as a sole objective, because it penalises recommending a genuinely excellent
popular item.

**Measured, 2026-08-24** (`measure_popularity_composition.py`, `results/popularity_composition.json`).
Using the random-exposure splits that Coat and Yahoo!R3 ship and KuaiRec's fully observed test
matrix, the earned share of popularity is: **KuaiRec 26.9% (rho +0.74), Yahoo 27.8% (rho +0.43),
Coat 3.7% (rho +0.17)**; Gowalla ships no unbiased split and is reported unmeasurable.

Prediction vs outcome: **KuaiRec confirmed** (earned popularity, every model loses to
MostPopular), **Yahoo confirmed** (earned popularity, v6 loses accuracy under validation
selection), **Coat contradicted** — its popularity is 96% amplified, so debiasing should have
helped and did not. Coat's failure is therefore *not* a popularity-composition effect and remains
open; the leading hypothesis is capacity (281 test users, 300 items, four embedding tables vs
two).

**Remediation attempted and REJECTED.** Decomposing degree into a quality-predicted component and
a residual requires a per-item quality estimate. Three were tested against the Coat and KuaiRec
ground truth and all failed: structural proxies are collinear with degree (the 2nd-order density
term at rho = 0.98, making its apparent R^2 = 0.81-0.99 degree explaining degree), and the model
affinity `aff_i` (`utilities.py:103-111`) predicts true quality no better than raw degree
(+0.70 vs +0.71 on KuaiRec) while correlating with degree at rho = 0.95. In every case the
residual correlated with log degree at +0.89 to +0.98 — the decomposition collapses to raw
popularity and is a no-op. Root cause: the training log contains no counterfactual, so
exposure-free quality is not recoverable from it. This is why TIDE requires timestamps and why
these datasets were built with random-exposure splits.

**Revised remediation:** use the measured composition as a *per-dataset diagnostic* that sets
debiasing strength, not as a per-item correction. Report a harmful-popularity companion to
AvgPop@20 only on datasets where an unbiased split exists.

#### J. Only one fairness metric
AvgPop@20 is the only fairness quantity in the codebase — grep for
`gini|coverage|entropy|APLT|long.?tail` returns zero hits. It is trivially reduced by moving
mass from the #1 item to the #50 item without improving tail exposure at all. It must stay
(it is named in the goal), but it needs companions.

### MODERATE

#### K. The evaluation protocol is not fixed
Gowalla LightGCN Recall@20 appears as **four different values simultaneously**: 0.085641
(`README.md:284`), 0.121115 (`run_component_search.py:16`), 0.140533
(`results/baselines_gowalla.json`), 0.14027778
(`checkpoints/lgn-Gowalla-3-64-best.results.txt`). KuaiRec has three.
`run_ppd_v6_experiments.py:99-104` holds round-number placeholders (0.060000 / 0.032000 /
0.012000) in a dict labelled `canonical`, against a measured Coat LightGCN Recall@20 of
0.124446.

#### L. Metric-scope mismatch *(CLOSED 2026-08-25, Phase 0)*
Accuracy is computed over `testDict` users (`Procedure.py:335`); AvgPop@20 over
`range(n_users)` (`Procedure.py:259`). Overlap is 16.6% on Yahoo and 19.6% on KuaiRec — the
two axes of the reported "Pareto trade-off" are measured on different user populations.
Seen-item masking is `-1024` in `Test` (`Procedure.py:374`) but `-1e9` in
`calculate_average_popularity`; PPD v6 returns unbounded temperature-scaled logits while
LightGCN returns a sigmoid, so the `-1024` mask is not provably safe for v6.

#### M. "Popularity" in PPD v3/v4 is not popularity
`utilities.py:103-111` computes `p_i = item_emb @ mean_user_emb`, min-max normalized — an
embedding-derived affinity, never item degree. The metric AvgPop@20 uses true train degree,
so method and metric optimise different quantities. (Inherited; v6 correctly fixes this via
`PopularityDensityStore`.) The guide's stated formula `Score = e_u^T e_i - beta*p_i` is not
what is implemented — the implementation is a subspace projection.

#### N. Unused assets that would strengthen the work
`data/Coat/unbiased_valid/` and `data/KuaiRec/{biased-valid,unbiased-valid}/` contain
randomized-exposure splits — the standard instrument for evaluating popularity debiasing —
and no script references them.

---

### AH. Every shipped hyperparameter set was test-influenced -- and re-selecting them honestly changes nothing *(NEW - 2026-09-05, Phase 5, remediation item 5.7)*

The last open item in the ledger. `sweep_yahoo_ppd_v6.py` did rank its 108 configurations by
validation hypervolume, but the configuration that actually shipped came from
`refine_yahoo_ppd_v6.py`, whose `score_fn` sorts candidates by **seed-2024 test metrics**
(`refine_yahoo_ppd_v6.py:171-176`). The proof is direct: the recorded sweep's validation winner
is `alpha_base 0.15 / damping 0.05`, while the shipped Yahoo config is `0.10 / 0.10`. Every v6
number in this repository therefore rested on hyperparameters that had seen the test set.

`run_phase5_hpo.py` re-selects them properly: **96 configurations, selection reading the
`[Best Valid Results]` block only**, scored by the same 4-D Pareto hypervolume `main.py` uses for
checkpoint selection. Test metrics are recorded per configuration and never compared. Two design
choices matter: dead knobs are **not** swept (`astd_topo_gate_init` is inert while the gate is
learnable -- sweeping it is what wasted two thirds of the original Yahoo sweep, item 5.6), and
each dataset's incumbent is a grid point, so the sweep can confirm or overturn what is shipped.

**1. The incumbent was never the validation winner on any dataset.**

| Dataset | configs | incumbent rank | winner's gain in validation HV | what changed |
|---|---|---|---|---|
| Coat | 36 | 7th | +4.6% | phi 1.5 -> 2.0, alpha 0.10 -> 0.05 |
| Yahoo | 36 | 20th | +0.8% | phi 1.0 -> 1.5, alpha 0.10 -> 0.05, damp 0.10 -> 0.05 |
| Gowalla | 12 | 5th | +0.39% | alpha 0.40 -> 0.60, damp 0.05 -> 0.10 |
| KuaiRec | 12 | 7th | +0.12% | phi 1.5 -> 1.0, alpha 0.35 -> 0.20 |

**2. It made no difference.** Each winner was retrained at 5 seeds (`run_phase5_final.py`) and
compared to the shipped configuration, paired by seed:

| Dataset | Recall@20 | NDCG@20 | AvgPop@20 | APLT@20 |
|---|---|---|---|---|
| Gowalla | +0.000167 (p=0.089) | +0.000161 (p=0.027) | +0.26 (p=0.036) | -0.0003 (p=0.395) |
| Yahoo | -0.001133 (p=0.253) | -0.000498 (p=0.070) | +6.13 (p=0.211) | -0.0080 (p=0.061) |
| KuaiRec | -0.000221 (p=0.047) | -0.000905 (p=0.001) | **-6.84 (p<0.001)** | +0.0012 (p=0.004) |
| Coat | +0.000811 (p=0.760) | +0.000847 (p=0.454) | +0.61 (p=0.195) | -0.0345 (p=0.172) |

The differences that reach significance are consistent-but-negligible shifts of 0.1-0.4%, which a
5-seed paired test detects because the seeds are matched, not because the effect is large. **No
conclusion on any dataset changes.** Gowalla still wins its three accuracy metrics and still
trails PPD v4 on AvgPop; Yahoo and Coat still trail LightGCN without significance; KuaiRec still
wins accuracy while being far more popularity-concentrated.

**3. The headline survives honest selection, and slightly strengthens.** Re-running the paired
tests for Gowalla's validation-selected configuration against LightGCN
(`results/significance_gowalla_v6_valsel.json`):

| | shipped config | validation-selected |
|---|---|---|
| Recall@20 | +0.002185, p = 0.0115 (+8.98 sigma) | +0.002352, **p = 0.00582** (+9.66 sigma) |
| NDCG@20 | +0.005327, p = 0.000177 | +0.005488, p = 9.21e-05 |
| Precision@20 | +0.001313, p = 0.00139 | +0.001366, p = 0.000827 |

**4. The sweep independently reproduces finding AE.** Ranked by validation hypervolume, the grids
sort almost entirely by `astd_tail_phi`: on Gowalla the twelve configurations split into a
phi=1.5 block (ranks 1-6) and a phi=1.0 block (ranks 7-12), with alpha_base 0.2 / 0.4 / 0.6
interleaved *within* each block. On Coat the top five configurations differ in alpha_base and
agree on validation HV to seven decimal places. The ablation reached "the spectral knob does
nothing" by deleting the mechanism; the hyperparameter landscape reaches it by showing the
criterion is flat along that axis. Two independent methods, same answer.

**What this means for the write-up.** The honest sentence is not "our hyperparameters were
compromised" nor "it did not matter, so ignore it". It is: *the selection protocol was wrong, it
has been redone properly on all four datasets, and the results are unchanged within seed noise --
which is itself evidence that the method is insensitive to the knobs it exposes.* The shipped
configurations are left in `experiment_configs.py` so that every earlier result keeps meaning what
it meant; the validation-selected numbers sit beside them in `phase5_final_*.json`.

---
### AG. The four-metric goal is unreachable on Gowalla: the penalty only slides along a frontier *(NEW - 2026-08-30, Phase 4, remediation item 5.4)*

Item 5.4 asked for "a validation-selected config targeting the AvgPop gap vs PPD v4" -- the one
run that could convert Gowalla's 3/4 near-miss into the stated goal. `--astd_inference_pop_penalty`
subtracts `mu * p_i` at ranking time (`model.py:1035`), touching no weights, so
`run_phase4_inference_penalty.py` re-ranks the five already-trained v6 checkpoints across a
12-point grid instead of retraining anything.

Test metrics, 5 seeds, mean. The bar a four-metric win must clear is the *strongest* baseline on
each axis: Recall 0.140693 (LightGCN), NDCG 0.116880 (TSP), Precision 0.042418 (TSP), AvgPop
138.44 (PPD v4).

| mu | valid HV | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | APLT@20 | 4/4? |
|---|---|---|---|---|---|---|---|
| 0.0 | **0.00049279** | 0.142843 | 0.121314 | 0.043491 | 141.61 | 0.1665 | no (AvgPop) |
| 1.0 | 0.00048442 | 0.141768 | 0.120516 | 0.043149 | 139.64 | 0.1867 | no (AvgPop) |
| 1.5 | 0.00047774 | 0.141099 | 0.120069 | 0.042954 | 138.55 | 0.1984 | no (AvgPop) |
| 1.6 | 0.00047594 | 0.140930 | 0.119952 | 0.042907 | 138.33 | 0.2009 | **yes** |
| 1.75 | 0.00047434 | 0.140721 | 0.119805 | 0.042847 | 137.98 | 0.2047 | **yes** |
| 2.0 | 0.00047114 | 0.140415 | 0.119577 | 0.042746 | 137.40 | 0.2113 | no (Recall) |
| 4.0 | 0.00044108 | 0.136654 | 0.116949 | 0.041664 | 131.68 | 0.2755 | no (Recall) |

**Two findings, and the second kills the first.**

**1. Under the thesis's own selection rule, mu = 0.** Validation hypervolume -- the same 4-D
Pareto volume `main.py:144` uses for checkpoint selection -- is *maximised at zero penalty* and
falls monotonically. It cannot select otherwise: the criterion enters AvgPop as
`1/(1 + pop/1000)`, so moving 141.6 -> 131.7 buys 0.8% of one factor while the 4.3% Recall loss
hits another linearly. A pre-registered validation rule therefore keeps the unpenalised model
and the AvgPop gap stays open.

**2. The mu = 1.6-1.75 window that *does* show 4/4 is a tie, not a win.** Across-seed paired
t-tests, n = 5:

| mu | Recall vs LightGCN | AvgPop vs PPD v4 |
|---|---|---|
| 0.0 | +0.002151, p = **0.013** | -3.17, p = 0.214 (n.s.) |
| 1.6 | +0.000237, p = 0.676 (n.s.) | +0.11, p = 0.964 (n.s.) |
| 1.75 | +0.000028, p = 0.961 (n.s.) | +0.45, p = 0.846 (n.s.) |
| 2.0 | -0.000278, p = 0.620 (n.s.) | +1.04, p = 0.659 (n.s.) |

At the crossover both binding margins are statistically indistinguishable from zero. The penalty
does not find a better model; it trades Recall for AvgPop along a frontier and the point where
it clears PPD v4 on popularity is exactly the point where it stops clearing LightGCN on accuracy.
**Reporting mu = 1.6 as "the goal achieved" would be selecting a configuration on test metrics at
a margin of 0.2 sigma -- precisely the practice findings F and X were raised against.**

**What can honestly be claimed instead.** The same test gives a stronger result at mu = 0: v6's
AvgPop deficit against PPD v4 is *itself* not significant (-3.17, p = 0.214), while v6 beats
PPD v4 on Recall by +12.5% (0.142843 vs 0.126957, p < 0.01) and more than doubles APLT@20
(0.1665 vs 0.0839). Against PPD v4 the honest statement is **not** "we lose on one metric" but
"we match its popularity reduction within noise and dominate its accuracy". Against LightGCN, v6
wins all three accuracy metrics with across-seed significance and improves Coverage@20 by 252%.

A caveat on the selection protocol, stated because it is not free: on Gowalla the validation and
test *user sets are identical* (both are all 29,858 users), and AvgPop@20 is a label-free property
of the ranking, so its validation and test values coincide exactly. Selection used validation
labels for Recall/NDCG/Precision -- no test label information entered -- but the AvgPop term of
the criterion is not held out on this dataset. The conclusion does not rest on it either way,
since the selected mu is 0.

---

### AF. The gate collapse tracks the dataset, not the initialization *(NEW - 2026-08-30, Phase 3, closes finding Y's confound)*

Finding **Y** observed that alpha and theta survive on Gowalla and collapse to numerical zero on
Yahoo and Coat, and flagged the obvious confound: Gowalla also *starts* alpha four times higher
(`astd_alpha_base` 0.40 vs 0.10), so "Gowalla keeps more alpha" might mean nothing more than
"Gowalla was given more alpha to keep". `run_phase3_gate_transfer.py` swaps the knob across
datasets, 3 seeds each.

| Dataset | native | given alpha_base = 0.40 | given alpha_base = 0.10 |
|---|---|---|---|
| **Gowalla** | 0.0841 of 0.40 = **21.0%** | 0.0841 = **21.0%** | 0.0212 of 0.10 = **21.2%** |
| **Yahoo** | 0.0003 of 0.10 = 0.3% | 0.0019 of 0.40 = 0.5% | 0.0003 = 0.3% |
| **Coat** | 0.0001 of 0.10 = 0.1% | 0.0004 of 0.40 = 0.1% | 0.0001 = 0.1% |

The *retention ratio* is a property of the dataset and is invariant to where alpha starts:
Gowalla holds ~21% of whatever it is given, Yahoo ~0.4%, Coat ~0.1%. Handed Gowalla's four-times
larger start, Yahoo still collapses; handed Yahoo's smaller start, Gowalla still retains.
**Finding Y's claim 2 survives the confound: the mechanisms really are adaptive**, switching on
where popularity is exposure-driven and off where it is earned. The absolute value of alpha is
simply proportional to its initialization, which is why the raw numbers looked confounded.

Accuracy is unaffected by the swap on Gowalla (Recall 0.142780 native vs 0.142723 under Yahoo's
init, a 5.7e-05 difference, inside the non-reproducibility floor of finding **AB**).

**One limitation, and it is a code defect, not a data result.** The transfer arm also sets
`astd_topo_gate_init`, which is a **dead flag** whenever `astd_topo_gate_learnable=1` -- theta
comes from `gate_mlp_theta` and the startup warning says so (remediation item 5.6). Theta is
consequently identical across the arms (0.2560 vs 0.2542 on Gowalla), so **this experiment tests
alpha_base only**. The theta half of finding Y's confound remains formally untested; testing it
would need a learnable-gate initialization knob that does not currently exist.

---

### AE. Gowalla's ablation: the sampler owns AvgPop, the mechanisms own APLT, the spectral stream owns nothing *(NEW - 2026-08-30, Phase 2 complete on all four datasets)*

The arm that finding **AA** said must run before any component claim could be written has now
run. Gowalla, 6 arms x 3 seeds:

| Arm | Recall@20 | AvgPop@20 | APLT@20 | alpha | theta |
|---|---|---|---|---|---|
| (a) full v6 | 0.142780 +/- 0.000695 | 141.03 +/- 1.15 | 0.1655 +/- 0.0040 | 0.084 | 0.256 |
| (b) no spectral | 0.142700 +/- 0.000610 | 140.92 +/- 1.01 | 0.1655 +/- 0.0035 | 0.000 | 0.256 |
| (c) no simplex | 0.149034 +/- 0.000469 | 143.18 +/- 0.50 | 0.1300 +/- 0.0019 | 0.085 | 0.000 |
| (d) no tail | 0.146103 +/- 0.000454 | 151.02 +/- 1.55 | 0.0871 +/- 0.0032 | 0.084 | 0.228 |
| (e) recipe only | 0.149980 +/- 0.000504 | 149.64 +/- 0.49 | 0.0863 +/- 0.0011 | 0.000 | 0.000 |
| (f) + uniform negatives | 0.150450 +/- 0.000440 | 182.28 +/- 0.59 | 0.0590 +/- 0.0010 | 0.000 | 0.000 |

**1. The spectral projection does nothing, on the one dataset where it was thought alive.**
Arm (b) matches full v6 to 8e-05 Recall, 0.11 AvgPop and 0.0000 APLT -- inside the
non-reproducibility floor. Finding **Y** established that alpha *survives* on Gowalla (21% of its
initialization, against 0.3% on Yahoo); this establishes that surviving and mattering are
different things. **Gowalla was the last dataset on which the spectral debiaser could have been
claimed to contribute, and it does not.**

**2. The stratified sampler owns 79% of the popularity reduction** -- (f) 182.28 -> (e) 149.64,
against the full 182.28 -> 141.03 -- matching the 88-96% measured on Yahoo. The three named
mechanisms split the remaining 21%.

**3. The named mechanisms own 74% of the tail exposure** -- (e) 0.0863 -> (a) 0.1655 of the
(f) 0.0590 -> 0.1655 total, with the tail reweighting the largest single contributor (removing
it alone costs 0.078 APLT). This is the half of the contribution that survives the ablation.

**4. Debiasing costs accuracy, and the cost is now measured.** Full v6 gives up 0.0077 Recall
(-5.1%) against arm (f). The trade is real and should be reported as such rather than as a free
lunch.

The consequence for the write-up is the same on all four datasets: **the fairness claim rests on
the popularity-stratified negative sampler and the tail reweighting -- two simple degree-based
interventions -- not on the spectral-topological machinery the method is named for.** The
exposure metrics (APLT, Coverage) are where the named mechanisms earn their place.

---
### AD. Evaluation recomputed the full graph propagation once per user batch *(NEW - 2026-08-28, performance, no effect on results)*

`getUsersRating` begins with `self.computer()` -- the full dual-stream multi-layer propagation
over the whole graph -- and `Procedure` calls it once per *batch of users*. With `--testbatch`
defaulting to **100**, Gowalla's 29,858 evaluation users produce **299 batches**, so a single
evaluation pass performed 299 identical propagations. The propagation depends only on the
weights, never on which users are being scored, so 298 of them were pure waste. Validation runs
twice per epoch (`Procedure.Valid` plus `calculate_average_popularity`) for up to 100 epochs.

Measured on Gowalla (29,858 users, 3 layers, 7.7M simplex edges):

| Operation | Cost |
|---|---|
| `computer()` -- full propagation | **145.4 ms** |
| per-batch scoring that genuinely differs | **3.9 ms** |
| evaluation pass as written (299 propagations) | **44.6 s** |
| evaluation pass with one propagation | **1.3 s** |

**97% of evaluation time was redundant**, which over a 100-epoch run is **2.48 h of a 3.5 h
training run**.

**Fix and why it cannot change a result.** `computer()` is now cached in eval mode, guarded by
`not self.training`, with `train()` overridden to clear the cache so it can never outlive the
weights that produced it. `PPDv6ASTD` contains no dropout, no `self.training` branch and no RNG,
so `computer()` is a pure function of the weights; LightGCN's graph dropout is already gated on
`self.training`, so its eval-mode propagation is pure too. Training is untouched -- the cache is
bypassed entirely whenever `self.training` is true.

**Verified by exact equality, not by tolerance.** Re-evaluating a fixed checkpoint before and
after the change reproduces every metric to full float precision:
`recall 0.1367217151`, `ndcg 0.1168003693`, `precision 0.0411966642`, `gini 0.9063642588`,
`coverage 0.4762450892`, `aplt 0.2550087079`, `tail_recall 0.0353448170`, `avgpop 129.8758506933`
-- all `==`, not `approx`. Evaluation from fixed weights is deterministic even on Gowalla (finding
**AB** concerns *re-training*), so exact equality is the correct acceptance test here.

**Effect.** Gowalla PPD v6 runs drop from ~3.5 h to ~1.1 h; the remaining Phase 2 Gowalla work
falls from ~56 h to ~18 h. The same fix applied to LightGCN is also bit-identical but yields no
speedup (1.0x) -- its propagation is cheap, and the cost here is specific to v6's 7.7M-edge
simplex adjacency.

**Nothing about the thesis changes.** No metric, no checkpoint-selection decision and no claim is
affected; this is the same computation performed once instead of 299 times. It is recorded
because it explains why the Phase 2 timeline was originally estimated at ~56 h and is not, and
because a reader comparing wall-clock figures across the project's history needs to know the
protocol did not change.

---

### AB. Gowalla PPD v6 training is not reproducible bit-for-bit; Yahoo and Coat are *(NEW - 2026-08-28)*

Deduplicating the Gowalla queue required checking an assumption first: that training the same
configuration with the same seed twice gives the same model. On Yahoo it does. On Gowalla it
does not.

The Phase 1 v6 run and the Phase 2 `a_full` arm are the *same command line* -- `a_full` applies
no overrides, so it is the dataset's native PPD v6 configuration. Both ran after the finding-**X**
fix. Comparing them:

| Dataset | seed | Phase 1 v6 Recall@20 | Ablation `a_full` Recall@20 | abs diff | as % of seed std |
|---|---|---|---|---|---|
| **Gowalla** | 2024 | 0.143263 | 0.143209 | 5.38e-05 | **5.9%** |
| **Gowalla** | 2025 | 0.141837 | 0.141855 | 1.77e-05 | **1.9%** |
| Yahoo | all 3 | \-- | \-- | **0.0 (identical to 8 dp)** | 0% |
| Coat | all 3 | \-- | \-- | **0.0 (identical to 8 dp)** | 0% |

Gowalla's run-to-run reproduction error is roughly 2-6% of its own seed standard deviation
(0.000910 at n=5) -- small, but not zero, and the only dataset where it is non-zero. The likely
cause is non-deterministic CUDA reductions (atomics in sparse scatter/gather) that Gowalla's much
larger graph exercises and the small dense datasets do not.

**Why this matters beyond bookkeeping.** Finding **X** reported that the selection-leak fix moved
Gowalla by "+0.01%" while leaving Yahoo and Coat bit-identical, and read that as a small real
effect. It was not: **+0.01% is inside Gowalla's nondeterministic reproduction error**, so the
correct statement is that the fix changed Gowalla by *no measurable amount either* -- the same
conclusion, but now for the right reason. Finding **X** has been amended accordingly.

**Consequences to observe from here on.**

1. "Bit-identical" as a verification standard works on Yahoo, KuaiRec and Coat. **On Gowalla the
   standard must be "within ~6e-05 on Recall@20"**, and any Gowalla difference smaller than that
   is not evidence of anything.
2. Checkpoint *reuse* remains exact -- evaluating the same weights twice is deterministic. It is
   *retraining* that drifts. So the dedup below is safe precisely because it reuses weights
   rather than assuming a retrain would reproduce them.
3. Where an arm's seeds must be comparable to each other, they should share a training
   provenance. The Gowalla `a_full` arm's seeds 2024 and 2025 were trained by the ablation
   driver, so seed 2026 was left to train the same way rather than being copied from the Phase 1
   v6 checkpoint -- a ~3.5 h saving declined to keep one arm internally consistent.

---

### AC. Two transfer arms were duplicates of `native`, and one had already run silently *(NEW - 2026-08-28)*

`run_phase3_gate_transfer.py` swaps `astd_alpha_base` / `astd_topo_gate_init` across datasets to
separate adaptation from initialization (finding **Y**). Its arms are:

| Arm | alpha_base | topo_gate_init |
|---|---|---|
| `native` | the dataset's own | the dataset's own |
| `gowalla_init` | 0.40 | 0.40 |
| `yahoo_init` | 0.10 | 0.75 |

But **Gowalla's native configuration *is* 0.40 / 0.40, and Yahoo's native *is* 0.10 / 0.75.** So
on each of those datasets one "transfer" arm is the native arm under a different name, and the
`native` arm is in turn identical to Phase 2's `a_full` arm, which applies no overrides at all.

**On Yahoo this already happened, silently**: `native` and `yahoo_init` were both trained and
agree to eight decimal places on all six metrics, as does `a_full`. Cheap there. On Gowalla the
same duplication would have cost **six redundant training runs at ~3.5 h each, ~21 GPU-hours**,
to retrain a configuration already trained twice.

**Fixed by making the equivalence explicit rather than accidental.** `alias_of()` detects arms
whose built command line matches an earlier arm's and reuses its result, tagging it `alias_of` so
it can never be counted as an independent replicate. `reusable_checkpoint()` additionally reuses a
Phase 2 `a_full` checkpoint when the command lines match, copying the weights and re-running
*evaluation only*. Gowalla's stage 7 therefore trains **three** models (`yahoo_init`) instead of
nine.

The equivalence was verified before being relied on, on the dataset where the duplicate work had
already been done: Yahoo's `native`, `yahoo_init` and `a_full` agree to 8 dp across all six
metrics, while `gowalla_init` -- genuinely different there -- does not. Per finding **AB**,
reusing weights is exact even on Gowalla; it is retraining that drifts.

**The general lesson.** An ablation grid defined as "override X to value V" silently collapses
whenever some dataset's default already equals V. The grid should either be defined relative to
each dataset or be checked for collisions at build time -- which `alias_of()` now does.

---

### AA. Arm (e) is not a clean "no debiasing" control -- the training recipe is itself a debiaser *(NEW - 2026-08-27, Phase 2, limitation of my own design)*

Yahoo's ablation, 5 arms x 3 seeds:

| Arm | Recall@20 | NDCG@20 | AvgPop@20 | APLT@20 | Coverage@20 | alpha | theta | lambda |
|---|---|---|---|---|---|---|---|---|
| (a) full v6 | 0.138287 +/- 0.000375 | 0.062616 | 527.13 +/- 15.02 | 0.1428 | 0.6253 | 0.0003 | 0.0007 | 0.4998 |
| (b) no spectral | 0.138287 +/- 0.000375 | 0.062617 | 527.13 +/- 15.01 | 0.1427 | 0.6250 | 0.0000 | 0.0007 | 0.4998 |
| (c) no simplex | 0.139127 +/- 0.000881 | 0.062817 | 535.86 +/- 9.10 | 0.1356 | 0.6183 | 0.0020 | 0.0000 | 0.4998 |
| (d) no tail reweighting | 0.141018 +/- 0.002383 | 0.063330 | 521.89 +/- 16.55 | 0.1273 | 0.5880 | 0.0050 | 0.0007 | 0.4998 |
| (e) recipe only | 0.141699 +/- 0.002161 | 0.063660 | 529.71 +/- 9.10 | 0.1224 | 0.5770 | 0.0000 | 0.0000 | 0.4998 |

**1. The spectral projection is provably inert on Yahoo.** Arm (b) reproduces arm (a) to six
decimal places on Recall (0.138287 both), to five on NDCG, and to 0.01 on AvgPop. Deleting the
mechanism changes *nothing measurable*. This is the strongest form finding **D** could take: not
"the effect is small" but "the effect is numerically absent".

**2. Lambda = 0.4998 in all five arms**, identical to four decimals. See finding **Y**.

**3. And the result I did not expect -- arm (e) is not the control I designed it to be.**
Compare it against LightGCN, which is the true no-debiasing reference:

| Metric | LightGCN | (e) recipe only | (a) full v6 | share of v6's gain already in (e) |
|---|---|---|---|---|
| Recall@20 | 0.141245 | 0.141699 (+0.32%) | 0.138287 | -- |
| NDCG@20 | 0.064582 | 0.063660 (-1.43%) | 0.062616 | -- |
| **AvgPop@20** | 595.72 | **529.71 (-11.08%)** | 527.13 (-11.5%) | **~96%** |
| **APLT@20** | 0.1101 | **0.1224 (+11.21%)** | 0.1428 (+29.7%) | **~38%** |

Arm (e) has every one of PPD v6's three debiasing mechanisms disabled, and it *still* delivers
**96% of v6's entire AvgPop reduction** and over a third of its APLT gain, at accuracy
indistinguishable from LightGCN.

The reason is a design error in the ablation, and it is mine. Arm (e) still trains through
`StratifiedPopularityNegativeSampler`, which draws negatives by popularity decile
(`--astd_neg_uniform_ratio / matched_ratio / hard_ratio`). **A popularity-stratified negative
sampler is a debiasing mechanism.** So "recipe only" is really "recipe, including an implicit
fourth debiaser, with the three explicit ones off".

**What this means for the thesis, stated plainly.** Finding **E** asked whether v6's gains come
from the debiasing or from the changed training recipe. The answer on Yahoo is that **most of the
popularity reduction comes from the sampler, not from the spectral-topological machinery the
contribution is named after**. That is a more interesting result than the one currently written
down, but it is also a smaller claim, and it must be reported as such.

**Arm (f) has now run on Yahoo, KuaiRec and Coat, and it splits the credit in a way neither the
original claim nor my first correction anticipated.** Arm (f) is arm (e) plus uniform negative
sampling -- every debiasing influence, explicit or implicit, switched off.

| Dataset | AvgPop (f -> e -> a) | sampler's share | sigma | APLT (f -> e -> a) | sampler's share | sigma |
|---|---|---|---|---|---|---|
| Yahoo | 601.11 -> 529.71 -> 527.13 | **97%** | **7.56** | 0.1240 -> 0.1224 -> 0.1428 | **-8%** | 0.27 |
| KuaiRec | 1708.67 -> 1627.87 -> 1616.60 | **88%** | **8.89** | 0.0016 -> 0.0059 -> 0.0102 | **50%** | **5.64** |
| Coat | 26.35 -> 26.62 -> 25.46 | -30% | 0.35 | 0.2843 -> 0.2471 -> 0.3117 | -136% | 0.92 |

**1. The popularity *reduction* is the sampler's.** On the two datasets where the effect is
resolvable, the stratified negative sampler accounts for **97% and 88%** of the AvgPop@20
reduction, at 7.6 and 8.9 sigma. The three named mechanisms contribute the remaining 3-12%.

**2. But the long-tail *exposure* is the mechanisms'.** On Yahoo the sampler contributes
essentially nothing to APLT@20 (-8%, 0.27 sigma -- indistinguishable from zero) while the named
mechanisms deliver the entire +0.0204 gain. On KuaiRec the split is even, 50/50 at 5.6 sigma.
So the two components are not doing the same job at different strengths; **they are doing
different jobs.** The sampler lowers the *mean popularity* of what gets recommended; the
spectral/simplex/tail machinery is what actually pushes items into the tail.

**3. Coat cannot resolve any of this** and its point estimates reverse sign (the sampler appears
to *hurt* there). At 0.35 and 0.92 sigma these are noise, and Coat must not be quoted in this
decomposition beyond saying it is uninformative.

**Why this materially improves the thesis's position.** My earlier statement -- that most of the
debiasing comes from a component the contribution is not named after -- is correct *for
AvgPop@20* and misleading as a general claim. Finding **Z** established that AvgPop@20 cannot
carry a fairness claim on its own, because a Random ranker wins it on every dataset. So the
metric the sampler dominates is precisely the metric that was already known to be weak, and the
metrics that measure real catalogue exposure -- APLT@20, and by extension Coverage@20 -- are
where the named mechanisms do the work.

The defensible claim is therefore narrower than the original and stronger than my first
correction: **PPD v6's spectral-topological machinery is what produces long-tail exposure; its
negative sampler is what lowers mean recommended popularity; and the two are separable, measured,
and should be reported separately.**

**Resolved 2026-08-30.** Gowalla's arm (f) has now run, and the Yahoo pattern holds: the
sampler owns 79% of the AvgPop reduction, the named mechanisms own 74% of the APLT gain, and the
spectral stream contributes nothing. The contribution stands on the exposure metrics. Full
numbers in finding **AE**.

---

### Z. On KuaiRec every method except PPD v6 loses to MostPopular, and v6 wins by becoming 83% as popularity-concentrated *(NEW - 2026-08-27, sanity floors)*

`check_sanity_floors.py` turns the remediation plan's two reference rankers into an executable
gate. Every model clears the **Random** floor on every accuracy metric on all four datasets, so
there is no repeat of the finding-**P** class of harness failure. The **MostPopular** floor is
where it gets interesting.

**KuaiRec, NDCG@20:**

| Ranker | NDCG@20 | AvgPop@20 |
|---|---|---|
| **MostPopular** (no personalization at all) | **0.313162** | 1948.98 |
| LightGCN | 0.304830 | 1148.79 |
| TSP | 0.302951 | 1139.84 |
| PPD v4 | 0.303910 | 1158.82 |
| PPD v3 | 0.119512 | 392.20 |
| **PPD v6** | **0.395928** | **1616.56** |

Every method in the study except PPD v6 **loses to a ranker that shows all 1,410 users the same
20 items**. PPD v3 loses to it on all three accuracy metrics, not just NDCG.

And the way v6 beats it is the point: v6's AvgPop@20 of 1616.56 is **82.9% of MostPopular's
1948.98**. It does not defeat the popularity ranker by being less popularity-driven; it defeats
it by being *almost as* popularity-driven while retaining some personalization.

**This is the cleanest empirical statement of the thesis's central question.** "When is
popularity bias actually bias?" has an answer on KuaiRec: it is not bias, it is the signal.
Measured earned-popularity share there is **26.9%** (Part 2.6), the catalogue is dense at ~103
relevant items per user, and under those conditions the popularity ranker is close to optimal.
A debiaser is being asked to discard the most predictive feature in the dataset.

So v6's KuaiRec "fairness reversal" (finding **V**: AvgPop +41%, Coverage -80%) is better read
as v6 *correctly discovering* that the accuracy half of its objective is maximised by moving
toward MostPopular. The hypervolume it optimises contains accuracy; the data says popularity buys
accuracy; the gate does what it was told. **The failure is not in the optimiser, it is in
expecting a debiaser to help on a catalogue where popularity is earned.**

**Two consequences for the write-up.**

1. **MostPopular belongs in every KuaiRec table**, not in a footnote. A KuaiRec table without it
   invites the reader to be impressed by numbers that a one-line baseline beats.
2. **PPD v3's KuaiRec row needs an explicit health warning.** It loses to MostPopular on all
   three accuracy metrics while posting the best AvgPop of any method (392.20). That combination
   -- trivial-baseline-beating fairness with sub-trivial accuracy -- is what a degenerate
   solution looks like, and it should be labelled as one rather than presented as a favourable
   fairness/accuracy trade.

**Also confirmed: Random wins AvgPop@20 on all four datasets** (Gowalla 15.8 vs the best model's
139.9; Yahoo 86.5 vs 509.3; KuaiRec 86.9 vs 392.2; Coat 11.7 vs 25.5). This is the standing
argument for why AvgPop@20 cannot carry a fairness claim alone, and why Coverage@20 and APLT@20
were added in Phase 0.

---

### Y. The fusion gate saturates at its ceiling, and the mechanisms are alive only where the method wins *(NEW - 2026-08-27, Phase 2)*

Two corrections to finding **D**, both sharper than the original.

**1. Lambda does not collapse to its initialization -- it saturates at its configured ceiling.**
Finding **D** reported that the "personalized" fusion gate "sits at its initialization
lambda = 0.4998" across all 108 Yahoo configurations. That diagnosis was an artifact of Yahoo's
`astd_fuse_lambda_max` happening to equal its init. Across three datasets with three *different*
ceilings:

| Dataset | `astd_fuse_lambda_max` | observed final lambda | % of ceiling |
|---|---|---|---|
| Gowalla | 0.35 | 0.3500 | **100.0%** |
| Yahoo | 0.50 | 0.4998 | **99.96%** |
| Coat | 0.30 | 0.2975 | **99.2%** |

The gate pins itself to whatever ceiling the operator sets. The consequence is worse than "the
gate is inert": the fusion weight is **a hyperparameter wearing the costume of a learned,
per-user mechanism**. `astd_fuse_lambda_max` does not bound the gate, it *is* the gate. Anywhere
the write-up describes lambda_u as personalized or adaptive, it is describing a constant that the
configuration file chose. This should be stated plainly and the parameter reported as what it is.

**2. Alpha and theta are *not* universally inert -- they survive precisely on the dataset where
PPD v6 wins.** Finding **D** generalized "all three mechanisms are inert" from Yahoo alone. The
ablation's `a_full` arm says otherwise:

| Dataset | alpha_base -> final alpha | topo_gate_init -> final theta | v6 outcome |
|---|---|---|---|
| **Gowalla** | 0.40 -> **0.0830** (21% retained) | 0.40 -> **0.2270** (57% retained) | **Pareto win** |
| Yahoo | 0.10 -> 0.0000 (collapsed) | 0.75 -> 0.0004 (collapsed) | accuracy loss (n.s.) |
| Coat | 0.10 -> 0.0001 (collapsed) | 0.60 -> 0.0216 (collapsed) | accuracy loss |

On Gowalla the spectral projection retains a fifth of its base strength and the topological gate
retains over half of its initialization. On the two datasets where v6 fails to win, both
collapse to numerical zero. **The machinery switches itself off exactly where it does not help,
and stays on exactly where it does.**

If this survives the remaining seeds it is the best result in the thesis, because it is the
mechanism-level version of the central framing. Gowalla is sparse, and its popularity is
plausibly exposure-driven; Yahoo and KuaiRec have **27.8%** and **26.9%** measured *earned*
popularity (Part 2.6 of `docs/THE_STORY.md`). A debiaser given little harmful popularity to
remove learns to remove nothing -- which is correct behaviour, not failure. The honest claim
becomes *"the mechanisms are adaptive, and their adaptation is legible: they activate in
proportion to how much of the popularity signal is exposure rather than quality."*

**Confidence, stated honestly.** Claim 1 (lambda saturation) holds on Yahoo, KuaiRec and Coat:
three distinct ceilings, every seed, every ablation arm including those that delete lambda's own
inputs.

> **Qualification found 2026-08-28 — Gowalla is the exception.** Its ceiling is 0.35, and the
> ablation's `a_full` arm gives lambda = **0.3500** on seed 2024 (saturated) but **0.2602** on
> seed 2025 — 74% of the ceiling. So the correct statement is *"lambda saturates at its ceiling
> on every dataset where the mechanisms are inert, and varies below it on Gowalla, the one
> dataset where they are not"* — which is consistent with claim 2 rather than a problem for it,
> but it is **not** the unqualified "always saturates" I first wrote. Any write-up must use the
> qualified form. The remaining Gowalla seeds will settle how much lambda actually moves.

Claim 2 was confounded when first written -- Gowalla also *starts* alpha four times higher
(0.40 vs 0.10), so "Gowalla keeps more alpha" might have meant nothing more than "Gowalla was
given more alpha to keep". `run_phase3_gate_transfer.py` swaps the two knobs across datasets to
separate the explanations. **Coat has now run, and it removes the confound:**

| Coat arm | alpha start | alpha final | **% kept** | theta start | theta final | Recall@20 |
|---|---|---|---|---|---|---|
| native | 0.10 | 0.0001 | **0.1%** | 0.60 | **0.0216** | 0.114685 |
| **gowalla_init** | **0.40** | 0.0004 | **0.1%** | 0.40 | **0.0216** | 0.114685 |
| yahoo_init | 0.10 | 0.0001 | **0.1%** | 0.75 | **0.0216** | 0.114685 |

Two things to read here.

**The retained *fraction* of alpha is invariant to the starting value.** Handed Gowalla's 4x
larger alpha_base, Coat's final alpha scales exactly 4x (0.0001 -> 0.0004) and the proportion
retained stays pinned at 0.1%. Coat destroys 99.9% of whatever spectral strength it is given.

**Theta converges to the same absolute value from every starting point.** Started at 0.60, 0.40
or 0.75, it lands on **0.0216 every time**. Its final value is determined by the data, not by the
initialization -- which is about as clean a demonstration of data-driven adaptation as this
setup can produce.

Recall@20 and lambda are also identical across all three arms, so these knobs are inert on Coat
in every sense.

**Yahoo agrees, and adds the same pattern for theta:**

| Yahoo arm | alpha start | alpha final | % kept | theta start | theta final | Recall@20 |
|---|---|---|---|---|---|---|
| native | 0.10 | 0.0003 | **0.3%** | 0.75 | **0.0007** | 0.138287 |
| **gowalla_init** | **0.40** | 0.0019 | **0.5%** | 0.40 | **0.0007** | 0.138427 |

Handed Gowalla's 4x larger alpha_base, Yahoo still retains only 0.5% of it -- still collapsed.
And theta converges to **0.0007 from both 0.75 and 0.40**, exactly as it converged to 0.0216
from 0.60, 0.40 and 0.75 on Coat. On both datasets **theta's final value is a property of the
dataset, not of where it started.**

**What this does and does not license.** The negative direction is now established on two
datasets: giving a dataset whose gates collapse the initialization of the dataset whose gates
survive does **not** revive them. So the collapse on Coat and Yahoo is a property of Coat and
Yahoo.

It does **not** yet establish the converse -- that Gowalla's retention (21% of alpha, 57% of
theta) survives Yahoo's *smaller* initialization. That asymmetry matters, because Gowalla is the
dataset the thesis's positive claim rests on. Stage 7 (`run_phase3_gate_transfer.py Gowalla`,
`yahoo_init` arm) is the decisive run. **Until it lands, claim 2 is supported in one direction
only and must be written that way.**

---

### X. The Phase 0 scope fix leaked the test user population into checkpoint selection *(NEW - 2026-08-26, self-inflicted, CLOSED same day)*

Phase 0.2 fixed finding **L** by restricting `calculate_average_popularity` to `testDict` users,
so that accuracy and popularity would describe the same people. Correct for *reporting*. But
`main.py:130` also calls that function -- to build the validation hypervolume that selects which
epoch's checkpoint to keep:

```python
val_results = Procedure.Valid(dataset, Recmodel, epoch, w, ...)   # validation users
val_pop     = Procedure.calculate_average_popularity(...)          # TEST users  <-- leak
val_hv      = utilities.compute_pareto_hypervolume(val_metrics)    # three from valid, one from test
```

So three of the four components of the selection criterion were validation-derived and the
fourth was not. This is not label leakage -- AvgPop reads item degrees, not test labels -- but it
is population leakage: every PPD v6 checkpoint was chosen partly by how it behaved on the exact
users it would later be scored on. It is precisely the class of defect finding **B** is about,
and Phase 0 introduced it while fixing a different one.

**Scope, measured rather than assumed.** The size of the leak is the divergence between the
validation and test user populations:

| Dataset | valid users | test users | overlap | Jaccard | Effect on selection |
|---|---|---|---|---|---|
| Gowalla | 29,858 | 29,858 | 29,858 | **1.0000** | **none** -- populations are identical |
| KuaiRec | 1,410 | 1,410 | 1,410 | **1.0000** | candidate mask only (3,176 valid items vs 3,247 test) |
| Coat | 261 | 281 | 261 | 0.9288 | small (valid is a subset of test) |
| **Yahoo** | **11,130** | **2,383** | 2,050 | **0.1788** | **substantial** |

Gowalla -- the dataset carrying the thesis -- is provably unaffected, because its validation and
test users are the same 29,858 people and it has no candidate mask. Yahoo is the real casualty.

**Fix.** `calculate_average_popularity` gained a `split` parameter ("test" by default, so every
reporting call is unchanged) which selects both the user population *and* the matching candidate
universe; `main.py` and `run_thesis.py`'s `[Valid]` block now pass `split="valid"`.

**Verified by invariance, the same check that validated the candidate mask.** Re-evaluating Coat
LightGCN seed 2024 after the change reproduces all seven stored metrics bit-identically --
`recall@20 0.13019917`, `ndcg@20 0.07134334`, `precision@20 0.02348754`, `avgpop@20 27.424021`,
`gini@20 0.82700119`, `coverage@20 0.35333333`, `aplt@20 0.22846975` -- while the newly separated
validation AvgPop reads 27.371456, confirming the split parameter actually bites.

**Consequence, and the outcome of the re-run.** The selection path is v6-only (baselines
early-stop on validation BPR loss and were never affected), so only PPD v6 checkpoints were
contaminated. All 12 were quarantined and Phase 1 v6 plus the entire Phase 2 ablation were re-run
under the corrected protocol rather than patched.

**The re-run changed essentially nothing, and that is the useful result:**

| Dataset | Recall@20 pre-fix -> post-fix | Largest change on any metric |
|---|---|---|
| Yahoo | 0.138287 -> 0.138287 | **bit-identical** on all 6 metrics |
| Coat | 0.114685 -> 0.114685 | **bit-identical** on all 6 metrics |
| Gowalla | 0.142825 -> 0.142836 | APLT@20, **-0.14%** *(see finding **AB**: this is inside Gowalla's nondeterministic reproduction error, so it is not a measured effect of the fix at all)* |
| KuaiRec | 0.177441 -> 0.177448 | APLT@20, **-0.35%** |

Every change is one to two orders of magnitude inside the seed standard deviation. Yahoo -- the
dataset with the *worst* population divergence (Jaccard 0.18) -- reproduced bit-identically,
meaning the leaked criterion selected the same epoch the clean one does. So the leak was real in
principle and inert in practice.

That is worth reporting rather than quietly dropping. "We found a selection leak, fixed it,
re-ran every affected experiment, and the numbers did not move" is a much stronger statement
than never having looked -- and it is only available because the re-run was actually done instead
of the defect being argued away as immaterial. The Phase 2 Coat ablation reported in finding **U** was discarded mid-flight
for the same reason: `main.py` was edited while it was running, so its later arms would have used
a different selection rule from its earlier ones -- exactly the "mutually incomparable numbers"
failure the plan exists to prevent.

**The lesson worth writing down.** A fix aimed at reporting silently changed selection because
one function served both. Restricting a metric's scope is only safe once you know every caller;
`grep` for call sites before changing what a shared function measures.

---

### W. Every reported Coat and KuaiRec number is already on a randomized-exposure test split *(NEW - 2026-08-26, Phase 4)*

Item 5.10 and finding **N** recorded that `data/Coat/unbiased_valid/` and
`data/KuaiRec/{biased-valid,unbiased-valid}/` ship randomized-exposure splits that no script
references, and the audit's own correction narrowed this to "only the *validation* split
differs". Hashing the files shows both statements need revising, and the revision is in the
thesis's favour.

**The default split directory *is* the unbiased one, byte for byte:**

| File | Default | Unbiased variant | Identical? |
|---|---|---|---|
| `Coat/train.txt` | `4a1c2975e4c1` | `unbiased_valid/train.txt` `4a1c2975e4c1` | **yes** |
| `Coat/valid.txt` | `8b23ad561e0b` | `unbiased_valid/valid.txt` `8b23ad561e0b` | **yes** |
| `Coat/test.txt` | `cea8e5e0cd72` | `unbiased_valid/test.txt` `cea8e5e0cd72` | **yes** |
| `KuaiRec/train.txt` | `cc86a527fb5e` | `unbiased-valid/train.txt` `cc86a527fb5e` | **yes** |
| `KuaiRec/valid.txt` | `5d495e9b951b` | `unbiased-valid/valid.txt` `5d495e9b951b` | **yes** |
| `KuaiRec/test.txt` | `b2664fd6e56d` | `unbiased-valid/test.txt` `b2664fd6e56d` | **yes** |

So the scripts were never ignoring the unbiased splits -- they have been loading them all along
under the default path. **Every Coat and KuaiRec result in this repository is already an
evaluation on a randomized-exposure test set**, which is the standard instrument for judging
popularity debiasing. The thesis has been entitled to that claim and has never made it. It
should: it is a credibility upgrade obtained for free.

**And the audit's "only validation differs" note is wrong for KuaiRec.** `biased-valid/` is a
different split of all three files, not a validation swap:

| Split | train interactions | valid users | valid interactions | valid items | test interactions | test items |
|---|---|---|---|---|---|---|
| `KuaiRec/` (unbiased) | 936,390 | **1,410** | 71,535 | 3,176 | 145,181 | 3,247 |
| `KuaiRec/biased-valid/` | 822,067 | **7,141** | 93,606 | **6,893** | 216,716 | 3,275 |

The validation sets differ by 5x in users and 2.2x in item coverage -- selecting on
`biased-valid` exposes model selection to the sparse "big matrix" distribution, selecting on the
default exposes it only to the dense "small matrix". That is a real and interesting contrast for
Phase 4, but it is a **different experiment** from the one finding N described, and it needs the
*test* split held fixed to be interpretable, since the two variants' test files differ too.

Coat is analogous: `Coat/main/test.txt` holds 1,862 interactions over 297 items against the
default `Coat/test.txt`'s 1,245 over 288, on an identical training set.

**Consequence for the plan.** Phase 4 shrinks from "evaluate on unbiased splits" (already done,
unknowingly) to the narrower and sharper question: *does selecting a model on a biased validation
distribution change what you pick, when scored on the same unbiased test set?* That is a
selection-protocol experiment, and it belongs next to Phase 3.

---

### U. Coat ablation: the tail reweighting carries the fairness gain; the spectral and simplex streams do not *(NEW - 2026-08-26, revised 2026-08-27 under the corrected protocol)*

Finding **D** predicted that all three headline mechanisms of PPD v6 are inert. Phase 2 tests
that directly. Coat, 5 arms x 3 seeds, re-run after finding **X** and reproducing the pre-fix
numbers **bit-identically**:

| Arm | Recall@20 | AvgPop@20 | APLT@20 | Coverage@20 | alpha | theta | lambda |
|---|---|---|---|---|---|---|---|
| (a) full v6 | 0.114685 +/- 0.004982 | 25.46 +/- 0.76 | 0.3117 +/- 0.0406 | 0.5533 +/- 0.1416 | 0.0001 | 0.0216 | 0.2975 |
| (b) no spectral | 0.110051 +/- 0.001948 | 25.25 +/- 0.53 | 0.3183 +/- 0.0326 | 0.6133 +/- 0.0828 | 0.0000 | 0.0247 | 0.2973 |
| (c) no simplex | 0.108553 +/- 0.004138 | 25.09 +/- 1.05 | 0.3255 +/- 0.0531 | 0.6533 +/- 0.1414 | 0.0001 | 0.0000 | 0.2972 |
| (d) no tail reweighting | 0.112797 +/- 0.010758 | 25.89 +/- 0.51 | **0.2801 +/- 0.0222** | 0.5911 +/- 0.0880 | 0.0001 | 0.0251 | 0.2973 |
| (e) recipe only | **0.120501 +/- 0.002559** | 26.62 +/- 0.31 | **0.2471 +/- 0.0182** | 0.4922 +/- 0.0993 | 0.0000 | 0.0000 | 0.2975 |

**1. The debiasing machinery is not decoration -- arm (e) prices it.** Stripping all three
mechanisms gives the **best** accuracy (0.1205, above full v6's 0.1147) and the **worst** exposure
(APLT 0.2471 vs 0.3117, -21%; Coverage 0.4922 vs 0.5533). So v6's contribution on Coat is real,
and it is a *trade*: roughly 5% of recall for 26% more long-tail exposure. That is a defensible
thing to report, and it is not what the write-up claims (which is a free lunch).

**2. Within that trade, the tail reweighting does the work.** Removing it alone (arm d) drops
APLT to 0.2801 -- most of the way to the no-debiasing arm. Removing the spectral projection (b)
or the simplex stream (c) *raises* APLT slightly (0.3183, 0.3255) while costing recall. On Coat
the two "sophisticated" mechanisms are not buying fairness at all; the `(1-p_i)^phi` reweighting
is the entire fairness mechanism, exactly as finding **D** anticipated.

**3. Lambda is invariant to architecture.** It sits at 0.2973 +/- 0.0001 across every arm --
including (b), (c) and (e), which delete the streams it is supposed to be fusing. A gate that
does not move when its inputs are removed is a constant. See finding **Y**, which identifies what
it is actually pinned to.

**Caveat, and it is a large one.** Coat is 590 nodes and its seed spread exceeds every
arm-to-arm gap: only the (a) vs (e) extremes separate by more than ~1 sigma. **No single-mechanism
conclusion above is statistically resolved on Coat.** It is reported because the direction is
consistent with Yahoo's gate values and with finding **D**, not because Coat proves it. The
decisive test is Gowalla, whose baseline seed std is 0.000183 -- two orders of magnitude tighter,
and the one dataset where alpha and theta do *not* collapse (finding **Y**). Yahoo, KuaiRec and
Gowalla arms are running; Gowalla is deliberately scheduled last because it costs ~3.5 h per run.

---

### V. Phase 1 is complete on all four datasets; Yahoo's v6 loss is not significant *(NEW - 2026-08-26)*

All five methods, all four datasets, 3 seeds, frozen protocol, with per-user dumps and paired
significance tests.

> **Resolved 2026-08-27:** the PPD v6 rows were regenerated under finding **X**'s corrected
> selection protocol. Yahoo and Coat came back **bit-identical**; Gowalla and KuaiRec moved by at
> most 0.35% on one metric, far inside the seed spread. The table below is final.
 The full tables are regenerated in `RESULTS_CANONICAL.md`. Yahoo was the
last to land and is worth recording separately, because it is the only dataset where v6's
accuracy loss is *statistically indistinguishable from zero*:

| Metric | LightGCN | PPD v6 | delta | significance |
|---|---|---|---|---|
| Recall@20 | 0.141245 +/- 0.002547 | 0.138287 +/- 0.000375 | -2.1% | -1.16 sigma, p=0.539 **n.s.** |
| NDCG@20 | 0.064582 +/- 0.000928 | 0.062616 +/- 0.000584 | -3.0% | -2.12 sigma, p=0.539 **n.s.** |
| Precision@20 | 0.010995 | 0.010827 | -1.5% | -1.61 sigma, p=0.539 **n.s.** |
| AvgPop@20 | 595.72 +/- 3.17 | **527.13 +/- 15.02** | **-11.5%** | |
| APLT@20 | 0.1101 +/- 0.0004 | **0.1428 +/- 0.0145** | **+29.7%** | |
| Gini@20 | 0.8572 | **0.8283** | better | |
| Coverage@20 | **0.6800 +/- 0.0022** | 0.6253 +/- 0.0216 | **-8.0%** | |

Read honestly, Yahoo is v6's cleanest *trade*: a measurable fairness gain (-11.5% AvgPop, +30%
APLT, lower Gini) bought with an accuracy cost that three seeds cannot distinguish from noise.
It is not a Pareto win -- Coverage moves the wrong way, and "not significant" at n=3 is a
statement about power, not a demonstration of equivalence. But it is the one dataset where the
claim "fairness improved at no *demonstrated* accuracy cost" is defensible as written.

Note also that only **16.6%** of Yahoo users are evaluated (finding L), so every Yahoo number
describes a minority of the user base. This is a limitation of the split, not of the method,
and it must be stated wherever Yahoo is quoted.

---

### S. TSP is inactive, and on Coat it does not run at all *(NEW - 2026-08-25, Phase 1)*

Finding **G** recorded that TSP at `--k_tsp 1` is kNN-Laplacian smoothing rather than
higher-order topology, and that it is "~0.0% in every table". Measuring the graph it actually
builds shows the situation is starker: TSP barely has a graph to smooth over.

`build_semantic_graph_knn_capped` (`run_thesis.py:109-145`) only adds an edge when the cosine
similarity between two embeddings clears a hardcoded `--theta_tsp 0.90`. Measured, seed 2024:

| Dataset | Nodes | Edges built | Edges per node | Effect |
|---|---|---|---|---|
| **Coat** | 590 | **0** | 0.000 | `No simplices found for any k>=1; skipping TSP update` -- **TSP never runs** |
| Yahoo | 15,382 | 294 | 0.019 | at most 3.8% of nodes touched |
| KuaiRec | 17,904 | 1,865 | 0.104 | at most 21% of nodes touched |

**Consequence for the tables.** Every "TSP" number reported on **Coat is numerically identical
to LightGCN** -- verified byte-identical across all three seeds (Recall@20 = 0.130199 /
0.124493 / 0.118646 for both methods). It is not a weak baseline there; it is the same baseline
listed twice under two names. On Yahoo and KuaiRec the vast majority of nodes are isolated
vertices in the semantic graph, so the Laplacian leaves them unchanged.

**Why this matters more than finding G.** G said TSP was mislabelled; S says TSP is largely
*absent*. The near-zero deltas throughout the thesis were read as "the topological prior adds
little". The truth is that the prior was mostly never applied, because a fixed 0.90 cosine
threshold interacts with per-dataset embedding geometry in a way nobody checked. Any claim of
the form "we outperform TSP" is, on Coat, a claim of outperforming LightGCN by 0.000.

**The threshold was swept to check whether this is just a bad setting.** It is not -- Coat,
seed 2024, `--k_tsp 1`, varying only `--theta_tsp`:

| theta | Edges built | Embedding delta norm | Recall@20 | vs LightGCN (0.130199) |
|---|---|---|---|---|
| 0.90 | **0** | n/a -- update skipped | 0.130199 | **exactly 0.000000** |
| 0.75 | 1,909 | 2.27 | 0.129865 | -0.000334 |
| 0.60 | 5,821 | 1.28 | 0.131063 | +0.000864 |

Even with 5,821 edges TSP moves Recall@20 by under 0.001, against a Coat LightGCN seed spread
of ~0.005. So TSP is **inert on Coat at every threshold**, and at the canonical 0.90 it is a
literal no-op. The two readings agree, which makes the conclusion robust rather than a
tuning artifact.

**A protocol inconsistency this exposed.** `evaluate_canonical.py:32` runs TSP at
`--theta_tsp 0.90` while `run_all_remaining_datasets.py:38` uses `0.60` for the same dataset.
The published Coat "TSP" baseline therefore depends on which script produced it, and one of
those two settings makes TSP disappear entirely.

**How to report it.** State the edge counts. TSP's `--theta_tsp` must be selected on validation
in Phase 3 like every other hyperparameter, so the baseline is given its best honest setting --
but on current evidence it will remain within noise of LightGCN on Coat either way. Do not
present a no-op as a defeated competitor. This is inherited code, so it is not the candidate's
error; failing to check it before comparing against it would be.

---

### T. The "PPD v4 hidden win on KuaiRec" was an artifact of the metric-scope bug *(NEW - 2026-08-25, Phase 1)*

**This retracts a claim made in finding P.** P reported that fixing the KuaiRec candidate set
revealed PPD v4 as a genuine Pareto win: accuracy tied with LightGCN at **-33.2% AvgPop**. Once
AvgPop@20 is computed over test users (finding L, fixed in Phase 0), that win disappears.

| Method | AvgPop@20 over ALL users (old) | over TEST users (correct) | change |
|---|---|---|---|
| LightGCN | 1024.97 | 1148.79 | +12.1% |
| TSP | 1016.43 | 1139.84 | +12.1% |
| PPD v3 | 461.89 | 392.20 | -15.1% |
| **PPD v4** | **684.22** | **1158.82** | **+69.4%** |

vs LightGCN: PPD v4 was **-33.2%**, it is actually **+0.9%** -- marginally *worse* than the baseline.

**The mechanism, and why it is the most instructive bug in the audit.** Every method's AvgPop
rose when the population narrowed, because KuaiRec's test users are its heavy users. But
LightGCN rose 12% while PPD v4 rose 69%. PPD v4 was suppressing popular items *for the ~80% of
users who are never evaluated for accuracy*, and behaving like LightGCN on the users who are.
The old metric paid it for exactly the users whose recommendations no accuracy number ever
checked. That is not debiasing; it is arbitrage against a mis-specified objective.

**Status of PPD v4 on KuaiRec, corrected:** accuracy statistically tied with LightGCN
(Recall +0.000461, p = 0.645 across seeds), AvgPop marginally worse, Coverage 0.1608 vs 0.1582
and APLT 0.0740 vs 0.0631. A near-no-op with a small tail-exposure gain -- not a Pareto win.

**Where a real fairness result does survive:** PPD v3 on KuaiRec cuts AvgPop 66% and lifts APLT
from 0.063 to **0.582** (9.2x), at a catastrophic accuracy cost (Recall -0.090, p = 0.0037).
And PPD v4 on **Gowalla** and **Yahoo** improves every exposure axis at once -- Gowalla APLT
0.046 -> 0.083 (+79%), Coverage +34%, Gini 0.9828 -> 0.9703. Those trades are real because on
those datasets test users are essentially all users (100% and, for Coat, 96.9%), so the
population mismatch could not distort them.

**Lesson for the write-up.** Two of the three headline fairness numbers in this thesis have now
been destroyed by protocol defects rather than by re-tuning. Both times the defect flattered the
proposed method. State this explicitly: the reason the surviving Gowalla result is credible is
that it was re-measured after both fixes and did not move.

---

### Q. Two more shadowed duplicate definitions, one of them live *(CLOSED 2026-08-25)*

Finding **P** turned up a duplicate `calculate_average_popularity`. Looking for the same pattern
found two more:

1. **`dataloader.py` defines `class Loader` twice** (lines 222 and 576) and `class LastFM` twice.
   Python binds the last definition, so **~350 lines are dead code** - and the first fix attempted
   in Phase 0 was written into the dead copy and did nothing. The live class is the second one.
   Both now carry the split-aware candidate universe; the dead block is left in place for now and
   flagged, because deleting 350 lines is a separate, riskier change.
2. **`run_thesis.py` held the duplicate `calculate_average_popularity`** (finding P). Removed; the
   module now re-exports `Procedure.calculate_average_popularity` so existing imports keep working.

**Lesson for the write-up:** this repository has a systematic duplicate-definition problem, and in
both confirmed cases the shadowing copy silently produced results. Before trusting any fix here,
`grep -c "def <name>"` first.

### R. Validation candidate universe would have leaked test membership *(CLOSED 2026-08-25)*

While adding the validation path (item 5.7), `_apply_candidate_mask` was found to derive its
universe from `testDict` unconditionally. Had validation-based selection been switched on with that
in place, KuaiRec model selection would have been told **which items are testable** - a leak
introduced by the fix for finding P. `candidateItems` and `validCandidateItems` are now derived per
split. Worth stating in the viva: the protocol fix needed its own protocol check.

---

### P. The KuaiRec evaluation ranked over a catalogue the test set cannot score *(CLOSED 2026-08-25)*

**What was wrong.** `Procedure.Test` masked each user's training items and then took the top-K
over all `m_items`. On Gowalla, Yahoo and Coat that is correct — their test splits cover 94-96%
of the catalogue. KuaiRec's test split is the dense fully-observed "small matrix", covering
**3,247 of 10,728 items (30.6%)**. Ranking over the full catalogue let the model spend its top-K
on items that appear nowhere in the test set and therefore could never be a hit:

| | value |
|---|---|
| distinct items LightGCN recommended | 820 |
| of those, inside the test universe | **7 (0.9%)** |
| LightGCN P@20 | 0.000461 |
| Random P@20 | 0.008830 (**19x better**) |
| MostPopular P@20 | 0.266525 (**578x better**) |

A weak model converges toward random; it does not land an order of magnitude below it. That gap
was the tell. MostPopular escaped the trap only by accident — the small-matrix items are also the
globally popular ones, so popularity-seeking lands in-universe by luck.

**Fix.** `restrict_candidates` (default on for KuaiRec/KuaiRec_Gen only, `--restrict_candidates`
to override): `dataloader.py::candidateItems` exposes the test-split item universe,
`Procedure.py::_apply_candidate_mask` applies it at all three top-K sites. Gowalla re-verified
bit-identical (Recall@20 `0.14027778`, AvgPop `214.189936`) — no other dataset moves.

**What it changed.**

| Method | Recall@20 before | after | verdict change |
|---|---|---|---|
| LightGCN | 0.000424 | **0.155524** | "collapsed baseline" → healthy, beats MostPopular |
| TSP | 0.000372 | 0.155190 | no effect on KuaiRec (was: 37.4x loss) |
| PPD v3 | 0.005990 | 0.065414 | **DOMINANT → LOSS** (-57.9% vs LightGCN) |
| PPD v4 | 0.005462 | 0.155985 | ~~hidden win surfaced: -33.2% AvgPop~~ **RETRACTED, see finding T** -- the AvgPop gain was a metric-scope artifact; corrected value is +0.9% (worse) |
| PPD v6 | 0.013903 | **0.177384** | still dominant, on real numbers (+14.1% R, -5.6% pop) |

In the 27-cell KuaiRec component search, `A. DOMINANT` verdicts fell from **30 to 1** — and that
one is `lgn/off/off/off/off`, i.e. the baseline graded against its own 3-seed mean, which is a
classification artifact rather than a result. The three surviving genuine wins are all
`lgn + PPD v4` (`C. FAIRNESS-WIN`), stable across TSP=off/standard/hodge.

**Collateral defects found while fixing this.**

1. `run_thesis.py:244` holds a **duplicate `calculate_average_popularity`** that shadows the one
   in `Procedure.py`; the sweeps used the duplicate. Patching only `Procedure.py` produced a
   byte-identical AvgPop, which is how it surfaced. Both are fixed; the duplication should be
   collapsed.
2. **Nine scripts parsed metrics with `re.search(..., "(.*?)")` and no `re.S`.** When NDCG is small
   enough for numpy to switch to scientific notation, the array repr wraps to two lines, the match
   returns `None`, and the driver's bare `continue` **silently discards a successful run**. This is
   what made the 81-run sweep appear to report only partial results: 78 rows landed, 3 completed
   runs were thrown away. All nine fixed; `run_component_search.py` now records failures and prints
   a `RUN(S) PRODUCED NO ROW — THE TABLES BELOW ARE INCOMPLETE` banner instead of failing quietly.

**Viva framing.** This is the strongest methodological item in the audit, and it should be
volunteered rather than defended: an implausible result (a GNN 19x below random) was traced to a
protocol defect, the fix was verified not to move the three unaffected datasets, and it *cost* the
thesis its headline KuaiRec claim while *revealing* a real one (PPD v4). Retracting finding #3 and
promoting PPD v4 is the honest trade.

---

## 4. Decision review — were the right calls made?

| Decision | Verdict | Better alternative |
|---|---|---|
| Framing the goal as "beat LightGCN + TSP + PPD on 4 metrics" | **Right** — falsifiable, well-scoped | Add "under a validation-only protocol" to the statement |
| Building on the group's inherited TSP/PPD codebase | **Right** — that is what prior art is for | Attribute it explicitly (§2) |
| Keeping inherited TSP as a baseline | Right to keep; wrong to describe as higher-order topology | Report it as kNN-Laplacian smoothing, which is what K=1 runs |
| PPD v5 → v6 dual-stream redesign | **Right** — this is the real contribution | — |
| Using real item degree in v6 vs the inherited embedding proxy | **Right** — fixes finding M | — |
| Validation-hypervolume early stopping in v6 | **Right** — foreground it as a methodological contribution | Apply it to the LightGCN baseline too; `main.py:176-205` uses BPR loss and selected the collapsed KuaiRec model |
| Selecting beta / theta / components on test | **Wrong** — this is what invalidates the Yahoo claim | Freeze the protocol; select on validation only |
| Changing loss + sampler + capacity alongside the architecture | **Wrong** — confounds the result | Run the recipe-only ablation (finding E) |
| AvgPop@20 as sole fairness metric | Insufficient | Keep it; add Gini, coverage, APLT, tail-recall |
| Claiming a 4-dataset Pareto win | **Wrong** — the repo's own reports say otherwise | Claim Gowalla; report dense datasets as a negative result |
| Framing KuaiRec as a 32.8x win | **Wrong — and for a different reason than first diagnosed** | The denominator was an eval-protocol artifact, not a collapsed model (finding **P**). Real margin 1.14x. Note the corrected data *reverses* the "popularity is the signal on KuaiRec" reading: PPD v4 holds LightGCN's accuracy at -33% AvgPop, so debiasing there does **not** have to cost accuracy |
| 3 seeds, no significance test | Insufficient | 5 seeds + paired test; report mean +/- std |
| Five phases, four datasets, many variants | Over-scoped for an MSc | One question, one method, one protocol, defended thoroughly |

---

## 5. Prioritised remediation plan

Ordered by value per GPU-hour.

1. **Commit the v6 code.** It is the contribution and it is untracked.
2. **Correct the attribution** in `README.md:3` and `MSC_THESIS_CONTRIBUTION_GUIDE.md`. Costs
   nothing, removes the most serious exposure.
3. **Add MostPopular + Random to the harness** and report them on every dataset. Hours of
   work; `test_pop_baseline.py` is most of it already.
4. ~~**Re-run Gowalla with a validation-selected config** targeting the AvgPop gap vs PPD v4.~~
   **DONE — negative.** The gap does not close: the penalty trades Recall for AvgPop point for
   point, and validation selects no penalty (finding **AG**).
5. ~~**Run the mechanism-isolation ablation.**~~ **DONE on all four datasets, six arms x 3
   seeds** (findings **U**, **AA**, **AE**). The prediction below held: (b) matches (a)
   everywhere, and the tail reweighting plus the negative sampler carry the effect. Five runs
   per dataset:
   (a) full v6; (b) v6 with `astd_alpha_base=0` (no spectral projection);
   (c) v6 with the simplex stream removed; (d) v6 with `astd_tail_phi=0` and the
   `tail_weights` term disabled (no tail reweighting); (e) LightGCN + sampled-softmax/logQ/
   stratified-negatives only. The prediction from the diagnostics is that (b) and (c) match
   (a), while (d) loses the popularity reduction. If that holds, report it — it is a
   cleaner contribution than the current claim.
6. **Fix or remove the dead flags** before any further sweeping: `--astd_topo_gate_init`
   (§3.D2), `--beta` for PPD v3, `--ppd_beta_schedule`, `--ppd_seq_beta_v3` (§3.H). Sweeping
   over dead knobs wasted two thirds of the Yahoo sweep.
7. ~~**Re-select all hyperparameters on validation**; re-run Yahoo and report that number.~~
   **DONE on all four datasets** (finding **AH**). The incumbent was never the validation winner
   anywhere, and re-selecting honestly moves no result outside seed noise.
8. ~~**Fix or drop** the collapsed `lgn-KuaiRec-2-256` baseline~~ — **DONE**: it was never collapsed, the eval was (finding **P**). `*-Gowalla-2-256` still to check.
9. **Add Gini / coverage / APLT**; recompute AvgPop over test users only (finding L).
10. **Evaluate on** `Coat/unbiased_valid` and `KuaiRec/unbiased-valid`.
11. ~~**5 seeds + paired significance test.**~~ **DONE on all four datasets.** It changed the
    headline: Gowalla's Recall margin moved from p = 0.056 (n=3) to p = 0.0115 (n=5).

---

## 6. What to say in the viva

- The contribution is **PPD v6 / ASTD-PD**: a dual-stream GCN that separates an accuracy
  stream from a debiased stream, removes popularity via a detached low-rank spectral subspace
  rather than a hand-crafted vector, and fuses them with a per-user gate.
- **The defensible claim is Gowalla**, and at n = 5 seeds it now clears across-seed
  significance on all three accuracy metrics: +1.6% Recall@20 (p = 0.0115), +4.6% NDCG@20
  (p = 0.000177), +3.1% Precision@20 (p = 0.00139) over LightGCN, at -33% AvgPop@20, +252%
  Coverage@20 and +255% APLT@20, on a backbone that beats MostPopular 3.4x. It beats TSP and
  PPD v3 on all four metrics, and against PPD v4 it wins Recall by +12.5% while its AvgPop
  deficit is not statistically distinguishable from zero (p = 0.214).
- **Say plainly that the four-metric goal was tested and not met, and that you tested it
  properly.** Phase 4 swept the inference-time popularity penalty over 12 values x 5 seeds.
  A window exists where the point estimates clear every baseline on all four metrics -- and
  reporting it would have been test-set selection at 0.2 sigma. Both binding margins are ties.
  Volunteering this is worth more than the claim would have been.
- **Own the ablation result, on all four datasets.** "The popularity-stratified negative
  sampler accounts for 79-96% of the AvgPop reduction, and the spectral projection I named the
  method after is inert everywhere -- including on Gowalla, the one dataset where its parameter
  survives training. What the named mechanisms do own is tail *exposure*: 74% of the APLT gain."
- **The mechanisms are adaptive, and that part held up.** Swapping the initialization across
  datasets shows retention is a property of the data, not the starting value: Gowalla keeps
  ~21% of whatever alpha it is given, Yahoo ~0.4%, Coat ~0.1% (finding **AF**).
- **If asked "were your hyperparameters selected on test?" -- say yes, then say what you did.**
  They were: `refine_yahoo_ppd_v6.py` sorted candidates by seed-2024 test metrics. All four
  datasets have since been re-selected over 96 configurations reading validation only, each
  winner retrained at 5 seeds. Nothing moved outside seed noise and the Gowalla margin
  strengthened slightly (finding **AH**). The failure and its correction are both in the record.
- **Be first to raise the negative results.** Dense datasets do not meet the goal; on KuaiRec
  every model including yours loses to most-popular, because on a dense skewed catalogue
  popularity is genuinely predictive. Framing this as a finding about *when debiasing is
  possible* is stronger than any claim the current write-up makes.
- **Be first to raise the attribution.** State which parts were inherited.
- **Own the gate collapse before the examiner finds it.** "I instrumented every gate. On
  Yahoo all three collapse, so I can show that the effect comes from tail reweighting rather
  than from the spectral subspace. That is why my ablation is structured the way it is." An
  examiner who finds this in your own logs when you have not mentioned it will conclude you
  did not read your own diagnostics.
