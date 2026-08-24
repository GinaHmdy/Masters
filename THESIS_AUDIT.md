# MSc Thesis — Contribution Audit

**Scope.** Independent audit of the contribution's integrity and validity, measured against
the work's own stated goal. Every claim below cites a file:line or a re-runnable command.

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

3. **Attribution must be corrected.** TSP, DAP and PPD variants 1-4 were inherited from a
   prior author, not created in this work. The README currently claims otherwise. See §2.

**The constructive read.** Finding 2 is not fatal to the thesis; it is a better thesis. "I
built an elaborate spectral-topological debiaser, instrumented it, and found that the simple
tail reweighting inside it accounts for the entire effect" is an honest, interesting and
defensible result. It requires the ablation in §5.5 to state properly.

---

## 1. Scorecard against the goal

From the repo's own outputs (`LightGCN/code/results/*_pareto_report.md`, produced by
`utilities.format_pareto_report`):

| Dataset | Density | vs LightGCN | vs TSP | vs PPD v3 | vs PPD v4 | Goal met? |
|---|---|---|---|---|---|---|
| **Gowalla** | sparse | YES | YES | YES | NO — AvgPop 141.11 vs 139.86 | **Near-miss (3/4)** |
| **Yahoo** | dense | YES\* | YES\* | NO | YES\* | NO |
| **KuaiRec** | dense | YES† | YES† | NO | NO | NO |
| **Coat** | dense | NO | NO | NO | NO | NO |

\* Only with a test-set-selected configuration. The validation-selected run of the same
experiment (`results/yahoo_pareto_report.md`) fails against all four. See §3B.
† Both baselines are collapsed training runs. See §3A.

**Gowalla is one metric away from the full goal.** It beats PPD v4 on Recall@20 by +12%
(0.142838 vs 0.127509) and misses only on AvgPop@20 by 1.25 units (0.9%). This is the single
most promising target for further work.

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
| **KuaiRec** | **0.123342** | **0.313162** | **0.266525** | **0.013903 (loses)** |

Every model in the thesis loses to a non-personalized most-popular ranker on KuaiRec by ~9x
Recall and ~14x NDCG. The "32.8x recall breakthrough"
(`MSC_THESIS_CONTRIBUTION_GUIDE.md:176`) is measured against `lgn-KuaiRec-2-256-best` at
Recall@20 = 0.00074 with **Recall@5 = exactly 0.0** — a collapsed run, not a baseline.
`LightGCN/code/test_pop_baseline.py` already exists and would have caught this; its output
appears in no report.

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

#### G. TSP is not topological at the settings used
With `--k_tsp 1` (every README command; hardcoded at `run_component_search.py:107`),
`build_simplicial_complex_from_graph` (`run_thesis.py:148-158`) adds only nodes and edges —
no triangles, no higher-order simplices. What runs is Laplacian smoothing over a kNN graph
built from **the model's own embeddings** (`run_thesis.py:108-146`), which is circular: the
"topology" encodes what the model already believes. This is inherited code, so not the
candidate's error — but the thesis describes TSP as higher-order topological modelling, and
at K=1 that is not what executes. It explains why TSP is ~0.0% in every table.

#### H. Dead hyperparameters presented as tuned
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

#### L. Metric-scope mismatch
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
| Framing KuaiRec as a 32.8x win | **Wrong** | Reframe: on dense skewed catalogues popularity *is* the signal — with MostPopular in the table |
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
4. **Re-run Gowalla with a validation-selected config** targeting the AvgPop gap vs PPD v4.
   This is the one run that could convert a 3/4 near-miss into the stated goal.
5. **Run the mechanism-isolation ablation.** This is now the single most valuable
   experiment, because §3.D says the mechanisms are inert. Five runs per dataset:
   (a) full v6; (b) v6 with `astd_alpha_base=0` (no spectral projection);
   (c) v6 with the simplex stream removed; (d) v6 with `astd_tail_phi=0` and the
   `tail_weights` term disabled (no tail reweighting); (e) LightGCN + sampled-softmax/logQ/
   stratified-negatives only. The prediction from the diagnostics is that (b) and (c) match
   (a), while (d) loses the popularity reduction. If that holds, report it — it is a
   cleaner contribution than the current claim.
6. **Fix or remove the dead flags** before any further sweeping: `--astd_topo_gate_init`
   (§3.D2), `--beta` for PPD v3, `--ppd_beta_schedule`, `--ppd_seq_beta_v3` (§3.H). Sweeping
   over dead knobs wasted two thirds of the Yahoo sweep.
7. **Re-select all hyperparameters on validation**; re-run Yahoo and report that number.
8. **Fix or drop** the collapsed `lgn-KuaiRec-2-256` and `*-Gowalla-2-256` baselines.
9. **Add Gini / coverage / APLT**; recompute AvgPop over test users only (finding L).
10. **Evaluate on** `Coat/unbiased_valid` and `KuaiRec/unbiased-valid`.
11. **5 seeds + paired significance test.**

---

## 6. What to say in the viva

- The contribution is **PPD v6 / ASTD-PD**: a dual-stream GCN that separates an accuracy
  stream from a debiased stream, removes popularity via a detached low-rank spectral subspace
  rather than a hand-crafted vector, and fuses them with a per-user gate.
- **The defensible claim is Gowalla**: +1.6% Recall@20, +4.7% NDCG@20, -34% AvgPop@20 over
  LightGCN, all several sigma above the 3-seed spread, on a backbone that beats MostPopular
  3.4x. It beats TSP and PPD v3 on all four metrics.
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
