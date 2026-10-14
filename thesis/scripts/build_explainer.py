#!/usr/bin/env python
"""Compute every number shown on the interactive ASTD-PD explainer page.

    python scripts/build_explainer.py [--out-json PATH] [--device cuda] [--no-tsp]   # from thesis/

Everything is read from or recomputed with the installed checkpoints and results (seed 2024 for
model-derived data); nothing is trained. The result is a dict DATA, written to --out-json when
given and, if docs/astdpd_explained.html exists, injected between the markers /*DATA:BEGIN*/
and /*DATA:END*/ as `window.DATA = <json>;`. The output is deterministic for fixed inputs
(meta.generated is the commit time of HEAD, not the wall clock). Self-checks (asserted):
  * the example user's FULL top-20 equals src.evaluate.rank(model.scorer()) for that user;
  * term-removal FULL and LightGCN test metrics equal results.json (seed 2024) to 1e-6;
  * the stored per-user test recall of the example user equals the recomputed one.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src import backbone as B  # noqa: E402
from src import data as D  # noqa: E402
from src.astdpd import tune as TU  # noqa: E402
from src.astdpd.model import ASTDPD, DEFAULTS as MODEL_DEFAULTS, popularity_features  # noqa: E402
from src.evaluate import evaluate, long_tail_mask, rank  # noqa: E402

SEED = 2024
SEEDS = [2024, 2025, 2026, 2027, 2028]
DATASETS = ["Coat", "Yahoo", "KuaiRec", "Gowalla"]
METHODS = ["LightGCN", "PPD", "TSP", "DAP", "ASTD-PD", "ASTD-PD (shipped)"]
SIG_METRICS = ["recall@20", "ndcg@20", "precision@20", "avgpop@20"]
STEP_METRICS = ["recall@20", "ndcg@20", "precision@20", "avgpop@20", "aplt@20", "coverage@20"]
HTML = ROOT / "docs" / "astdpd_explained.html"
CANONICAL = ROOT.parent / "RESULTS_CANONICAL.md"
RUNS = ROOT / "runs"

SCHEMA_NOTES = """\
DATA.meta: generated = commit time of HEAD (deterministic), git_commit = HEAD.
DATA.ds[<dataset>]:
 stats: counts from src.data.load. n_train_pairs.counted = rows of train.txt (duplicates counted,
  KuaiRec only), .unique = distinct pairs. long_tail_cut: evaluate.long_tail_mask (head = top 20%
  of items by counted degree, stable ties); head_min_degree / head_min_p = smallest degree / p of a
  head item. p = item popularity rank in [0,1] (average-tie rank of counted degree, model.py).
 config.selected/shipped: full config incl. model DEFAULTS; overrides = configs/astdpd_selected.yaml.
 features: degree_hist = counts over log2 bins [lo,hi] of item degree (bin 0 = degree 0);
  deciles[d] = {n_items, min_degree, max_degree, min_p, max_p} for decile = floor(10 p) clipped to 9
  (the sampler/stratification deciles; average-tie ranks make them unequal in size).
 gates: lambda_u over test users (hist: 20 equal bins over [lambda_min, lambda_max]); t_curve =
  (1-p)^phi at p = 0,0.05..1 (tail_reweight on); alpha/theta = gate outputs of the trained model.
  alpha and theta depend only on static node features, so they are identical at every layer
  (per_layer lists repeat the value); alpha_mean_all = model.last_mean_alpha after a forward.
 sampler: nominal mixture mass per popularity decile before rejection of the user's positives:
  uniform r_u*n_d/N + matched r_m*(share of training pairs whose positive is in decile d) +
  hard r_h*[d=9] (empty-decile fallbacks of sampler.py applied). uniform_sampler = n_d/N.
 example: rule in example.rule (medians, the user's deltas and n_tied_primary stored alongside). Scores are float32 values of the seed-2024 model, rounded to 4 dp.
  s_A = <u_A,i_A>/tau_A, fusion_term = lambda_u t_i <u_D,i_D>/tau_D, penalty_term = -mu p_i.
  Steps: A = s_A, F = s_A + fusion, FULL = F + penalty (= model.scorer()). Ranks are 1-based over the
  candidate set left after masking the user's training items (and, KuaiRec, items outside the test
  universe), stable ties. hit = item in the user's test set; longtail = evaluate.long_tail_mask.
  lightgcn_top20 scores = raw LightGCN inner products (backbone.scorer). Epochs: LightGCN 1-based;
  ASTD-PD history epochs and best_epoch are 0-based (as stored in astdpd.json).
 term_removal: evaluate() on the test split, same trained weights, terms removed at inference only.
 deciles: share of all top-20 slots (all test users) per decile (as features.deciles).
 curves: LightGCN valid recall@20 every 5 epochs; ASTD-PD per-epoch validation history.
 tune: from runs/<ds>/astdpd_tune.json; feasible = stage-2 mean valid AvgPop@20 < thr5.
 results/significance: copied from results/results.json / significance.json (test split).
 justifications: computed facts, see each block's 'note'.
DATA.ablation_prior: parsed verbatim from ../RESULTS_CANONICAL.md 'Mechanism ablation' (earlier harness).
Floats: 6 significant digits unless stated (scores 4 dp)."""


# ----------------------------------------------------------------------------------- helpers
def rnd(x, sig=6):
    """Recursively round floats to `sig` significant digits; numpy -> python; NaN/inf -> None."""
    if isinstance(x, dict):
        return {str(k): rnd(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, sig) for v in x]
    if isinstance(x, np.ndarray):
        return rnd(x.tolist(), sig)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if not math.isfinite(x):
            return None
        return 0.0 if x == 0 else float(f"{x:.{sig}g}")
    return x


def r4(x):
    return round(float(x), 4)


def jload(p):
    with open(p) as f:
        return json.load(f)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def full_config(cfg):
    out = json.loads(json.dumps(cfg))
    out["model"] = {**MODEL_DEFAULTS, **cfg["model"],
                    "latent_dim": cfg["latent_dim"], "n_layers": cfg["n_layers"], "decay": cfg["decay"]}
    return out


def load_astdpd(split, cfg, path, device):
    counts = (split.train_counts if split.train_counts is not None else split.train).tocsr()
    m = ASTDPD(counts, device, latent_dim=cfg["latent_dim"], n_layers=cfg["n_layers"],
               decay=cfg["decay"], **cfg["model"])
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval()


@torch.no_grad()
def cpu_embeddings(model):
    """model.embeddings() propagated on the CPU (CUDA sparse matmuls are not bitwise
    reproducible run to run), returned on the model's device -> deterministic output."""
    dev = model.item_pop_rank_tensor.device
    G, S = model.Graph, model.Simplex_Item_Graph
    try:
        model.to("cpu")
        model.Graph, model.Simplex_Item_Graph = G.cpu(), S.cpu()
        emb = model.embeddings()
    finally:
        model.Graph, model.Simplex_Item_Graph = G, S
        model.to(dev)
    return tuple(e.to(dev) for e in emb)


class Steps:
    """Score functions of the cumulative inference steps A, F, FULL of one ASTD-PD model.
    FULL reproduces model.scorer() operation for operation (propagation on the CPU, see cpu_embeddings)."""

    @torch.no_grad()
    def __init__(self, model):
        self.m, hp = model, model.hp
        self.uA, self.iA, self.uD, self.iD = cpu_embeddings(model)
        self.p = model.item_pop_rank_tensor.view(1, -1)
        self.t = model._tail(self.p) if hp["tail_reweight"] else torch.ones_like(self.p)
        self.tA, self.tD = max(1e-4, hp["temperature_acc"]), max(1e-4, hp["temperature_deb"])
        self.mu = hp["inference_pop_penalty"]

    @torch.no_grad()
    def parts(self, users):
        users = users.long().to(self.uA.device)
        s_A = self.uA[users] @ self.iA.t() / self.tA
        s_D = self.uD[users] @ self.iD.t() / self.tD
        fus = (self.m._lambda(users) * self.t) * s_D
        return s_A, fus

    def fn(self, step):
        def f(users):
            s_A, fus = self.parts(users)
            if step == "A":
                return s_A
            s = s_A + fus
            if step == "FULL" and self.mu > 0.0:
                s = s - self.mu * self.p
            return s
        return f


def masked_candidates(split, u):
    """Boolean [n_items]: rankable for user u on the test split (evaluate.rank's masking)."""
    ok = np.ones(split.n_items, dtype=bool)
    ok[split.train.indices[split.train.indptr[u]:split.train.indptr[u + 1]]] = False
    if split.test_candidates is not None:
        inside = np.zeros(split.n_items, dtype=bool)
        inside[np.asarray(split.test_candidates)] = True
        ok &= inside
    return ok


def ranks_of(score, ok):
    """1-based rank of every candidate item (stable on ties); non-candidates -> 0."""
    cand = np.where(ok)[0]
    order = cand[np.argsort(-score[cand], kind="stable")]
    r = np.zeros(len(score), dtype=np.int64)
    r[order] = np.arange(1, len(order) + 1)
    return r, order


def decile_share(tops, deciles):
    c = np.bincount(deciles[tops.reshape(-1)], minlength=10).astype(np.float64)
    return (c / c.sum()).tolist()


# ----------------------------------------------------------------------------------- blocks
def stats_block(split, feats, tail):
    deg = np.asarray(split.item_pop)
    head = ~tail
    p = feats["item_pop_rank"].numpy()
    test_users = [u for u, v in split.test.items() if len(v) > 0]
    return dict(
        n_users=split.n_users, n_items=split.n_items,
        n_train_pairs=dict(counted=int(split.train_counts.sum()), unique=int(split.train.nnz)),
        train_density=split.train.nnz / (split.n_users * split.n_items),
        n_valid=dict(users=sum(1 for v in split.valid.values() if len(v)),
                     pairs=int(sum(len(v) for v in split.valid.values()))),
        n_test_users=len(test_users), n_test_pairs=int(sum(len(split.test[u]) for u in test_users)),
        long_tail_cut=dict(head_fraction=0.2, n_head=int(head.sum()), n_tail=int(tail.sum()),
                           head_min_degree=int(deg[head].min()), head_min_p=float(p[head].min()),
                           tail_max_degree=int(deg[tail].max()), tail_max_p=float(p[tail].max())),
        zero_degree_items=int((deg == 0).sum()),
        candidates=dict(valid=None if split.valid_candidates is None else len(split.valid_candidates),
                        test=None if split.test_candidates is None else len(split.test_candidates)),
    )


def features_block(split, feats):
    deg = np.asarray(split.item_pop)
    p = feats["item_pop_rank"].numpy()
    dec = feats["item_decile"].numpy()
    top = int(np.ceil(np.log2(max(2, deg.max() + 1))))
    edges = [(0, 0)] + [(2 ** k, 2 ** (k + 1) - 1) for k in range(top + 1)]
    hist = [dict(lo=lo, hi=hi, n=int(((deg >= lo) & (deg <= hi)).sum())) for lo, hi in edges]
    while hist and hist[-1]["n"] == 0:
        hist.pop()
    deciles = []
    for d in range(10):
        s = dec == d
        deciles.append(dict(decile=d, n_items=int(s.sum()),
                            min_degree=int(deg[s].min()) if s.any() else None,
                            max_degree=int(deg[s].max()) if s.any() else None,
                            min_p=float(p[s].min()) if s.any() else None,
                            max_p=float(p[s].max()) if s.any() else None))
    return dict(
        definition="p_i = (average-tie rank of counted training degree - 1) / (n_items - 1); decile = min(9, floor(10 p_i))",
        degree_hist=hist, deciles=deciles,
        degree_quantiles={str(q): float(np.quantile(deg, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)},
        user_features="user pop = mean p of the user's training items; density = rank of sum of their degrees; diversity = #distinct item deciles / 10",
    )


@torch.no_grad()
def gates_block(model, test_users):
    hp = model.hp
    lo, hi = hp["fuse_lambda_min"], hp["fuse_lambda_max"]
    lam = model._lambda(torch.as_tensor(test_users, device=model.item_pop_rank_tensor.device)).view(-1).cpu().numpy()
    counts, edges = np.histogram(lam, bins=20, range=(lo, hi) if hi > lo else None)
    p = np.linspace(0, 1, 21)
    t = np.clip(1 - p, 0, 1) ** hp["tail_phi"]
    # alpha / theta exactly as layer_embeddings computes them (layer-invariant)
    feat_3 = torch.cat([model.node_popularity, model.node_density, model.node_type], dim=1)
    feat_2 = torch.cat([model.node_popularity, model.node_density], dim=1)
    pp, e = model.node_popularity, hp["alpha_pop_exponent"]
    rho = pp ** e / (pp ** e + 0.5 ** e + 1e-8)
    user_scale = model.node_type * hp["user_alpha_scale"] + (1.0 - model.node_type)
    alpha = (model.effective_alpha_max * model.gate_mlp_alpha(feat_2) * rho * user_scale).view(-1)
    theta = (model.gate_mlp_theta(feat_3) if hp["topo_gate_learnable"] else
             torch.full((feat_3.shape[0], 1), hp["topo_gate_init"], device=feat_3.device)).view(-1)
    gamma = (hp["highfreq_gamma"] * model.gate_mlp_gamma(feat_2)).view(-1)
    model.layer_embeddings()
    nu = model.num_users
    assert abs(model.last_mean_alpha - alpha.mean().item()) < 1e-6
    L = hp["n_layers"]
    return dict(
        lambda_u=dict(min_cfg=lo, max_cfg=hi, mean=float(lam.mean()), std=float(lam.std()),
                      min=float(lam.min()), max=float(lam.max()), mean_over_max=float(lam.mean() / hi) if hi else None,
                      hist_counts=counts.tolist(), hist_edges=edges.tolist(), n_users=len(lam)),
        tail_phi=hp["tail_phi"], tail_reweight=hp["tail_reweight"],
        t_curve=dict(p=p.tolist(), t=t.tolist()),
        effective_alpha_max=model.effective_alpha_max, alpha_base=hp["alpha_base"],
        alpha_mean_all=model.last_mean_alpha, theta_mean_all=model.last_mean_theta,
        alpha_mean_items=alpha[nu:].mean().item(), alpha_mean_users=alpha[:nu].mean().item(),
        alpha_mean_over_max=model.last_mean_alpha / model.effective_alpha_max if model.effective_alpha_max else None,
        theta_mean_items=theta[nu:].mean().item(), theta_mean_users=theta[:nu].mean().item(),
        gamma_mean_all=gamma.mean().item(),
        alpha_per_layer=[model.last_mean_alpha] * L, theta_per_layer=[model.last_mean_theta] * L,
        n_layers=L, basis_rank=hp["pop_basis_rank"],
        inference_pop_penalty=hp["inference_pop_penalty"],
    )


def sampler_block(cfg, feats, split):
    s = cfg["sampler"]
    tot = s["uniform_ratio"] + s["matched_ratio"] + s["hard_ratio"]
    ru, rm, rh = s["uniform_ratio"] / tot, s["matched_ratio"] / tot, s["hard_ratio"] / tot
    dec = feats["item_decile"].numpy()
    N = len(dec)
    n_d = np.bincount(dec, minlength=10).astype(np.float64)
    counts = split.train_counts.tocoo()
    pos_w = np.bincount(dec[counts.col], weights=counts.data, minlength=10)
    pos_w = pos_w / pos_w.sum()
    unif = n_d / N
    mass = ru * unif
    for d in range(10):   # matched draws: within the positive's decile (whole catalogue if empty)
        mass = mass + rm * pos_w[d] * (np.eye(10)[d] if n_d[d] > 0 else unif)
    mass = mass + rh * (np.eye(10)[9] if n_d[9] > 0 else unif)
    return dict(ratios=dict(uniform=ru, matched=rm, hard=rh),
                q_mass_per_decile=mass.tolist(), uniform_sampler_mass=unif.tolist(),
                positive_mass_per_decile=pos_w.tolist(), decile_sizes=n_d.astype(int).tolist(),
                note="nominal mixture before rejecting the user's own positives")


def curves_block(ds, lgcn_ckpt, ajs, shipped):
    def hist(j):
        keys = ("epoch", "hv", "recall@20", "ndcg@20", "precision@20", "avgpop@20", "loss", "lambda_fuse", "alpha", "theta")
        h = j.get("history") or []
        ep = j["config"]["epochs"]
        return dict(history={k: [x.get(k) for x in h] for k in keys}, best_epoch=j["best_epoch"],
                    epochs_run=j["epochs_run"], epoch_cap=ep, warmup=min(10, ep // 2),
                    patience=j["config"]["patience"], best_valid_hypervolume=j["best_valid_hypervolume"])
    vc = lgcn_ckpt["valid_curve"]
    return dict(lightgcn=dict(epoch=[e for e, _ in vc], recall20=[r for _, r in vc],
                              best_epoch=lgcn_ckpt["best_epoch"], epochs_run=vc[-1][0],
                              eval_every=lgcn_ckpt["config"]["eval_every"], patience_checks=lgcn_ckpt["config"]["patience"],
                              epoch_cap=lgcn_ckpt["config"]["epochs"]),
                astdpd=hist(ajs), astdpd_shipped=hist(shipped))


def mu_grid_of(budget):
    """The stage-1 mu grid (Coat's older log stores only its length, = the default MU_GRID)."""
    if "incumbent_mu_grid" in budget:
        return budget["incumbent_mu_grid"]
    assert budget["incumbent_mu_points"] == len(TU.MU_GRID)
    return list(TU.MU_GRID)


def tune_block(ds):
    t = jload(RUNS / ds / "astdpd_tune.json")
    s = t["selection"]
    st2 = s["stage2"]
    thr = s["threshold_all_seeds"]
    cand = st2["candidates"]
    trials = t["trials"]
    by_stage = {}
    for x in trials:
        by_stage.setdefault(x["stage"], []).append(x["wall_s"])
    sel = yaml.safe_load(open(ROOT / "configs" / "astdpd_selected.yaml"))[ds]
    return dict(
        rule=s["rule"], threshold_all_seeds=thr, thr3_info=st2.get("threshold_confirm_seeds_info"),
        stage1_threshold_seed2024=s["stage1"]["threshold"],
        n_configs=dict(stage1=s["stage1"]["n_configs"], stage2=len(cand), final=1),
        n_trials=dict(total=len(trials), **{k: len(v) for k, v in by_stage.items()}),
        stage2_candidates={c: dict(ndcg20=m["ndcg@20"], avgpop20=m["avgpop@20"], recall20=m["recall@20"],
                                   feasible=m["avgpop@20"] < thr) for c, m in cand.items()},
        feasible_stage2=sum(m["avgpop@20"] < thr for m in cand.values()),
        winner=s["winner"], winner_overrides=s["winner_overrides"], winner_valid_mean=s["winner_valid_mean"],
        winner_feasible=st2["feasible"], config_id_installed=sel["config_id"],
        budget=dict(n_random=s["budget"]["n_random"], mu_grid=mu_grid_of(s["budget"]),
                    n_mu=len(mu_grid_of(s["budget"])), top_k=s["budget"]["top_k"],
                    confirm_seeds=s["budget"]["confirm_seeds"]),
        mean_wall_s_per_trial=float(np.mean([x["wall_s"] for x in trials])),
        mean_wall_s_by_stage={k: float(np.mean(v)) for k, v in by_stage.items()},
        total_trial_hours=float(sum(x["wall_s"] for x in trials) / 3600),
        baselines_valid_avgpop20_per_seed=s["baselines"]["per_seed_baselines"],
        per_seed_min=s["baselines"]["per_seed_min"], per_seed_argmin=s["baselines"]["per_seed_argmin"],
        search_space=s["search_space"], reselected=s.get("reselected"), reinstall=s.get("reinstall"),
    )


def results_block(R, ds):
    out = {}
    for m in METHODS:
        r = R[ds][m]
        out[m] = dict(n_seeds=r["n_seeds"], seeds=r["seeds"],
                      mean={k: v for k, v in r["mean"].items() if k.endswith(("@10", "@20"))},
                      std={k: v for k, v in r["std"].items() if k.endswith(("@10", "@20"))},
                      best_epoch=r.get("best_epoch"), selected_hparams=r.get("selected_hparams"),
                      wall_time_s=r.get("wall_time_s"))
    out["popularity_race"] = R[ds]["popularity_race"]
    return out


def significance_block(S, ds):
    out = {}
    for comp, v in S[ds].items():
        out[comp] = {lvl: {m: {k: v[lvl][m].get(k) for k in
                               ("mean_diff", "p_holm", "p_value", "n_better", "n_worse", "n_tied", "warning")
                               if k in v[lvl][m]} for m in SIG_METRICS if m in v[lvl]}
                     for lvl in ("across_seed", "per_user")}
    return out


def ablation_prior():
    txt = CANONICAL.read_text()
    m = re.search(r"^## Mechanism ablation.*?$(.*?)^---", txt, re.S | re.M)
    sec = m.group(1)
    head = re.search(r"\*(Generated.*?)\*", sec, re.S).group(1).replace("\n", " ")
    arms = []
    for line in sec.splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 6 and c[0].startswith("("):
            def pm(s):
                a = s.split("±")
                return dict(mean=float(a[0]), std=float(a[1]) if len(a) > 1 else None)
            arms.append(dict(arm=c[0], recall20=pm(c[1]), avgpop20=pm(c[2]), aplt20=pm(c[3]),
                             alpha=float(c[4]), theta=float(c[5])))
    reading = [re.sub(r"\s+", " ", b).strip().replace("**", "")
               for b in re.findall(r"^- (.*?)(?=^- |\Z)", sec.split("**Reading:**")[1], re.S | re.M)]
    return dict(source="../RESULTS_CANONICAL.md, section 'Mechanism ablation — what actually produces the effect' "
                       "(earlier harness, PPD v6 = ASTD-PD shipped config; Gowalla, 3 seeds)",
                header=head.strip(), dataset="Gowalla", arms=arms, reading=reading)


# ----------------------------------------------------------------------------------- dataset
def build_dataset(ds, R, S, device, do_tsp, checks):
    t0 = time.time()
    run = RUNS / ds / f"seed{SEED}"
    print(f"[{ds}] loading", flush=True)
    net, split = B.load_backbone(ds, SEED, device)
    lgcn_ckpt = torch.load(run / "lightgcn.pt", map_location="cpu", weights_only=False)
    ajs, sjs = jload(run / "astdpd.json"), jload(run / "astdpd_shipped.json")
    model = load_astdpd(split, ajs["config"], run / "astdpd.pt", device)
    feats = popularity_features(split.train_counts)
    deciles = feats["item_decile"].numpy()
    p_np = feats["item_pop_rank"].numpy()
    assert np.allclose(p_np, model.item_pop_rank_tensor.cpu().numpy())
    deg = np.asarray(split.item_pop)
    tail = long_tail_mask(deg)
    test_users = np.array(sorted(u for u, v in split.test.items() if len(v) > 0), dtype=np.int64)

    out = dict(stats=stats_block(split, feats, tail))
    sel = yaml.safe_load(open(ROOT / "configs" / "astdpd_selected.yaml")).get(ds, {})
    out["config"] = dict(selected=full_config(ajs["config"]), shipped=full_config(sjs["config"]),
                         overrides=sel.get("overrides"), config_id=sel.get("config_id"))
    out["features"] = features_block(split, feats)
    out["gates"] = gates_block(model, test_users)
    out["sampler"] = dict(selected=sampler_block(ajs["config"], feats, split),
                          shipped=sampler_block(sjs["config"], feats, split))

    # ---- term removal + LightGCN reference (evaluate on test, all test users) -------------
    steps = Steps(model)
    tr = {}
    tops = {}
    for st in ("A", "F", "FULL"):
        ev = evaluate(steps.fn(st), split, "test", ks=(20,), device=device)
        tr[st] = {k: ev[k] for k in STEP_METRICS}
        tops[st] = rank(steps.fn(st), split, "test", 20, device=device)[2]
    lg_fn = B.scorer(net)
    ev = evaluate(lg_fn, split, "test", ks=(20,), device=device)
    tr["LightGCN"] = {k: ev[k] for k in STEP_METRICS}
    u_lg, _, tops["LightGCN"], hits_lg = rank(lg_fn, split, "test", 20, device=device)
    assert (u_lg == test_users).all()
    tr["label"] = "inference-time term removal, same trained seed-2024 ASTD-PD weights; all test users; evaluate.py masking"
    tr["steps"] = dict(A="s_A only", F="s_A + lambda_u t_i s_D", FULL="s_A + lambda_u t_i s_D - mu p_i (deployed)")
    for name, key in (("FULL", "ASTD-PD"), ("LightGCN", "LightGCN")):
        ref = R[ds][key]["per_seed_test"][str(SEED)]
        for k in ("recall@20", "avgpop@20", "ndcg@20", "coverage@20"):
            d = abs(tr[name][k] - ref[k])
            checks.append(dict(dataset=ds, check=f"term_removal {name} {k} vs results.json seed {SEED}",
                               ours=tr[name][k], ref=ref[k], abs_diff=d, ok=d < 1e-6))
            assert d < 1e-6 * max(1.0, abs(ref[k])), (ds, name, k, tr[name][k], ref[k])
    out["term_removal"] = tr

    # ---- deciles ---------------------------------------------------------------------------
    shipped = load_astdpd(split, sjs["config"], run / "astdpd_shipped.pt", device)
    tops["shipped"] = rank(shipped.scorer(), split, "test", 20, device=device)[2]
    dec = dict(LightGCN=decile_share(tops["LightGCN"], deciles), **{"ASTD-PD": decile_share(tops["FULL"], deciles)},
               **{"ASTD-PD (shipped)": decile_share(tops["shipped"], deciles)},
               **{"ASTD-PD step A": decile_share(tops["A"], deciles), "ASTD-PD step F": decile_share(tops["F"], deciles)})
    dec_note = "share of top-20 slots per popularity decile, all test users, seed 2024"
    if do_tsp:
        from src.methods import tsp as TSPm
        hp = jload(run / "tsp_selected.json")["selected_hparams"]
        t1 = time.time()
        fn = TSPm.build_scorer(net, split, device, **hp)
        ttsp = rank(fn, split, "test", 20, device=device)[2]
        dec["TSP"] = decile_share(ttsp, deciles)
        ref = jload(run / "tsp_selected.json")["test"]["avgpop@20"]
        ours = float(deg[ttsp].mean())
        checks.append(dict(dataset=ds, check="TSP recomputed test avgpop@20 vs tsp_selected.json",
                           ours=ours, ref=ref, abs_diff=abs(ours - ref), ok=abs(ours - ref) < 1e-3 * ref))
        print(f"[{ds}] TSP rebuilt in {time.time() - t1:.0f}s", flush=True)
        dec_note += f"; TSP rebuilt from the seed-2024 backbone with its selected hparams {hp}"
        del fn
    else:
        dec_note += "; TSP skipped (--no-tsp)"
    out["deciles"] = dict(note=dec_note, shares=dec, decile_sizes=np.bincount(deciles, minlength=10).tolist(),
                          catalogue_share=(np.bincount(deciles, minlength=10) / len(deciles)).tolist())
    del shipped

    # ---- example user ------------------------------------------------------------------------
    za = np.load(run / "astdpd_per_user_test.npz")
    zl = np.load(run / "lightgcn_per_user_test.npz")
    assert (za["users"] == zl["users"]).all() and (za["users"] == test_users).all()
    delta = za["recall@20"] - zl["recall@20"]
    dpop = za["avgpop@20"] - zl["avgpop@20"]
    med, med_pop = float(np.median(delta)), float(np.median(dpop))
    elig = (za["recall@20"] > 0) | (zl["recall@20"] > 0)
    dist = np.abs(delta - med)
    dist_pop = np.abs(dpop - med_pop)
    dist[~elig] = np.inf
    i = int(np.lexsort((test_users, dist_pop, dist))[0])   # last key is primary
    uid = int(test_users[i])
    sel = dict(median_delta_recall20=med, median_delta_avgpop20=med_pop,
               delta_recall20=float(delta[i]), delta_avgpop20=float(dpop[i]),
               n_tied_primary=int((dist == dist[i]).sum()),
               n_eligible=int(elig.sum()), n_test_users=len(test_users),
               share_delta_recall_zero=float(np.mean(delta == 0)))
    out["example"] = example_block(ds, split, model, steps, net, uid, deg, p_np, tail, feats,
                                   tops, test_users, sel, za, zl, i, checks, device)
    out["curves"] = curves_block(ds, lgcn_ckpt, ajs, sjs)
    out["tune"] = tune_block(ds)
    out["results"] = results_block(R, ds)
    out["significance"] = significance_block(S, ds)
    out["justifications"] = justifications_block(ds, split, R, lgcn_ckpt)
    out["runtime_s"] = time.time() - t0
    print(f"[{ds}] done in {time.time() - t0:.0f}s; example user {uid} dR {delta[i]:+.4f} (median {med:+.4f}) "
          f"dPop {dpop[i]:+.3f} (median {med_pop:+.3f})", flush=True)
    del model, net, steps
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def example_block(ds, split, model, steps, net, uid, deg, p_np, tail, feats, tops, test_users,
                  sel, za, zl, row, checks, device):
    u = torch.tensor([uid], device=device)
    s_A, fus = steps.parts(u)
    F = s_A + fus
    FULL = F - steps.mu * steps.p if steps.mu > 0.0 else F
    pen = (-steps.mu * steps.p).expand_as(F) if steps.mu > 0.0 else torch.zeros_like(F)
    s_A, fus, F, FULL, pen = (x[0].float().cpu().numpy() for x in (s_A, fus, F, FULL, pen))
    ok = masked_candidates(split, uid)
    truth = set(np.asarray(split.test[uid]).tolist())
    n_true = len(split.test[uid])
    rk, orders = {}, {}
    for name, sc in (("A", s_A), ("F", F), ("FULL", FULL)):
        rk[name], orders[name] = ranks_of(sc, ok)
    # self-check: FULL top-20 vs evaluate.rank with the model's own scorer (independent)
    users_r, _, top_r, _ = rank(model.scorer(), split, "test", 20, device=device)
    ref_top = top_r[row].tolist()
    assert users_r[row] == uid
    mine = orders["FULL"][:20].tolist()
    checks.append(dict(dataset=ds, check=f"example user {uid} FULL top-20 == evaluate.rank(model.scorer())",
                       ok=mine == ref_top, same_set=set(mine) == set(ref_top)))
    assert mine == ref_top, (ds, uid, mine, ref_top)
    assert mine == tops["FULL"][row].tolist()
    for st in ("A", "F"):
        assert orders[st][:20].tolist() == tops[st][row].tolist(), (ds, uid, st)

    lg = B.scorer(net)(u)[0].float().cpu().numpy()
    lrk, lorder = ranks_of(lg, ok)
    assert lorder[:20].tolist() == tops["LightGCN"][row].tolist()

    def summary(order):
        t20 = order[:20]
        h = sum(int(x) in truth for x in t20)
        return dict(avgpop20=float(deg[t20].mean()), recall20=h / n_true, hits=h, n_longtail=int(tail[t20].sum()))
    summ = {st: summary(orders[st]) for st in ("A", "F", "FULL")}
    summ["LightGCN"] = summary(lorder)
    assert abs(summ["FULL"]["recall20"] - za["recall@20"][row]) < 1e-9
    assert abs(summ["LightGCN"]["recall20"] - zl["recall@20"][row]) < 1e-9
    checks.append(dict(dataset=ds, check=f"example user {uid} recall@20 vs stored per-user npz", ok=True))

    def item(i):
        i = int(i)
        return dict(item=i, degree=int(deg[i]), p=float(p_np[i]))
    union = []
    for st in ("A", "F", "FULL"):
        for i in orders[st][:20]:
            if int(i) not in union:
                union.append(int(i))
    union.sort(key=lambda i: rk["FULL"][i])
    hist_items = split.train.indices[split.train.indptr[uid]:split.train.indptr[uid + 1]]
    lam = model._lambda(u).item()
    return dict(
        rule=("eligible = test users with >= 1 test hit (recall@20 > 0) under LightGCN or ASTD-PD (seed-2024 "
              "per-user npz). Primary: minimal |dRecall@20 - median dRecall@20|; tie-break 1: minimal "
              "|dAvgPop@20 - median dAvgPop@20|; tie-break 2: lowest user id. d = ASTD-PD minus LightGCN per user; "
              "both medians over ALL test users"),
        **sel,
        user_id=uid,
        user_features=dict(pop=model.node_popularity[uid].item(), density=model.node_density[uid].item(),
                           diversity=model.user_diversity[uid].item(), lambda_u=lam,
                           n_train=int(len(hist_items)), n_test=n_true),
        history=sorted([item(i) for i in hist_items], key=lambda d: -d["degree"]),
        test_items=sorted([item(i) for i in truth], key=lambda d: -d["degree"]),
        n_candidates=int(ok.sum()), mu=steps.mu, t_phi=model.hp["tail_phi"],
        lightgcn_top20=[dict(**item(i), rank=r + 1, score=r4(lg[i]), hit=int(i) in truth, longtail=bool(tail[i]))
                        for r, i in enumerate(lorder[:20])],
        astdpd=[dict(**item(i), hit=i in truth, longtail=bool(tail[i]), t_i=float((1 - p_np[i]) ** model.hp["tail_phi"]),
                     s_A=r4(s_A[i]), fusion_term=r4(fus[i]), penalty_term=r4(pen[i]), total=r4(FULL[i]),
                     rank_A=int(rk["A"][i]), rank_F=int(rk["F"][i]), rank_FULL=int(rk["FULL"][i]),
                     rank_LightGCN=int(lrk[i])) for i in union],
        summary=summ,
    )


def justifications_block(ds, split, R, lgcn_ckpt):
    j = {}
    # train/test overlap: test pairs whose item is masked as a training item of the same user
    n_pairs = n_un = 0
    for u, items in split.test.items():
        tr = set(split.train.indices[split.train.indptr[u]:split.train.indptr[u + 1]].tolist())
        n_pairs += len(items)
        n_un += sum(int(i) in tr for i in items)
    j["unreachable_test_pairs"] = dict(count=n_un, share=n_un / max(1, n_pairs), n_test_pairs=n_pairs,
                                       note="test pairs whose item is in the user's training set: masked at evaluation, can never be hit")
    if split.test_candidates is not None:
        pass
    lg_json = {s: jload(RUNS / ds / f"seed{s}" / "lightgcn.json") for s in SEEDS}
    a_json = {s: jload(RUNS / ds / f"seed{s}" / "astdpd.json") for s in SEEDS}
    s_json = {s: jload(RUNS / ds / f"seed{s}" / "astdpd_shipped.json") for s in SEEDS}
    vc = lgcn_ckpt["valid_curve"]
    rec = [r for _, r in vc]
    k = int(np.argmax(rec))
    j["lightgcn_epochs"] = dict(
        best_epoch={s: lg_json[s]["best_epoch"] for s in SEEDS},
        epochs_trained={s: lg_json[s]["epochs_run"] for s in SEEDS}, cap=lg_json[SEED]["config"]["epochs"],
        seed2024_curve=dict(peak_epoch=vc[k][0], peak_recall20=rec[k], last_epoch=vc[-1][0], last_recall20=rec[-1],
                            first_recall20=rec[0], declines_after_peak=rec[-1] < rec[k],
                            drop_peak_to_last=rec[k] - rec[-1]),
        note="LightGCN: valid Recall@20 every 5 epochs, stop after 10 checks without improvement, cap 1000")
    old_cap = a_json[SEED]["config"]["epochs"]
    j["astdpd_epochs"] = dict(
        cap=old_cap,
        selected={s: dict(best_epoch=a_json[s]["best_epoch"], epochs_run=a_json[s]["epochs_run"],
                          hit_cap=a_json[s]["epochs_run"] >= a_json[s]["config"]["epochs"]) for s in SEEDS},
        shipped={s: dict(best_epoch=s_json[s]["best_epoch"], epochs_run=s_json[s]["epochs_run"],
                         hit_cap=s_json[s]["epochs_run"] >= s_json[s]["config"]["epochs"]) for s in SEEDS},
        n_selected_hit_cap=sum(a_json[s]["epochs_run"] >= a_json[s]["config"]["epochs"] for s in SEEDS),
        n_shipped_hit_cap=sum(s_json[s]["epochs_run"] >= s_json[s]["config"]["epochs"] for s in SEEDS),
        lightgcn_best_beyond_astdpd_cap=sum(lg_json[s]["best_epoch"] > old_cap for s in SEEDS),
        note="ASTD-PD epochs are 0-based (best_epoch) / counts (epochs_run); cap = the old harness budget "
             "(50 Coat, 100 otherwise). lightgcn_best_beyond_astdpd_cap = seeds whose LightGCN best epoch "
             "exceeds that cap (the old harness cut LightGCN at the same budget)")
    meanR = lambda m, k: R[ds][m]["mean"][k]  # noqa: E731
    j["coverage20"] = {m: meanR(m, "coverage@20") for m in METHODS}
    j["coverage20"]["astdpd_minus_lightgcn"] = meanR("ASTD-PD", "coverage@20") - meanR("LightGCN", "coverage@20")
    j["coverage20"]["astdpd_highest"] = max(METHODS[:5], key=lambda m: meanR(m, "coverage@20")) == "ASTD-PD"
    j["ppd_vs_lightgcn_avgpop20"] = dict(
        test_PPD=meanR("PPD", "avgpop@20"), test_LightGCN=meanR("LightGCN", "avgpop@20"),
        ppd_higher=meanR("PPD", "avgpop@20") > meanR("LightGCN", "avgpop@20"),
        rel_change=meanR("PPD", "avgpop@20") / meanR("LightGCN", "avgpop@20") - 1,
        valid_per_seed={s: dict(PPD=jload(RUNS / ds / f"seed{s}" / "ppd_selected.json")["valid"]["avgpop@20"],
                                LightGCN=lg_json[s]["valid"]["avgpop@20"]) for s in SEEDS},
        ppd_selected=R[ds]["PPD"]["selected_hparams"])
    t = jload(RUNS / ds / "astdpd_tune.json")["selection"]
    j["tuning_budget"] = dict(n_random=t["budget"]["n_random"], n_mu=len(mu_grid_of(t["budget"])),
                              top_k=t["budget"]["top_k"], n_trials=t["budget"]["n_trials"])
    return j


def shared_overrides_block():
    """Do the Yahoo and KuaiRec winners share overrides, and why (tune.py's rng_seed=0)."""
    out = {}
    for ds in ("Yahoo", "KuaiRec", "Coat", "Gowalla"):
        t = jload(RUNS / ds / "astdpd_tune.json")
        s = t["selection"]
        base = TU.shipped_config(ds)
        ovs = TU.stage1_overrides(base, s["budget"]["n_random"], 0, mu_grid_of(s["budget"]))
        n_mu = len(mu_grid_of(s["budget"]))
        idx = [k for k, ov in enumerate(ovs) if TU.config_id(TU.make_config(base, ov)) == s["winner"]]
        scale = 0.2 / max(1e-4, base["model"].get("temperature_acc", 0.1))
        out[ds] = dict(winner=s["winner"], overrides=s["winner_overrides"],
                       stage1_position=idx[0] if idx else None,
                       random_draw_index=(idx[0] - n_mu) if idx and idx[0] >= n_mu else None,
                       mu_scale=scale, temperature_acc=base["model"].get("temperature_acc", 0.1))
    y, k = out["Yahoo"]["overrides"], out["KuaiRec"]["overrides"]
    same_except_mu = {a: b for a, b in y.items() if a != "inference_pop_penalty"} == \
                     {a: b for a, b in k.items() if a != "inference_pop_penalty"}
    mu_equiv = abs(y["inference_pop_penalty"] / out["Yahoo"]["mu_scale"]
                   - k["inference_pop_penalty"] / out["KuaiRec"]["mu_scale"]) < 1e-9
    return dict(per_dataset=out, yahoo_kuairec_identical_except_mu=same_except_mu,
                yahoo_kuairec_same_draw=out["Yahoo"]["random_draw_index"] == out["KuaiRec"]["random_draw_index"],
                mu_equivalent_after_temperature_scaling=mu_equiv,
                note="src/astdpd/tune.py draws stage-1 random configurations with np.random.default_rng(rng_seed=0) "
                     "for every dataset, so draw #k is the same override set everywhere; mu is rescaled by "
                     "0.2/temperature_acc (Yahoo's shipped temperature is the 0.1 default -> x2, KuaiRec 0.2 -> x1)")


# ----------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-tsp", action="store_true", help="skip rebuilding TSP for the decile panel")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    a = ap.parse_args()
    t0 = time.time()
    R, S = jload(ROOT / "results" / "results.json"), jload(ROOT / "results" / "significance.json")
    checks = []
    DATA = dict(meta=dict(generated=git("log", "-1", "--format=%cI"), seed=SEED, datasets=a.datasets,
                          git_commit=git("rev-parse", "HEAD"), results_generated=R["meta"]["generated"],
                          seeds_results=R["meta"]["seeds"], selection=R["meta"]["selection"]),
                schema_notes=SCHEMA_NOTES, ds={})
    for ds in a.datasets:
        DATA["ds"][ds] = build_dataset(ds, R, S, a.device, not a.no_tsp, checks)
    DATA["shared_overrides"] = shared_overrides_block()
    DATA["ablation_prior"] = ablation_prior()
    DATA["self_checks"] = checks
    runtimes = {ds: DATA["ds"][ds].pop("runtime_s") for ds in a.datasets}
    DATA = rnd(DATA)
    blob = json.dumps(DATA, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if a.out_json:
        Path(a.out_json).write_text(blob)
        print(f"wrote {a.out_json} ({len(blob) / 1024:.0f} KiB)")
    if HTML.exists():
        html = HTML.read_text()
        pat = re.compile(r"/\*DATA:BEGIN\*/.*?/\*DATA:END\*/", re.S)
        if pat.search(html):
            safe = blob.replace("</", "<\\/")
            HTML.write_text(pat.sub(lambda _: f"/*DATA:BEGIN*/window.DATA = {safe};/*DATA:END*/", html, count=1))
            print(f"injected DATA into {HTML}")
        else:
            print(f"{HTML} has no /*DATA:BEGIN*/ ... /*DATA:END*/ markers; not modified")
    print(f"self-checks: {sum(bool(c['ok']) for c in checks)}/{len(checks)} ok; "
          f"runtime {time.time() - t0:.0f}s {({k: round(v) for k, v in runtimes.items()})}; JSON {len(blob) / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
