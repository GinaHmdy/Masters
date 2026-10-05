"""PPD (Post-hoc Popularity Debiasing, edgeslab/Posthoc_Popularity_Debiasing) on the trained backbone.

What the official LightGCN/code/main.py does after training (model_name 'lgn'):
    compute_pop_bias_scores("../../data/<ds>", Recmodel.dataset._allPos, Recmodel, Recmodel.dataset,
                            best_epoch, VARIANT_VECTOR_BIAS_ZERO_LAYER_UPDATE, beta, phi)
and for that variant compute_pop_bias_scores (utilities.py) runs, in this order:
    U, I  = normalize(embedding_user.weight), normalize(embedding_item.weight)   # layer-0, L2
    p     = compute_global_popularity_score(U, I, 'dot', sample_size=-1)
    r     = normalize_scores_global(compute_personalized_preference_scores(I, allPos, p, 'dot', beta))
    b     = normalize_scores_global(compute_bias_scores(p, r, allPos))
    d_u, d_i = compute_d_pop_all_normalized(U, I, n_users, n_items, b, allPos, phi)
    embedding_user.weight <- project_and_remove(U, d_u); embedding_item.weight <- project_and_remove(I, d_i)
    Procedure.Test(...)  -> Recmodel.getUsersRating = sigmoid(LightGCN propagation of the new layer-0)
`_official_debiased` calls exactly those official functions in that order. We do not call
compute_pop_bias_scores itself only because its tail calls the PPD repo's Procedure.Test (own
evaluator; writes ../../data/<ds>/topk_predict_ppd.txt into the repo); the steps before it are
reproduced call-for-call. The PPD model.LightGCN.computer() and its dataloader.getSparseGraph
('lgn' variant) are identical to LightGCN-PyTorch's (diffed), so scores are formed by the
backbone's own official computer() with the replaced layer-0 weights (weights restored afterwards).
Scores are raw inner products (same ranking as the official sigmoid, minus fp32 saturation ties).

Import-time workarounds (no vendored file is changed):
  * utilities.py does `from register import dataset`; the official register.py builds a Loader from
    ../../data/<ds> (cwd-relative, data not vendored). We put a stub `register` module with
    dataset=None in sys.modules for the import; `dataset` is only used inside Procedure.Test calls,
    which we never make. The algorithm is unaffected.
  * world.py parses sys.argv -> argv is set to `--dataset <ds> --model lgn` for the import only.
  * cppimport is blocked (utils.py would otherwise compile into the vendored tree); unused here.
  * matplotlib uses the Agg backend (utilities imports pyplot; plotting code is never called).

Only the official pure-Python loops are used, for every dataset. They cost O(n_users*n_items)
Python iterations for p (Gowalla: 1.2e9, ~65 min on one core) plus O(sum_u |N_u|^2) per beta for r
(KuaiRec: ~90 min per beta) and run on CPU tensors (one thread suffices: every op is a scalar or a
64/256-d dot product). Since p depends only on the backbone and b only on (backbone, beta), both
official outputs are cached on disk when `CACHE_DIR` is set (torch.save of the exact
objects the official functions returned), so a grid can be split over processes: one computes p,
then one process per beta computes b and evaluates both phi values (see run.py).
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import types

import torch
import torch.nn.functional as F

from . import THIRD, isolated_modules, main

PPD_CODE = THIRD / "PPD" / "LightGCN" / "code"
DEFAULTS = {"beta": 0.2, "phi": 1.0}  # official parse.py defaults
_PPD_NAMES = ("world", "parse", "utils", "dataloader", "model", "Procedure", "register",
              "constants", "utilities", "DAP", "cppimport")
_ppd = None
_cache: dict = {}
last_timing: dict = {}   # seconds spent computing p / b(beta) in this process


def official_modules(dataset: str = "Coat"):
    """(utilities, constants) of the PPD repo, imported in isolation."""
    global _ppd
    if _ppd is None:
        os.environ.setdefault("MPLBACKEND", "Agg")
        stub = types.ModuleType("register")
        stub.dataset = None  # see module docstring
        argv, sys.argv = sys.argv, ["main.py", "--dataset", "Coat", "--model", "lgn"]
        try:
            with isolated_modules(PPD_CODE, _PPD_NAMES, stubs={"register": stub, "cppimport": None}):
                import contextlib
                with contextlib.redirect_stdout(sys.stderr):
                    import utilities, constants  # noqa: E401
        finally:
            sys.argv = argv
        _ppd = (utilities, constants)
    return _ppd


CACHE_DIR = None   # Path: on-disk cache of p and b(beta) for the current backbone (set by run.py)


def _cached(name, compute):
    """In-memory, then on-disk (CACHE_DIR/<name>.pt) memo of an official intermediate result."""
    if name in _cache:
        return _cache[name]
    path = CACHE_DIR / f"{name}.pt" if CACHE_DIR is not None else None
    if path is not None and path.exists():
        val = torch.load(path, weights_only=False)
    else:
        t = time.time()
        val = compute()
        last_timing[name] = time.time() - t
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(val, str(path) + ".tmp")
            os.replace(str(path) + ".tmp", path)
    _cache[name] = val
    return val


def popularity(U, I, key):
    ut, _ = official_modules()
    return _cached(f"{key}_p", lambda: ut.compute_global_popularity_score(U, I, "dot", sample_size=-1))


def bias(U, I, all_pos, beta, key):
    ut, _ = official_modules()
    p = popularity(U, I, key)

    def compute():
        r = ut.compute_personalized_preference_scores(I, all_pos, p, "dot", beta)
        r = ut.normalize_scores_global(r)
        return ut.normalize_scores_global(ut.compute_bias_scores(p, r, all_pos))
    return _cached(f"{key}_b_beta{beta:g}", compute)


def _inputs(model):
    with torch.no_grad():
        U = F.normalize(model.embedding_user.weight.detach().float().cpu())
        I = F.normalize(model.embedding_item.weight.detach().float().cpu())
    # cache key: content of the backbone's layer-0 weights (a new backbone never reuses a cache)
    h = hashlib.sha256(U.numpy().tobytes() + I.numpy().tobytes()).hexdigest()[:12]
    return U, I, model.dataset._allPos, f"ppd_{h}"   # official Loader.getUserPosItems, all users


def prepare(model):
    """Compute (and cache) the global popularity scores p of this backbone."""
    U, I, _, key = _inputs(model)
    with torch.no_grad():
        popularity(U, I, key)


def debiased_layer0(model, beta, phi):
    ut, _ = official_modules()
    U, I, all_pos, key = _inputs(model)
    with torch.no_grad():
        b = bias(U, I, all_pos, beta, key)
        d_u, d_i = ut.compute_d_pop_all_normalized(U, I, len(U), len(I), b, all_pos, phi)
        return ut.project_and_remove(U, d_u), ut.project_and_remove(I, d_i)


# ------------------------------------------------------------------------------------ scorer
@torch.no_grad()
def build_scorer(model, split, device, beta=DEFAULTS["beta"], phi=DEFAULTS["phi"]):
    U, I = debiased_layer0(model, float(beta), float(phi))
    model.eval()
    wu, wi = model.embedding_user.weight, model.embedding_item.weight
    old_u, old_i = wu.detach().clone(), wi.detach().clone()
    try:
        wu.copy_(U.to(wu.device))
        wi.copy_(I.to(wi.device))
        users, items = model.computer()  # official LightGCN propagation of the new layer 0
    finally:
        wu.copy_(old_u)
        wi.copy_(old_i)
    return lambda u: users[u.to(users.device)] @ items.T


if __name__ == "__main__":
    main("ppd")
