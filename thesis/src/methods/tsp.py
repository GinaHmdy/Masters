"""TSP (Test-time Simplicial Propagation, sotaagi/TSP, Ji et al. NeurIPS 2025) on the trained backbone.

The upstream main.py is not in the repository (README lists it), so this is the minimal driver,
following the paper (arXiv 2411.13892, Sec. 4 and Alg. 1) and the docstrings of the official
tsp.py / model.py:
  * X0 = final pretrained LightGCN node embeddings, users then items, shape (n_users+n_items, d)
    (TSP model.LightGCN.get_final_embeddings = torch.cat([E_u, E_i]) of the layer-mean
    propagation; here the backbone's official computer(), the same quantity);
  * X_rec = tsp.TSP(n_users, n_items, K, beta, n_layers=L, theta, topk, variant, normalized,
    device)(X0)  -- the official class, unmodified (semantic graph -> clique complex (TopoNetX)
    -> intra-simplex lift -> L steps of (I - beta L_k) -> multi-order fusion, L2-normalised);
  * score(u, i) = <X_rec[u], X_rec[n_users + i]>  (the official utils.evaluate scores
    user_emb @ item_emb.T).
tsp.py does `from utils import scipy_sparse_mat_to_torch_sparse_tensor`, so it is imported with the
TSP directory first on sys.path and the LightGCN-PyTorch `utils` temporarily hidden
(src.methods.isolated_modules).

Runtime monkeypatch (documented, algorithm-preserving, disable with CACHE_COMPLEX = False):
TSP.__call__ rebuilds the clique complex and its incidence / Laplacian matrices on every call,
although they depend only on the semantic graph (X0, theta, topk) and K -- not on beta or L.
With TopoNetX this preprocessing is ~250-600 s per call on Gowalla (profiled: 85% in
SimplicialComplex.incidence_matrix / normalized_laplacian_matrix), i.e. ~7 h per seed for the
84-point grid. We replace the module global `tsp.graph_to_clique_complex` by a memoised version
keyed on (exact edge set of the semantic graph, max_rank) that returns the same SimplicialComplex
object, whose incidence_matrix / normalized_laplacian_matrix results are memoised per argument.
The official code only reads these matrices (abs/.T/astype/tocoo all copy), so every value is
identical to an uncached call (checked on Coat: identical top-20 lists; scores differ by <=2e-7,
the run-to-run noise of CUDA sparse matmuls). One entry is kept (the grid iterates K outermost).
X0 is computed once per backbone object (CUDA sparse-matmul noise could otherwise flip topk ties
in the semantic graph between calls: seen once on Gowalla, 588642 vs 588643 edges).
"""
from __future__ import annotations

import contextlib
import sys
import time

import torch

from . import THIRD, isolated_modules, main

TSP_DIR = THIRD / "TSP"
# README run command: --K 3 --beta 1.0 --L 2 --theta 0.3 --topk 10; class defaults otherwise
DEFAULTS = {"K": 3, "beta": 1.0, "L": 2, "theta": 0.3, "topk": 10, "variant": "full",
            "normalized": True}
CACHE_COMPLEX = True
_tsp = None
_complex_cache: dict = {}
last_timing: dict = {}
_x0: dict = {}


def _memo_method(obj, name):
    orig, memo = getattr(obj, name), {}

    def wrapped(*a, **k):
        key = (a, tuple(sorted(k.items())))
        if key not in memo:
            memo[key] = orig(*a, **k)
        return memo[key]
    setattr(obj, name, wrapped)


def _cached_clique_complex(orig):
    import hashlib

    import numpy as np

    def graph_to_clique_complex(graph, max_rank=None):
        if not CACHE_COMPLEX:
            return orig(graph, max_rank=max_rank)
        e = np.asarray(sorted((min(u, v), max(u, v)) for u, v in graph.edges()), dtype=np.int64)
        key = (graph.number_of_nodes(), hashlib.sha256(e.tobytes()).hexdigest(), max_rank)
        if key not in _complex_cache:
            _complex_cache.clear()
            sc = orig(graph, max_rank=max_rank)
            for name in ("incidence_matrix", "normalized_laplacian_matrix"):
                _memo_method(sc, name)
            _complex_cache[key] = sc
        return _complex_cache[key]
    return graph_to_clique_complex


def official_module():
    global _tsp
    if _tsp is None:
        with isolated_modules(TSP_DIR, ("utils", "tsp")):
            import tsp  # noqa: F401
            _tsp = sys.modules["tsp"]
        _tsp.graph_to_clique_complex = _cached_clique_complex(_tsp.graph_to_clique_complex)
    return _tsp


@torch.no_grad()
def build_scorer(model, split, device, K=DEFAULTS["K"], beta=DEFAULTS["beta"], L=DEFAULTS["L"],
                 theta=DEFAULTS["theta"], topk=DEFAULTS["topk"], variant=DEFAULTS["variant"],
                 normalized=DEFAULTS["normalized"]):
    T = official_module().TSP
    model.eval()
    if _x0.get("model") is not model:
        users, items = model.computer()
        _x0.update(model=model, X0=torch.cat([users, items]).detach().float())
    X0 = _x0["X0"]
    t = time.time()
    tsp = T(split.n_users, split.n_items, K=int(K), beta=float(beta), n_layers=int(L),
            theta=float(theta), topk=int(topk), variant=variant, normalized=bool(normalized),
            device=X0.device)
    with contextlib.redirect_stdout(sys.stderr):
        X = tsp(X0)
    last_timing.update(total_s=time.time() - t, preprocess_s=tsp.preprocess_time,
                       inference_s=tsp.inference_time)
    Xu, Xi = X[:split.n_users], X[split.n_users:]
    return lambda u: Xu[u.to(Xu.device)] @ Xi.T


if __name__ == "__main__":
    main("tsp")
