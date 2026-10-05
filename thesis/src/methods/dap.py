"""DAP (Debias the Amplification of Popularity, MEICRS/DAP, Chen et al. FCS 2023) on the backbone.

The paper (arXiv 2305.14886, Sec. 3.3, Eqs. 16-18) applies DAP to a trained GCN at inference:
at every layer l the propagated embeddings are clustered with Kmeans (P clusters); for node v the
mean of the normalised higher-degree / lower-degree members of its cluster give anchors H, L and
    e_v <- e_v - alpha * M(e_v, H) H - beta * M(e_v, L) L.
That is `LightGCN.computer_DAP` in the official DAP.py (config keys alpha1 = alpha, alpha2 = beta,
cluster_num = P). `computer_one` uses one-hop neighbours instead of clusters, i.e. the paper's
DAP-nh/-nl ablation (Sec. 4.3.1), so it is not used. The README's usage ("modify the model.py in
LightGCN-PyTorch") means: swap the model class in, load trained weights, score with
getUsersRating(*computer_DAP(), users). We construct the official DAP.LightGCN, load the backbone's
state_dict into it (same parameter names: embedding_user.weight, embedding_item.weight) and give
it the backbone's official normalised adjacency.

DAP.py cannot be imported unmodified into LightGCN-PyTorch as-is; adapter-only workarounds (the
two stubs below -- BasicModel from the official LightGCN-PyTorch, cos_similarity verbatim from the
PPD repo's DAP port -- are approved by the thesis author):
  * `from model.basic_model import BasicModel`: LightGCN-PyTorch has model.py, not a `model`
    package. A stub package `model` with submodule `basic_model` exposing the official
    LightGCN-PyTorch model.BasicModel is put in sys.modules for the import only.
  * `from utils import cprint, cos_similarity`: neither LightGCN-PyTorch nor the DAP repo ships
    cos_similarity. A stub `utils` provides cprint = official world.cprint and cos_similarity taken
    verbatim from the PPD repo's port of DAP (third_party/PPD/LightGCN/code/DAP.py: row-wise
    cosine, shape (N,)) -- the only definition consistent with the `.unsqueeze(1) * anchor` use.
  * `sys.path.append('../')` (cwd-relative) is undone after the import.
  * __init__ reads config['GCN_n_layers'] and expects dataset.getSparseGraph() -> (Graph, Graph_one);
    we pass the official config plus GCN_n_layers/alpha1/alpha2/cluster_num and a dataset view
    whose getSparseGraph returns (backbone Graph, None) (Graph_one is never used).
Scores are raw inner products of computer_DAP's outputs (the official getUsersRating applies a
sigmoid: same ranking, minus fp32 saturation ties).
Known official behaviour kept as-is: a cluster whose members all share one degree yields an
empty-mean (NaN) anchor for them; they are counted in `last_info['nan_rows']`.
"""
from __future__ import annotations

import contextlib
import sys
import types

import torch

from . import THIRD, isolated_modules, load_file, main

DAP_FILE = THIRD / "DAP" / "DAP.py"
PPD_DAP_FILE = THIRD / "PPD" / "LightGCN" / "code" / "DAP.py"
# alpha1/alpha2/cluster_num have no defaults upstream; PPD main.py's DAP run uses 0.1 / 0.0 / {1, 5}
DEFAULTS = {"alpha1": 0.1, "alpha2": 0.0, "cluster_num": 5}
_dap = None
last_info: dict = {}


def official_module():
    """The official DAP.py module, imported against the already-loaded LightGCN-PyTorch modules."""
    global _dap
    if _dap is None:
        import src.backbone as B
        assert B._official is not None, "load the backbone first (official modules)"
        off = B._official
        path_before, dwb = list(sys.path), sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            with contextlib.redirect_stdout(sys.stderr):
                # The ONLY two stubs on DAP's import path (approved by the thesis author):
                #   model.basic_model.BasicModel = the official LightGCN-PyTorch model.BasicModel;
                #   utils.cos_similarity = verbatim from the PPD repo's DAP port
                #   (third_party/PPD/LightGCN/code/DAP.py), since neither DAP nor
                #   LightGCN-PyTorch ships it. Nothing else of DAP.py is replaced.
                # PPD's port of DAP imports `world`, `dataloader`, `model.BasicModel` -> resolves to
                # the LightGCN-PyTorch modules in sys.modules; we only take cos_similarity from it.
                ppd_dap = load_file(PPD_DAP_FILE, "_ppd_dap_port")
                ut = types.ModuleType("utils")
                ut.cprint, ut.cos_similarity = off.world.cprint, ppd_dap.cos_similarity
                pkg = types.ModuleType("model")
                pkg.__path__ = []
                bm = types.ModuleType("model.basic_model")
                bm.BasicModel = off.model.BasicModel
                pkg.basic_model = bm
                with isolated_modules(None, ("utils", "model"),
                                      stubs={"utils": ut, "model": pkg, "model.basic_model": bm}):
                    _dap = load_file(DAP_FILE, "_dap_official")
        finally:
            sys.path[:] = path_before
            sys.dont_write_bytecode = dwb
    return _dap


def dap_model(model, alpha1, alpha2, cluster_num):
    import src.backbone as B
    D = official_module()
    cfg = dict(B._official.world.config, GCN_n_layers=model.n_layers, alpha1=float(alpha1),
               alpha2=float(alpha2), cluster_num=int(cluster_num), pretrain=0)
    ds = model.dataset
    view = types.SimpleNamespace(n_users=ds.n_users, m_items=ds.m_items, users_D=ds.users_D,
                                 items_D=ds.items_D, getSparseGraph=lambda: (model.Graph, None))
    rng = torch.get_rng_state()  # __init__ draws random init weights; keep global RNG untouched
    with contextlib.redirect_stdout(sys.stderr):
        net = D.LightGCN(cfg, view)
    torch.set_rng_state(rng)
    net.load_state_dict(model.state_dict(), strict=True)
    net = net.to(model.embedding_user.weight.device)
    net.Graph = model.Graph
    return net.eval()


# ------------------------------------------------------------------ memoised computer_DAP
# Profile (Gowalla, official computer_DAP): almost all time is the per-node Python loop, which for
# EVERY node re-scans its whole cluster (degree_cluster.min()/max(), torch.where, mean of the
# higher/lower part): O(N * |cluster| * d) and ~10 tiny CUDA kernels per node, which crawl when the
# GPUs are shared. Kmeans is a few seconds.
# MEMO = True replaces only that loop by a memoised but otherwise line-for-line copy of
# computer_DAP: inside one cluster the anchors of node v depend on nothing but its degree d_v
# (H = mean of members with degree > d_v, L = mean of members with degree < d_v, chosen by the same
# min/max branches), so they are computed once per (cluster, distinct degree) with the very same
# tensor expressions on the same index subsets and written to every node of that degree. The
# layer-1 Kmeans labels (input G @ E0 does not depend on alpha1/alpha2) are additionally cached
# per (backbone, P) (all P values kept, so the grid needs no reordering). Everything else -- persistence of anchors across layers, normalize,
# cos_similarity (the official module's), the update and the layer mean -- is the official code.
# Verified bit-identical to net.computer_DAP() (MEMO = False) on Coat and Yahoo, every P of the
# grid (see report); the official method stays callable with MEMO = False.
MEMO = True
_kmeans_cache: dict = {}


def _computer_dap_memo(net, D, cache_key):
    import numpy as np
    world = D.world
    nn = torch.nn
    users_emb = net.embedding_user.weight
    items_emb = net.embedding_item.weight
    all_emb = torch.cat([users_emb, items_emb])
    u_degree = torch.tensor(net.dataset.users_D)
    i_degree = torch.tensor(net.dataset.items_D)
    degree = torch.cat([u_degree, i_degree], dim=0)
    all_embs_anchor_higher = torch.zeros([net.num_users + net.num_items, net.latent_dim]).to(world.device)
    all_embs_anchor_lower = torch.zeros([net.num_users + net.num_items, net.latent_dim]).to(world.device)
    embs = [all_emb]
    g_droped = net.Graph
    weight1, weight2 = net.config['alpha1'], net.config['alpha2']
    num_cluster = net.config['cluster_num']
    for layer in range(net.n_layers):
        all_emb_i = torch.sparse.mm(g_droped, all_emb)
        key = (cache_key, num_cluster)
        if layer == 0 and key in _kmeans_cache:
            all_emb_cluster = _kmeans_cache[key]
        else:
            kmeans = D.KMeans(n_clusters=num_cluster, random_state=9)
            all_emb_cluster = kmeans.fit_predict(all_emb_i.to('cpu'))
            if layer == 0:  # one entry per P of the current backbone (grid order is irrelevant)
                for k in [k for k in _kmeans_cache if k[0] != cache_key]:
                    del _kmeans_cache[k]
                _kmeans_cache[key] = all_emb_cluster
        for k_cluster in range(num_cluster):
            index = np.where(all_emb_cluster == k_cluster)
            embs_cluster = all_emb_i[index[0]]
            degree_cluster = degree[index[0]].unsqueeze(1)
            dmin, dmax = degree_cluster.min(), degree_cluster.max()
            node_deg = degree[index[0]]
            for d in torch.unique(node_deg):
                members = index[0][(node_deg == d).numpy()]
                if d > dmin and d < dmax:
                    degree_higher_index = torch.where(degree_cluster > d)
                    degree_lower_index = torch.where(degree_cluster < d)
                    all_embs_anchor_higher[members] = embs_cluster[degree_higher_index[0]].mean(0)
                    all_embs_anchor_lower[members] = embs_cluster[degree_lower_index[0]].mean(0)
                elif d == dmin:
                    degree_higher_index = torch.where(degree_cluster > d)
                    all_embs_anchor_higher[members] = embs_cluster[degree_higher_index[0]].mean(0)
                elif d == dmax:
                    degree_lower_index = torch.where(degree_cluster < d)
                    all_embs_anchor_lower[members] = embs_cluster[degree_lower_index[0]].mean(0)
        all_embs_anchor_higher = nn.functional.normalize(all_embs_anchor_higher)
        all_embs_anchor_lower = nn.functional.normalize(all_embs_anchor_lower)
        alpha_sim_higher = D.cos_similarity(all_emb, all_embs_anchor_higher.to(world.device))
        alpha_sim_lower = D.cos_similarity(all_emb, all_embs_anchor_lower.to(world.device))
        all_emb = all_emb_i - weight1 * alpha_sim_higher.unsqueeze(1).to(world.device) * all_embs_anchor_higher.to(world.device)\
                  - weight2 * alpha_sim_lower.unsqueeze(1).to(world.device) * all_embs_anchor_lower.to(world.device)
        embs.append(all_emb)
    embs = torch.stack(embs, dim=1)
    light_out = torch.mean(embs, dim=1)
    users, items = torch.split(light_out, [net.num_users, net.num_items])
    return users, items


@torch.no_grad()
def build_scorer(model, split, device, alpha1=DEFAULTS["alpha1"], alpha2=DEFAULTS["alpha2"],
                 cluster_num=DEFAULTS["cluster_num"]):
    net = dap_model(model, alpha1, alpha2, cluster_num)
    with contextlib.redirect_stdout(sys.stderr):
        if MEMO:
            key = (id(model), float(model.embedding_user.weight.detach().sum()))
            users, items = _computer_dap_memo(net, official_module(), key)
        else:
            users, items = net.computer_DAP()
    last_info.update(nan_rows=int(torch.isnan(users).any(1).sum() + torch.isnan(items).any(1).sum()))
    return lambda u: users[u.to(users.device)] @ items.T


if __name__ == "__main__":
    main("dap")
