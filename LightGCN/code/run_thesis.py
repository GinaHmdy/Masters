# run_thesis.py

import os
from pathlib import Path

import torch
import numpy as np
import networkx as nx
import toponetx as tnx
import scipy.sparse as sp

import world
import utils
import register
from register import dataset
import Procedure

from utilities import compute_pop_bias_scores
from constants import VARIANT_VECTOR_BIAS_ZERO_LAYER_UPDATE

# ======================
# Saving helpers
# ======================

def _fmt(x, nd=3):
    if isinstance(x, float):
        s = f"{x:.{nd}f}"
        s = s.rstrip("0").rstrip(".") if "." in s else s
        return s
    return str(x)

def build_out_tag(args) -> str:
    parts = []
    if getattr(args, "tsp", False):
        parts.append(
            "TSP"
            f"-th{_fmt(args.theta_tsp, 3)}"
            f"-m{args.top_m_tsp}"
            f"-k{args.k_tsp}"
            f"-b{_fmt(args.beta_tsp, 3)}"
            f"-L{args.layers_tsp}"
        )
    if getattr(args, "ppd", False):
        parts.append(
            "PPD"
            f"-b{_fmt(args.beta, 2)}"
            f"-phi{_fmt(args.phi, 2)}"
        )
    if not parts:
        parts.append("BASELINE")
    return "__".join(parts)

def derive_out_paths(ckpt_path: str, out_dir: str, tag: str):
    ckpt_path = Path(ckpt_path)
    out_dir = Path(out_dir)

    name = ckpt_path.name
    if name.endswith(".pth.tar"):
        base = name[:-len(".pth.tar")]
        ext = ".pth.tar"
    else:
        base = ckpt_path.stem
        ext = ckpt_path.suffix

    out_ckpt = out_dir / f"{base}__{tag}{ext}"
    out_res = out_dir / f"{base}__{tag}.results.txt"
    return str(out_ckpt), str(out_res)

def save_checkpoint_and_results(model, out_ckpt, out_res, results, args, baseline_ckpt):
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_ckpt)

    with open(out_res, "w") as f:
        f.write(f"dataset={world.dataset}\n")
        f.write(f"model={world.model_name}\n")
        f.write(f"variant={world.config.get('variant')}\n")
        f.write(f"recdim={world.config.get('latent_dim_rec')}\n")
        f.write(f"layer={world.config.get('lightGCN_n_layers')}\n")
        f.write(f"baseline_ckpt={baseline_ckpt}\n")
        f.write(f"tsp={getattr(args, 'tsp', False)}\n")
        f.write(f"ppd={getattr(args, 'ppd', False)}\n")
        f.write(f"results={results}\n")

    print(f"[SAVE] checkpoint -> {out_ckpt}")
    print(f"[SAVE] results    -> {out_res}")

# ======================
# TSP code (Optimized for T4)
# ======================

def scipy_to_torch_sparse(mat: sp.spmatrix, device, dtype=torch.float32):
    mat = mat.tocoo()
    idx = torch.tensor(np.vstack([mat.row, mat.col]), dtype=torch.long, device=device)
    val = torch.tensor(mat.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idx, val, torch.Size(mat.shape), device=device).coalesce()

@torch.no_grad()
def build_semantic_graph_knn_capped(X_cpu, theta, top_m, batch, max_edges):
    X = torch.nn.functional.normalize(X_cpu, p=2, dim=1)
    n = X.size(0)
    edges = set()
    
    # 🚀 T4 GPU OPTIMIZATION: Move matrix to 16GB VRAM for blazing fast dot products
    X_gpu = X.to('cuda')

    for s in range(0, n, batch):
        e = min(s + batch, n)

        # GPU Matrix Mult
        sims = X_gpu[s:e] @ X_gpu.t()
        vals, idx = torch.topk(sims, k=min(top_m + 1, n), dim=1)

        # Move back to CPU for set construction
        vals = vals.cpu()
        idx = idx.cpu()

        for r in range(e - s):
            i = s + r
            for v, j in zip(vals[r].tolist(), idx[r].tolist()):
                if i == j:
                    continue
                if v >= theta:
                    a, b = (i, j) if i < j else (j, i)
                    edges.add((a, b))
                    if len(edges) >= max_edges:
                        del X_gpu; torch.cuda.empty_cache()
                        G = nx.Graph()
                        G.add_nodes_from(range(n)); G.add_edges_from(edges)
                        return G, True
        del sims, vals, idx

    del X_gpu; torch.cuda.empty_cache()
    G = nx.Graph()
    G.add_nodes_from(range(n)); G.add_edges_from(edges)
    return G, False

def build_simplicial_complex_from_graph(G: nx.Graph, K: int):
    sc = tnx.SimplicialComplex()
    for v in G.nodes():
        sc.add_simplex([v])
    for clique in nx.enumerate_all_cliques(G):
        if len(clique) <= 1: continue
        if len(clique) > K + 1: break
        sc.add_simplex(clique)
    return sc

def list_k_simplices(sc: tnx.SimplicialComplex, k: int):
    return [tuple(sorted(list(s))) for s in sc.simplices if len(s) == k + 1]

@torch.no_grad()
def node_to_simplex_mean(X0, simplices_k):
    d = X0.size(1)
    Xk = torch.empty((len(simplices_k), d), device=X0.device, dtype=X0.dtype)
    for idx, verts in enumerate(simplices_k):
        Xk[idx] = X0[list(verts)].mean(dim=0)
    return Xk

@torch.no_grad()
def simplex_to_node_mean(Xk, simplices_k, n_nodes: int):
    d = Xk.size(1)
    X_back = torch.zeros((n_nodes, d), device=Xk.device, dtype=Xk.dtype)
    cnt = torch.zeros((n_nodes, 1), device=Xk.device, dtype=Xk.dtype)
    for idx, verts in enumerate(simplices_k):
        verts = list(verts)
        X_back[verts] += Xk[idx]
        cnt[verts] += 1.0
    return X_back / torch.clamp(cnt, min=1.0)

@torch.no_grad()
def inter_simplex_propagation_sparse(Xk, Lk_sparse, beta, layers):
    out = Xk
    for _ in range(layers):
        out = out - beta * torch.sparse.mm(Lk_sparse, out)
    return out

@torch.no_grad()
def apply_TSP(model, theta, top_m, K, beta, layers, semantic_batch, max_edges):
    u = model.embedding_user.weight.detach()
    it = model.embedding_item.weight.detach()
    device = next(model.parameters()).device

    X_cpu = torch.cat([u, it], dim=0).float().cpu()
    n_nodes = X_cpu.size(0)

    # 1. T4 Optimized Graph Building
    G, capped = build_semantic_graph_knn_capped(X_cpu, theta, top_m, semantic_batch, max_edges)
    print(f"[TSP] graph nodes={G.number_of_nodes()} edges={G.number_of_edges()}{' (CAPPED)' if capped else ''}")

    X0 = X_cpu.to(device)
    del X_cpu

    sc = build_simplicial_complex_from_graph(G, K=K)
    X0_from_orders = []

    for k in range(1, K + 1):
        simplices_k = list_k_simplices(sc, k)
        print(f"[TSP] {k}-simplices={len(simplices_k)}")
        if len(simplices_k) == 0: continue

        Xk = node_to_simplex_mean(X0, simplices_k)
        Lk_sp = sc.hodge_laplacian_matrix(k)
        Lk = scipy_to_torch_sparse(Lk_sp, device="cpu").to(device)

        Xk_p = inter_simplex_propagation_sparse(Xk, Lk, beta=beta, layers=layers)
        X0_from_k = simplex_to_node_mean(Xk_p, simplices_k, n_nodes=n_nodes)
        X0_from_orders.append(X0_from_k)

    if len(X0_from_orders) == 0:
        print("[TSP] No simplices found for any k>=1; skipping TSP update.")
        return model

    # 🚀 THE MATH FIX: Apply TSP as a Residual Delta, not an absolute addition
    alpha_tsp = 0.1 # Nudge factor
    X_mean = torch.stack(X0_from_orders, dim=0).mean(dim=0)
    
    # Blends (1 - alpha) of the Original and (alpha) of the TSP structure
    Xrec = X0 + alpha_tsp * (X_mean - X0) 

    before = torch.cat([u, it], dim=0)
    num_users = u.size(0)
    model.embedding_user.weight.data = Xrec[:num_users]
    model.embedding_item.weight.data = Xrec[num_users:]
    after = torch.cat([model.embedding_user.weight.data, model.embedding_item.weight.data], dim=0)

    print("[TSP] embedding delta norm:", torch.norm(after - before).item())
    return model

@torch.no_grad()
def calculate_average_popularity(model, dataset, topk=20):
    model.eval()
    u_emb = model.embedding_user.weight
    i_emb = model.embedding_item.weight
    item_counts = np.array(dataset.UserItemNet.sum(axis=0)).squeeze()

    num_users = u_emb.shape[0]
    batch_size = 512
    all_recs = []

    for i in range(0, num_users, batch_size):
        end = min(i + batch_size, num_users)
        scores = u_emb[i:end] @ i_emb.t()
        train_items = dataset.getUserPosItems(np.arange(i, end))
        for row_idx, items in enumerate(train_items):
            scores[row_idx, items] = -1e9
        _, top_items = torch.topk(scores, topk, dim=1)
        all_recs.append(top_items.cpu().numpy())

    all_recs = np.concatenate(all_recs, axis=0)
    avg_pop = np.mean(item_counts[all_recs])
    print(f"Average Popularity@{topk}: {avg_pop:.6f}")
    return avg_pop

# ======================
# Main
# ======================

if __name__ == "__main__":
    args = world.args 
    utils.set_seed(world.seed)

    model = register.MODELS[world.model_name](world.config, dataset).to(world.device)

    baseline_ckpt = args.ckpt if hasattr(args, "ckpt") and args.ckpt else utils.getFileName()
    model.load_state_dict(torch.load(baseline_ckpt, map_location=world.device, weights_only=False))
    print(f"[OK] loaded checkpoint: {baseline_ckpt}")

    if getattr(args, "tsp", False):
        model = apply_TSP(
            model,
            theta=args.theta_tsp,
            top_m=args.top_m_tsp,
            K=args.k_tsp,
            beta=args.beta_tsp,
            layers=args.layers_tsp,
            semantic_batch=args.semantic_batch,
            max_edges=args.max_edges_tsp,
        )

    if getattr(args, "ppd", False):
        compute_pop_bias_scores(
            "../../data/" + world.dataset,
            dataset._allPos,
            model,
            dataset,  
            best_epoch=0,
            variant=VARIANT_VECTOR_BIAS_ZERO_LAYER_UPDATE,
            beta=args.beta,
            phi=args.phi
        )

    print("\n[Eval]")
    results = Procedure.Test(
        dataset, model, epoch=0, w=None,
        multicore=world.config["multicore"],
        dataset_name=world.dataset,
        model_name=("ppd" if getattr(args, "ppd", False) else world.model_name),
    )
    print(results)

    if getattr(args, "fairness", False):
        print("\n[Average Popularity]")
        calculate_average_popularity(model, dataset, topk=20)

    if getattr(args, "save_ckpt", False):
        out_dir = getattr(args, "out_dir", "./checkpoints")
        tag = build_out_tag(args)
        out_ckpt, out_res = derive_out_paths(baseline_ckpt, out_dir, tag)
        save_checkpoint_and_results(model, out_ckpt, out_res, results, args, baseline_ckpt)