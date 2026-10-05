import time

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from toponetx.transform import graph_to_clique_complex

from utils import scipy_sparse_mat_to_torch_sparse_tensor


def _row_normalize(mat):
    """Row-normalize a scipy sparse matrix (rows sum to 1)."""
    mat = mat.tocsr().astype(np.float32)
    deg = np.asarray(mat.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg) @ mat


class TSP:
    def __init__(self, n_users, n_items, K=3, beta=1.0, n_layers=2,
                 theta=0.0, topk=10, variant="full", normalized=True,
                 device="cpu"):
        """
        Args:
            K: maximum simplicial complex dimension (paper searches 3-6).
            beta: simplicial filter coefficient (paper searches 1e-3 .. 1;
                use ~1 with normalized Laplacians, small values with raw ones).
            n_layers: number of simplicial message passing layers L (2-4).
            theta: similarity threshold of the semantic graph A_S.
            topk: keep at most this many neighbors per node in A_S (efficiency
                cap on top of the theta threshold; bounds clique complexity).
            variant: "full" (TSP), "se" (TSP-SE ablation: graph propagation on
                the semantic graph only), "off" (no-op, returns X_0).
            normalized: use normalized Hodge Laplacians (TopoNetX
                `normalized_laplacian_matrix`) for propagation; otherwise the
                raw L_k = B_k^T B_k + B_{k+1} B_{k+1}^T of Eq. (8).
        """
        self.n_users, self.n_items = n_users, n_items
        self.K, self.beta, self.n_layers = K, beta, n_layers
        self.theta, self.topk = theta, topk
        self.variant = variant
        self.normalized = normalized
        self.device = device
        self.preprocess_time = 0.0
        self.inference_time = 0.0

    @torch.no_grad()
    def build_semantic_graph(self, X0, chunk_size=4096):
        """A_S[i, j] = 1 if cos(x_i, x_j) >= theta, for at most `topk`
        neighbors per node. Returns a scipy csr adjacency (N x N)."""
        N = X0.shape[0]
        Xn = F.normalize(X0, dim=1)
        rows, cols = [], []
        for start in range(0, N, chunk_size):
            sim = Xn[start:start + chunk_size] @ Xn.T  # (B, N)
            idx = torch.arange(start, min(start + chunk_size, N), device=X0.device)
            sim[idx - start, idx] = -1.0  # remove self loops
            k = min(self.topk, N - 1) if self.topk else N - 1
            vals, nbrs = sim.topk(k, dim=1)
            keep = vals >= self.theta
            r, c = keep.nonzero(as_tuple=True)
            rows.append((r + start).cpu().numpy())
            cols.append(nbrs[r, c].cpu().numpy())
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        A_S = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                            shape=(N, N))
        # symmetrize so the semantic graph is undirected
        A_S = A_S.maximum(A_S.T)
        return A_S

    @torch.no_grad()
    def _graph_propagation(self, A_S, X0):
        """TSP-SE: LightGCN-style message passing on the semantic graph
        (no higher-order structure)."""
        deg = np.asarray(A_S.sum(axis=1)).flatten()
        deg[deg == 0] = 1.0
        A_norm = sp.diags(np.power(deg, -0.5)) @ A_S @ sp.diags(np.power(deg, -0.5))
        A_norm = scipy_sparse_mat_to_torch_sparse_tensor(A_norm, self.device)
        X = X0
        outs = [X0]
        for _ in range(self.n_layers):
            X = torch.sparse.mm(A_norm, X)
            outs.append(X)
        return torch.stack(outs, dim=0).mean(dim=0)

    @torch.no_grad()
    def __call__(self, X0):
        """Run TSP on pretrained embeddings X0 (N, d); returns X_rec (N, d)."""
        if self.variant == "off":
            return X0

        t0 = time.time()
        A_S = self.build_semantic_graph(X0)
        n_edges = A_S.nnz // 2
        print(f"[tsp] semantic graph: {A_S.shape[0]} nodes, {n_edges} edges "
              f"(theta={self.theta}, topk={self.topk})")

        if self.variant == "se":
            self.preprocess_time = time.time() - t0
            t1 = time.time()
            X_rec = self._graph_propagation(A_S, X0)
            self.inference_time = time.time() - t1
            return X_rec


        graph = nx.from_scipy_sparse_array(A_S)
        sc = graph_to_clique_complex(graph, max_rank=self.K)
        print(f"[tsp] {sc}")
        max_k = sc.dim
        if max_k < 1:  # no edges survived: nothing to propagate
            print("[tsp] warning: empty simplicial complex, returning X_0.")
            return X0

        B_lift = [None] + [scipy_sparse_mat_to_torch_sparse_tensor(
            _row_normalize(abs(sc.incidence_matrix(k)).T), self.device)
            for k in range(1, max_k + 1)]
        B_proj = [None] + [scipy_sparse_mat_to_torch_sparse_tensor(
            _row_normalize(abs(sc.incidence_matrix(k))), self.device)
            for k in range(1, max_k + 1)]

        L = []
        for k in range(0, max_k + 1):
            if self.normalized:
                Lk = sc.normalized_laplacian_matrix(k)
            else:
                Lk = sp.csr_matrix((sc.shape[k], sc.shape[k]), dtype=np.float32)
                if k > 0:
                    Bk = sc.incidence_matrix(k)
                    Lk = Lk + Bk.T @ Bk            # down (lower) component
                if k < max_k:
                    Bk1 = sc.incidence_matrix(k + 1)
                    Lk = Lk + Bk1 @ Bk1.T          # up (upper) component
            L.append(scipy_sparse_mat_to_torch_sparse_tensor(Lk, self.device))
        self.preprocess_time = time.time() - t0
        t1 = time.time()

        Xk = [X0]
        for k in range(1, max_k + 1):
            Xk.append(torch.sparse.mm(B_lift[k], Xk[k - 1]))

        for k in range(0, max_k + 1):
            for _ in range(self.n_layers):
                Xk[k] = Xk[k] - self.beta * torch.sparse.mm(L[k], Xk[k])

        fused = []
        for k in range(1, max_k + 1):
            Y = Xk[k]
            for i in range(k, 0, -1):          # X_{0<-k} = (B_1 B_2 ... B_k) Y
                Y = torch.sparse.mm(B_proj[i], Y)
            fused.append(Y)
        X_rec = F.normalize(X0 + torch.stack(fused, dim=0).mean(dim=0))

        self.inference_time = time.time() - t1
        return X_rec
