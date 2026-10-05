"""
ASTD-PD: Adaptive Spectral-Topological Debiasing with Pareto Dual-Stream Fusion.

Port of PPDv6ASTD (LightGCN/code/model.py) and PopularityDensityStore (dataloader.py). The math,
defaults, parameter-creation order (hence seeded initialisation) and state_dict names are
unchanged, so checkpoints of the old code load into this class. Removed because they were never
read by the old model: astd_debias_warmup_epochs, astd_seed_protocol, astd_use_logq,
astd_use_simplex_topk, the --beta flag, and the `embedding_user`/`embedding_item` aliases.
`topo_gate_init` is kept: it is the fixed theta of the no-simplex ablation
(topo_gate_learnable=False) and inert otherwise.

Two streams share one graph. Per layer, the accuracy stream A mixes the bipartite LightGCN
message M_G with a popularity-damped item-item ("simplex") message M_T through a learned gate
theta; the debiased stream D additionally removes the message's projection on a tracked
low-rank popularity subspace (adaptive strength alpha) and re-injects a high-frequency residual
(gamma). Scores fuse both streams per user: s = s_A + lambda_u * t_i * s_D, t_i = (1-p_i)^phi.
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.stats import rankdata

DEFAULTS = dict(
    latent_dim=64, n_layers=2, decay=1e-4,
    alpha_base=0.2, alpha_min=0.0, density_xi=1.0, alpha_pop_exponent=1.0,
    pop_basis_rank=2, pop_basis_ema=0.95, pop_basis_update_steps=50,
    topo_gate_init=0.5, topo_gate_learnable=True, topo_pop_damping=0.1,
    highfreq_gamma=0.1, user_alpha_scale=0.3,
    fuse_lambda_min=0.0, fuse_lambda_max=0.5, tail_phi=1.0, tail_reweight=True,
    temperature_acc=0.1, temperature_deb=0.1,
    lambda_final=1.0, lambda_acc=0.5, lambda_deb=0.5, lambda_ind=0.1, lambda_cov=0.1,
    ego_weight=-1.0, inference_pop_penalty=0.0, simplex_topk=200,
)


def _rank01(x):
    """Average-tie rank normalised to [0, 1]."""
    return ((rankdata(x, method="average") - 1.0) / (len(x) - 1.0)) if len(x) > 1 \
        else np.zeros(len(x))


def popularity_features(R):
    """Rank-normalised popularity / density features of users and items (PopularityDensityStore)."""
    R = sp.csr_matrix(R, dtype=np.float64)
    n_users, n_items = R.shape
    item_deg = np.asarray(R.sum(0)).ravel().astype(np.float32)
    user_deg = np.asarray(R.sum(1)).ravel().astype(np.float32)
    item_pop_rank = _rank01(item_deg).astype(np.float32)
    deciles = np.clip((item_pop_rank * 10.0).astype(np.int64), 0, 9)
    rows = [R.indices[R.indptr[u]:R.indptr[u + 1]] for u in range(n_users)]

    user_pop = np.array([item_pop_rank[r].mean() if len(r) else 0.5 for r in rows], np.float32)
    item_dens = np.asarray(R.tocsc().T.dot(user_deg)).ravel().astype(np.float32)   # sum_u R_ui deg_u
    user_dens = np.array([item_deg[r].sum() if len(r) else 0.0 for r in rows], np.float32)
    user_div = np.array([len(set(deciles[r])) / 10.0 if len(r) else 0.1 for r in rows], np.float32)
    t = lambda a, dt=torch.float32: torch.from_numpy(np.asarray(a)).to(dt)
    return dict(
        item_pop_rank=t(item_pop_rank), item_decile=t(deciles, torch.long),
        item_bucket=t(np.clip((item_pop_rank * 5.0).astype(np.int64), 0, 4), torch.long),
        item_density_rank=t(_rank01(item_dens).astype(np.float32)),
        user_pop=t(user_pop), user_density_rank=t(_rank01(user_dens).astype(np.float32)),
        user_diversity=t(user_div),
    )


def _to_torch(m, device):
    m = m.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack([m.row, m.col]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(m.data), m.shape).coalesce().to(device)


def normalized_adjacency(R):
    """D^-1/2 A D^-1/2 of the user-item bipartite graph (LightGCN's propagation matrix)."""
    n_users, n_items = R.shape
    A = sp.bmat([[sp.csr_matrix((n_users, n_users), dtype=np.float32), R],
                 [R.T, sp.csr_matrix((n_items, n_items), dtype=np.float32)]],
                format="csr", dtype=np.float32)
    rowsum = np.asarray(A.sum(1)).ravel()
    d = np.zeros_like(rowsum)
    d[rowsum > 0] = rowsum[rowsum > 0] ** -0.5
    return sp.diags(d).dot(A).dot(sp.diags(d))


def simplex_adjacency(R, item_pop_rank, kappa, top_k):
    """Popularity-damped item-item adjacency S^kappa, top-k pruned per item, sym. normalised:
    W_ij = sum_u R_ui R_uj / (sqrt(deg_i deg_j) (1+p_i)^(kappa/2) (1+p_j)^(kappa/2))."""
    R = sp.csr_matrix(R, dtype=np.float64)
    n_items = R.shape[1]
    deg = np.asarray(R.sum(0)).ravel().astype(np.float32)
    deg_inv_sqrt = np.zeros_like(deg)
    deg_inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
    damp = 1.0 / np.power(1.0 + item_pop_rank, kappa / 2.0)
    Rs = R.dot(sp.diags(deg_inv_sqrt * damp)).tocsr()
    C = Rs.T.dot(Rs).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    top_k = min(top_k, n_items)
    rows, cols, vals = [], [], []
    for i in range(n_items):
        c, v = C.indices[C.indptr[i]:C.indptr[i + 1]], C.data[C.indptr[i]:C.indptr[i + 1]]
        if len(c) > top_k:
            keep = np.argpartition(v, -top_k)[-top_k:]
            c, v = c[keep], v[keep]
        rows.extend([i] * len(c)); cols.extend(c); vals.extend(v)
    if not rows:  # no co-occurrence at all: identity
        rows = cols = list(range(n_items)); vals = [1.0] * n_items
    S = sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
    rowsum = np.asarray(S.sum(1)).ravel()
    d = np.zeros_like(rowsum)
    d[rowsum > 0] = rowsum[rowsum > 0] ** -0.5
    return sp.diags(d).dot(S).dot(sp.diags(d))


def _mlp(n_in):
    return nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())


class ASTDPD(nn.Module):
    """`train` is the COUNTED interaction matrix (Split.train_counts, the old UserItemNet):
    repeated pairs weigh 2 in the graphs, degrees and popularity features, as in the old code."""
    def __init__(self, train, device, **hp):
        super().__init__()
        unknown = set(hp) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown ASTD-PD hyperparameters: {sorted(unknown)}")
        self.hp = hp = {**DEFAULTS, **hp}
        self.num_users, self.num_items = train.shape
        d, L = hp["latent_dim"], hp["n_layers"]

        # creation order below fixes the seeded initialisation; do not reorder
        self.embedding_user_A = nn.Embedding(self.num_users, d)
        self.embedding_item_A = nn.Embedding(self.num_items, d)
        self.embedding_user_D = nn.Embedding(self.num_users, d)
        self.embedding_item_D = nn.Embedding(self.num_items, d)
        for e in (self.embedding_user_A, self.embedding_item_A,
                  self.embedding_user_D, self.embedding_item_D):
            nn.init.normal_(e.weight, std=0.1)
        for l in range(L + 1):   # per-layer popularity subspace basis, EMA-tracked, no gradient
            basis = torch.zeros(d, hp["pop_basis_rank"])
            nn.init.orthogonal_(basis)
            self.register_buffer(f"popularity_basis_{l}", basis)

        f = popularity_features(train)
        self.item_bucket = f["item_bucket"].to(device)
        self.item_decile = f["item_decile"]
        self.register_buffer("node_popularity", torch.cat([f["user_pop"], f["item_pop_rank"]]).view(-1, 1))
        self.register_buffer("node_density",
                             torch.cat([f["user_density_rank"], f["item_density_rank"]]).view(-1, 1))
        self.register_buffer("node_type", torch.cat([torch.ones(self.num_users), torch.zeros(self.num_items)]).view(-1, 1))
        self.register_buffer("user_diversity", f["user_diversity"].view(-1, 1))
        self.register_buffer("item_pop_rank_tensor", f["item_pop_rank"].clone())

        # alpha_max shrinks with the dataset's average item degree
        avg_item_deg = float(train.sum()) / max(1, self.num_items)
        scaled = hp["alpha_base"] / (1.0 + hp["density_xi"] * np.log1p(avg_item_deg / 20.0))
        self.effective_alpha_max = float(np.clip(scaled, hp["alpha_min"], hp["alpha_base"]))

        self.gate_mlp_theta = _mlp(3)   # [pop, density, is_user] -> topological gate theta
        self.gate_mlp_alpha = _mlp(2)   # [pop, density] -> projection strength
        self.gate_mlp_gamma = _mlp(2)   # [pop, density] -> high-frequency residual weight
        self.fuse_mlp_lambda = _mlp(3)  # [user pop, user density, user diversity] -> fusion lambda

        self.Graph = _to_torch(normalized_adjacency(sp.csr_matrix(train, dtype=np.float32)), device)
        self.Simplex_Item_Graph = _to_torch(simplex_adjacency(
            train, f["item_pop_rank"].numpy(), hp["topo_pop_damping"], hp["simplex_topk"]), device)
        self.register_buffer("train_step_count", torch.tensor(0, dtype=torch.long))
        self.last_mean_alpha = self.last_mean_theta = 0.0
        self.to(device)

    # ---- popularity subspace -------------------------------------------------------------
    def _basis(self, l):
        return getattr(self, f"popularity_basis_{l}")

    def _project(self, x, l):
        C = self._basis(l).detach()
        return x @ C @ C.t()

    @torch.no_grad()
    def update_popularity_basis(self):
        """EMA update of each layer's basis from popularity-bucket contrasts of item embeddings."""
        _, items_A, _, _ = self.layer_embeddings()
        r, ema = self.hp["pop_basis_rank"], self.hp["pop_basis_ema"]
        for l, emb in enumerate(items_A):
            emb = emb.detach()
            means = [emb[self.item_bucket == b].mean(0) if (self.item_bucket == b).any() else emb.mean(0)
                     for b in range(5)]
            V = torch.stack([means[4] - means[0], means[3] - means[1], means[2] - emb.mean(0)], dim=1)
            C_new = torch.linalg.qr(V)[0][:, :r]
            updated = ema * self._basis(l) + (1.0 - ema) * C_new
            self._basis(l).copy_(torch.linalg.qr(updated)[0][:, :r])

    # ---- propagation ---------------------------------------------------------------------
    def layer_embeddings(self):
        """Per-layer user/item embeddings of both streams (lists of length n_layers + 1)."""
        hp, nu = self.hp, self.num_users
        ea_u0, ea_i0 = self.embedding_user_A.weight, self.embedding_item_A.weight
        ed_u0, ed_i0 = self.embedding_user_D.weight, self.embedding_item_D.weight
        E_A, E_D = torch.cat([ea_u0, ea_i0]), torch.cat([ed_u0, ed_i0])
        users_A, items_A, users_D, items_D = [ea_u0], [ea_i0], [ed_u0], [ed_i0]

        feat_3 = torch.cat([self.node_popularity, self.node_density, self.node_type], dim=1)
        feat_2 = torch.cat([self.node_popularity, self.node_density], dim=1)
        p, e = self.node_popularity, hp["alpha_pop_exponent"]
        rho = p ** e / (p ** e + 0.5 ** e + 1e-8)                     # popularity modulation
        user_scale = self.node_type * hp["user_alpha_scale"] + (1.0 - self.node_type)
        alphas, thetas = [], []
        for layer in range(hp["n_layers"]):
            M_G = torch.sparse.mm(self.Graph, E_A)                       # bipartite message
            M_T = torch.cat([M_G[:nu], torch.sparse.mm(self.Simplex_Item_Graph, E_A[nu:])])
            theta = self.gate_mlp_theta(feat_3) if hp["topo_gate_learnable"] else \
                torch.full((E_A.shape[0], 1), hp["topo_gate_init"], device=E_A.device)
            m = theta * M_T + (1.0 - theta) * M_G
            h = E_A - m                                                  # high-frequency residual
            alpha = self.effective_alpha_max * self.gate_mlp_alpha(feat_2) * rho * user_scale
            gamma = hp["highfreq_gamma"] * self.gate_mlp_gamma(feat_2)
            m_D = m - alpha * self._project(m, layer) + gamma * h
            thetas.append(theta.mean().item()); alphas.append(alpha.mean().item())
            if hp["ego_weight"] >= 0.0:
                w = hp["ego_weight"]
                E_A = (1.0 - w) * m + w * torch.cat([ea_u0, ea_i0])
                E_D = (1.0 - w) * m_D + w * torch.cat([ed_u0, ed_i0])
            else:
                E_A, E_D = m, m_D
            for lst, E in ((users_A, E_A[:nu]), (items_A, E_A[nu:]), (users_D, E_D[:nu]), (items_D, E_D[nu:])):
                lst.append(E)
        self.last_mean_alpha, self.last_mean_theta = float(np.mean(alphas)), float(np.mean(thetas))
        return users_A, items_A, users_D, items_D

    def embeddings(self):
        """Layer-averaged (user_A, item_A, user_D, item_D)."""
        return tuple(torch.stack(x).mean(0) for x in self.layer_embeddings())

    def _lambda(self, users):
        u_feat = torch.cat([self.node_popularity[users], self.node_density[users],
                            self.user_diversity[users]], dim=1)
        lo, hi = self.hp["fuse_lambda_min"], self.hp["fuse_lambda_max"]
        return lo + (hi - lo) * self.fuse_mlp_lambda(u_feat)          # [B, 1]

    def _tail(self, p):
        return torch.pow(torch.clamp(1.0 - p, 0.0, 1.0), self.hp["tail_phi"])

    # ---- scoring -------------------------------------------------------------------------
    @torch.no_grad()
    def scorer(self):
        """score_fn for src.evaluate: propagates once, then scores user batches (Pareto fusion)."""
        self.eval()
        user_A, item_A, user_D, item_D = self.embeddings()
        hp = self.hp
        p_i = self.item_pop_rank_tensor.view(1, -1)
        t_i = self._tail(p_i) if hp["tail_reweight"] else torch.ones_like(p_i)

        def score(users):
            users = users.long().to(user_A.device)
            s_A = user_A[users] @ item_A.t() / max(1e-4, hp["temperature_acc"])
            s_D = user_D[users] @ item_D.t() / max(1e-4, hp["temperature_deb"])
            s = s_A + (self._lambda(users) * t_i) * s_D
            if hp["inference_pop_penalty"] > 0.0:
                s = s - hp["inference_pop_penalty"] * p_i
            return s
        return score

    # ---- training objective --------------------------------------------------------------
    def loss(self, users, pos, neg, neg_logq):
        """L = l_final L_fused + l_acc L_A + l_deb L_D,tail + l_ind L_ind + decay L2.
        Sampled softmax with logQ-corrected negatives; neg/neg_logq are [B, n_neg]."""
        hp = self.hp
        user_A, item_A, user_D, item_D = self.embeddings()
        tA, tD = max(1e-4, hp["temperature_acc"]), max(1e-4, hp["temperature_deb"])
        u_A, u_D = user_A[users], user_D[users]
        pos_A = (u_A * item_A[pos]).sum(-1) / tA
        neg_A = (u_A.unsqueeze(1) * item_A[neg]).sum(-1) / tA
        pos_D = (u_D * item_D[pos]).sum(-1) / tD
        neg_D = (u_D.unsqueeze(1) * item_D[neg]).sum(-1) / tD

        lam = self._lambda(users)
        # NB: the fused training score always uses t = (1-p)^phi; tail_reweight=False only
        # switches off the inference t_i and the L_D tail weights (as in the original code)
        t_pos, t_neg = self._tail(self.item_pop_rank_tensor[pos]), self._tail(self.item_pop_rank_tensor[neg])
        pos_F = pos_A + lam.squeeze(-1) * t_pos * pos_D
        neg_F = neg_A + lam * t_neg * neg_D

        def nll(p, n):   # -log softmax of the positive against logQ-corrected negatives
            return -torch.log_softmax(torch.cat([p.unsqueeze(1), n - neg_logq], dim=1), dim=1)[:, 0]

        loss_final, loss_acc = nll(pos_F, neg_F).mean(), nll(pos_A, neg_A).mean()
        w_tail = (1.0 + t_pos) if hp["tail_reweight"] else torch.ones_like(t_pos)
        loss_deb = (w_tail * nll(pos_D, neg_D)).mean()

        # popularity independence of the debiased item embeddings
        loss_proj = (self._project(item_D, hp["n_layers"]) ** 2).sum(-1).mean()
        p_c = self.item_pop_rank_tensor - self.item_pop_rank_tensor.mean()
        D_c = item_D - item_D.mean(0, keepdim=True)
        cov = (D_c * p_c.unsqueeze(1)).sum(0)
        loss_cov = (cov ** 2 / (((D_c ** 2).sum(0) + 1e-8) * ((p_c ** 2).sum() + 1e-8) + 1e-8)).mean()
        loss_ind = loss_proj + hp["lambda_cov"] * loss_cov

        loss_reg = 0.5 * sum(e.weight.norm(2).pow(2) for e in (
            self.embedding_user_A, self.embedding_item_A, self.embedding_user_D, self.embedding_item_D)) / len(users)
        total = (hp["lambda_final"] * loss_final + hp["lambda_acc"] * loss_acc + hp["lambda_deb"] * loss_deb
                 + hp["lambda_ind"] * loss_ind + hp["decay"] * loss_reg)
        stats = dict(final=loss_final.item(), acc=loss_acc.item(), deb_tail=loss_deb.item(),
                     ind=loss_ind.item(), reg=loss_reg.item(), lambda_fuse=lam.mean().item(),
                     alpha=self.last_mean_alpha, theta=self.last_mean_theta)
        return total, stats
