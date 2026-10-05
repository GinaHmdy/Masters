import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import sparse_dropout


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim, adj_norm, n_layers,
                 reg_weight=1e-4, dropout=0.0, device="cpu"):
        """
        Args:
            adj_norm: torch sparse (n_users x n_items) normalized rating matrix
                D_U^{-1/2} R D_I^{-1/2}.
        """
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.n_layers = n_layers
        self.reg_weight = reg_weight
        self.dropout = dropout
        self.device = device

        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_users, dim)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_items, dim)))
        self.adj_norm = adj_norm.to(device)          # U x I
        self.adj_norm_t = self.adj_norm.t().coalesce()  # I x U
        self.to(device)

        self.E_u_list = [None] * (n_layers + 1)
        self.E_i_list = [None] * (n_layers + 1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0

    # ------------------------------------------------------------ propagation
    def _propagate(self, dropout):
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        adj = sparse_dropout(self.adj_norm, dropout)
        adj_t = sparse_dropout(self.adj_norm_t, dropout)
        for layer in range(1, self.n_layers + 1):
            self.E_u_list[layer] = torch.sparse.mm(adj, self.E_i_list[layer - 1])
            self.E_i_list[layer] = torch.sparse.mm(adj_t, self.E_u_list[layer - 1])
        E_u = torch.stack(self.E_u_list, dim=0).mean(dim=0)
        E_i = torch.stack(self.E_i_list, dim=0).mean(dim=0)
        return E_u, E_i

    @torch.no_grad()
    def get_final_embeddings(self):
        """Final pretrained node embeddings X_0 (n_users + n_items, d)."""
        E_u, E_i = self._propagate(dropout=0.0)
        return torch.cat([E_u, E_i], dim=0)

    # ---------------------------------------------------------------- training
    def forward(self, uids, pos, neg):
        E_u, E_i = self._propagate(dropout=self.dropout)
        u_emb = E_u[uids]
        pos_emb = E_i[pos]
        neg_emb = E_i[neg]

        pos_scores = (u_emb * pos_emb).sum(-1)
        neg_scores = (u_emb * neg_emb).sum(-1)
        loss_bpr = -(pos_scores - neg_scores).sigmoid().log().mean()

        loss_reg = 0.5 * (self.E_u_0[uids].norm(2).pow(2)
                          + self.E_i_0[pos].norm(2).pow(2)
                          + self.E_i_0[neg].norm(2).pow(2)) / float(len(uids))
        loss_reg *= self.reg_weight

        return loss_bpr + loss_reg, loss_bpr
