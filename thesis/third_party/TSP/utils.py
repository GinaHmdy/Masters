import random

import numpy as np
import scipy.sparse as sp
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scipy_sparse_mat_to_torch_sparse_tensor(sp_mat: sp.coo_matrix, device="cpu"):
    """Convert a scipy sparse matrix to a torch sparse coo tensor."""
    sp_mat = sp_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([sp_mat.row, sp_mat.col]).astype(np.int64))
    values = torch.from_numpy(sp_mat.data)
    shape = torch.Size(sp_mat.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce().to(device)


def sparse_dropout(sp_mat: torch.Tensor, dropout: float):
    """Dropout on the nonzero entries of a torch sparse tensor."""
    if dropout == 0.0:
        return sp_mat
    sp_mat = sp_mat.coalesce()
    keep = torch.rand(sp_mat._nnz(), device=sp_mat.device) >= dropout
    indices = sp_mat.indices()[:, keep]
    values = sp_mat.values()[keep] / (1.0 - dropout)
    return torch.sparse_coo_tensor(indices, values, sp_mat.shape).coalesce()


def spmm(sp_mat: torch.Tensor, dense: torch.Tensor):
    return torch.sparse.mm(sp_mat, dense)


def normalized_rating_matrix(train_csr: sp.csr_matrix):
    """D_U^{-1/2} R D_I^{-1/2}, the (user x item) block of the LightGCN norm. adj."""
    train_csr = train_csr.tocsr()
    user_deg = np.asarray(train_csr.sum(axis=1)).flatten()
    item_deg = np.asarray(train_csr.sum(axis=0)).flatten()
    user_deg[user_deg == 0] = 1.0
    item_deg[item_deg == 0] = 1.0
    d_u_inv = sp.diags(np.power(user_deg, -0.5))
    d_i_inv = sp.diags(np.power(item_deg, -0.5))
    return (d_u_inv @ train_csr @ d_i_inv).tocoo()


@torch.no_grad()
def evaluate(user_emb, item_emb, eval_users, train_csr, test_csr,
             tail_item_mask=None, k=20, device="cpu", extra_mask_csr=None,
             chunk_size=1024):
    """Full-ranking Recall@k / NDCG@k.

    Args:
        user_emb, item_emb: final (n_users, d) / (n_items, d) embeddings.
        eval_users: user ids to evaluate.
        train_csr: interactions to mask out of the ranking.
        test_csr: ground truth.
        tail_item_mask: optional boolean array (n_items,). If given, only tail
            items count as relevant (tail-item evaluation of the paper).
        k: cutoff.
        extra_mask_csr: additional interactions to mask (e.g. valid set when
            evaluating on test).
    Returns:
        (recall, ndcg) floats.
    """
    n_items = item_emb.shape[0]
    recalls, ndcgs = [], []
    discount = 1.0 / torch.log2(torch.arange(2, k + 2, device=device, dtype=torch.float32))

    for start in range(0, len(eval_users), chunk_size):
        uids = np.asarray(eval_users[start:start + chunk_size])
        scores = user_emb[torch.as_tensor(uids, device=device)] @ item_emb.T  # (B, n_items)

        # mask already-seen items
        mask = train_csr[uids].toarray()
        if extra_mask_csr is not None:
            mask = mask + extra_mask_csr[uids].toarray()
        mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
        scores = scores.masked_fill(mask, -1e9)

        topk = scores.topk(k, dim=1).indices.cpu().numpy()  # (B, k)

        for row, u in enumerate(uids):
            truth = test_csr.indices[test_csr.indptr[u]:test_csr.indptr[u + 1]]
            if tail_item_mask is not None:
                truth = truth[tail_item_mask[truth]]
            if len(truth) == 0:
                continue
            hits = np.isin(topk[row], truth)
            recalls.append(hits.sum() / len(truth))
            dcg = (torch.as_tensor(hits, dtype=torch.float32, device=device) * discount).sum().item()
            idcg = discount[: min(len(truth), k)].sum().item()
            ndcgs.append(dcg / idcg)

    if len(recalls) == 0:
        return 0.0, 0.0
    return float(np.mean(recalls)), float(np.mean(ndcgs))
