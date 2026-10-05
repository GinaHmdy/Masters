"""
The single evaluator shared by every method in the benchmark.

Protocol (identical to the old harness's final "Phase 0" protocol, LightGCN/code/Procedure.py):
  * full-catalogue ranking: every item is scored for every evaluated user;
  * the user's TRAINING items are masked out (score -1e9). Validation items are NOT masked when
    evaluating the test split, exactly as in the old harness;
  * where the split restricts the item universe (KuaiRec), items outside that split's own
    candidate set are masked too -- the validation universe for "valid", the test universe for
    "test", so the test universe never leaks into model selection;
  * every metric, accuracy and exposure alike, is averaged over the users of the evaluated split
    only (the keys of split.valid / split.test);
  * long tail = items outside the top 20% by training degree (ties broken by a stable sort);
  * catalogue-level metrics (Coverage, Gini) are computed over all n_items, even when candidates
    are restricted, as in the old code.

Metrics at every k: recall, precision, ndcg, hr, mrr, map, avgpop, aplt, coverage, gini,
tail_recall, novelty. recall/precision/ndcg/hr/avgpop/aplt/coverage/gini/tail_recall reproduce
the old formulas (utils.py); mrr, map and novelty are new.
"""
import numpy as np
import scipy.sparse as sp
import torch

MASK_SCORE = -1e9            # below any real score; models may return unbounded logits
LONG_TAIL_HEAD_FRACTION = 0.2


def long_tail_mask(item_pop, head_fraction=LONG_TAIL_HEAD_FRACTION):
    """True for long-tail items: everything outside the `head_fraction` most-interacted items."""
    item_pop = np.asarray(item_pop).reshape(-1)
    n_head = max(1, int(round(head_fraction * item_pop.size)))
    mask = np.ones(item_pop.size, dtype=bool)
    mask[np.argsort(-item_pop, kind="stable")[:n_head]] = False
    return mask


def gini(counts):
    """Gini of the exposure distribution over the whole catalogue (0 = equal, 1 = one item)."""
    x = np.sort(np.asarray(counts, dtype=np.float64).reshape(-1))
    if x.sum() <= 0:
        return 0.0
    n = x.size
    return float(2.0 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1.0) / n)


def _rows(m, users):
    """CSR rows -> (row index within batch, column) pairs as LongTensors."""
    sub = m[users]
    r = np.repeat(np.arange(len(users)), np.diff(sub.indptr))
    return torch.from_numpy(r), torch.from_numpy(sub.indices.astype(np.int64))


@torch.no_grad()
def rank(score_fn, split, part, max_k, batch_size=1024, device=None):
    """Top-`max_k` lists and their hit matrix for the users of `part`.

    score_fn(user_ids: LongTensor[B]) -> FloatTensor[B, n_items]. It must return a fresh tensor:
    the evaluator writes the mask into it in place.
    """
    if part not in ("valid", "test"):
        raise ValueError(f"part must be 'valid' or 'test', got {part!r}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    truth = split.test if part == "test" else split.valid
    cand = split.test_candidates if part == "test" else split.valid_candidates
    users = np.array(sorted(u for u, v in truth.items() if len(v) > 0), dtype=np.int64)

    truth_lists = [np.asarray(truth[u], dtype=np.int64) for u in users]
    n_true = np.array([len(t) for t in truth_lists])
    T = sp.csr_matrix((np.ones(n_true.sum(), dtype=bool),
                       (np.repeat(np.arange(len(users)), n_true), np.concatenate(truth_lists))),
                      shape=(len(users), split.n_items))
    outside = None
    if cand is not None:
        outside = torch.ones(split.n_items, dtype=torch.bool, device=device)
        outside[torch.as_tensor(np.asarray(cand), dtype=torch.long, device=device)] = False

    tops, hits = [], []
    for s in range(0, len(users), batch_size):
        u = users[s:s + batch_size]
        scores = score_fn(torch.from_numpy(u).to(device)).float()
        r, c = _rows(split.train, u)
        scores[r.to(scores.device), c.to(scores.device)] = MASK_SCORE
        if outside is not None:
            scores[:, outside.to(scores.device)] = MASK_SCORE
        top = torch.topk(scores, max_k, dim=1).indices
        gt = torch.zeros_like(scores, dtype=torch.bool)
        r, c = _rows(T, np.arange(s, s + len(u)))
        gt[r.to(gt.device), c.to(gt.device)] = True
        tops.append(top.cpu().numpy())
        hits.append(gt.gather(1, top).cpu().numpy())
        del scores, gt
    return users, n_true, np.concatenate(tops), np.concatenate(hits).astype(np.float64)


def evaluate(score_fn, split, part="test", ks=(10, 20), batch_size=1024,
             return_per_user=False, device=None):
    """Evaluate `score_fn` on split.valid or split.test. Returns {"recall@20": ..., ...}.

    With return_per_user=True, out["per_user"] holds {"users": ids, "<metric>@<k>": vector} for
    recall, precision, ndcg, hr, avgpop and aplt -- the inputs of significance.py.
    """
    users, n_true, top_all, hit_all = rank(score_fn, split, part, max(ks), batch_size, device)
    item_pop = np.asarray(split.item_pop, dtype=np.float64).reshape(-1)
    tail = long_tail_mask(item_pop)
    # self-information of an item; items never seen in training count as seen once
    info = -np.log2(np.maximum(item_pop, 1.0) / split.n_users)
    tail_true = np.array([tail[np.asarray(split.test[u] if part == "test" else split.valid[u])].sum()
                          for u in users])

    out = {"n_users": int(len(users))}
    per_user = {"users": users}
    for k in ks:
        h, top = hit_all[:, :k], top_all[:, :k]
        n_hit = h.sum(1)
        disc = 1.0 / np.log2(np.arange(2, k + 2))
        idcg = np.cumsum(disc)[np.minimum(k, n_true) - 1]
        first = h.argmax(1)
        pu = {
            "recall": n_hit / n_true,
            "precision": n_hit / k,
            "ndcg": (h * disc).sum(1) / idcg,
            "hr": (n_hit > 0).astype(np.float64),
            "mrr": np.where(n_hit > 0, 1.0 / (first + 1), 0.0),
            "map": (np.cumsum(h, 1) / np.arange(1, k + 1) * h).sum(1) / np.minimum(k, n_true),
            "avgpop": item_pop[top].mean(1),
            "aplt": tail[top].mean(1),
            "novelty": info[top].mean(1),
        }
        for name, v in pu.items():
            out[f"{name}@{k}"] = float(v.mean())
        counts = np.bincount(top.reshape(-1), minlength=split.n_items)
        out[f"coverage@{k}"] = float((counts > 0).sum() / split.n_items)
        out[f"gini@{k}"] = gini(counts)
        has_tail = tail_true > 0
        tail_hits = (h * tail[top]).sum(1)
        out[f"tail_recall@{k}"] = float((tail_hits[has_tail] / tail_true[has_tail]).mean()) \
            if has_tail.any() else 0.0
        for name in ("recall", "precision", "ndcg", "hr", "avgpop", "aplt"):
            per_user[f"{name}@{k}"] = pu[name]
    if return_per_user:
        out["per_user"] = per_user
    return out
