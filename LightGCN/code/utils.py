'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
from model import LightGCN
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import random
import os
try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except:
    world.cprint("Cpp extension not loaded")
    sample_ext = False


class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    return S

def UniformSample_original_python(dataset):
    """
    vectorized implementation of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    
    pos_lens = np.array([len(allPos[u]) for u in range(dataset.n_users)])
    
    valid_mask = pos_lens[users] > 0
    users = users[valid_mask]
    user_num = len(users)
    
    pos_indices = (np.random.rand(user_num) * pos_lens[users]).astype(int)
    pos_items = np.array([allPos[u][idx] for u, idx in zip(users, pos_indices)])
    
    neg_items = np.random.randint(0, dataset.m_items, user_num)
    
    is_pos = np.array(dataset.UserItemNet[users, neg_items]).flatten()
    invalid_indices = np.where(is_pos > 0)[0]
    
    while len(invalid_indices) > 0:
        new_negs = np.random.randint(0, dataset.m_items, len(invalid_indices))
        new_users = users[invalid_indices]
        new_is_pos = np.array(dataset.UserItemNet[new_users, new_negs]).flatten()
        neg_items[invalid_indices] = new_negs
        invalid_indices = invalid_indices[np.where(new_is_pos > 0)[0]]
        
    S = np.column_stack((users, pos_items, neg_items))
    return S

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def checkpoint_tag():
    """Per-worker suffix for the shared default checkpoint filename.

    `getFileName()` depends only on model/dataset/layers/dim, so every arm and every seed of a
    dataset writes to the SAME file (e.g. ppd_v6-Gowalla-3-64.pth.tar). The experiment drivers
    then copy that file to a seed-specific name. That is fine sequentially and fatal in
    parallel: two concurrent runs overwrite each other's checkpoint, and the second writer's
    weights get copied under the first's name. The mtime guard in the drivers cannot catch it,
    because the clobbered file is *newer*, not older.

    Set THESIS_CKPT_TAG to a unique value per concurrent worker (the parallel launcher does this)
    to give each one a private default checkpoint. Unset, behaviour is exactly as before.
    """
    return os.environ.get("THESIS_CKPT_TAG", "")


def getFileName():
    if world.model_name == 'mf':
        file = f"mf-{world.dataset}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'lgn':
        file = f"lgn-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'dap':
        file = f"dap-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'ppd_v6':
        file = f"ppd_v6-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    else:
        file = f"{world.model_name}-{world.dataset}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    tag = checkpoint_tag()
    if tag:
        file = file.replace(".pth.tar", f"-{tag}.pth.tar")
    return os.path.join(world.FILE_PATH, file)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecisionHR_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    hit_ratio = np.sum(right_pred > 0)
    return {'recall': recall, 'precision': precis, 'hr': hit_ratio}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

# ---------------------------------------------------------------------------
# Fairness / exposure metrics
#
# AvgPop@20 was the only fairness quantity in this codebase, and it is gameable:
# a Random recommender wins it outright on every dataset, because moving mass from
# the #1 item to the #50 item lowers it without exposing the tail at all. These four
# are computed from the same top-K matrix the accuracy metrics use, so they are free.
# ---------------------------------------------------------------------------

LONG_TAIL_HEAD_FRACTION = 0.2  # top 20% of items by train degree = "short head"


def long_tail_mask(item_degree, head_fraction=LONG_TAIL_HEAD_FRACTION):
    """
    Boolean mask over the catalogue, True for long-tail items.

    Short head = the `head_fraction` most-interacted items in the TRAINING graph
    (Abdollahpouri et al.'s definition). Ties are broken by the sort, so an item on
    the boundary lands deterministically.
    """
    item_degree = np.asarray(item_degree).reshape(-1)
    n_items = item_degree.shape[0]
    n_head = max(1, int(round(head_fraction * n_items)))
    head_idx = np.argsort(-item_degree, kind="stable")[:n_head]
    mask = np.ones(n_items, dtype=bool)
    mask[head_idx] = False
    return mask


def gini_at_k(rec_counts):
    """
    Gini coefficient of the item recommendation-frequency distribution.

    0.0 = every item recommended equally often, 1.0 = all exposure on one item.
    Computed over the WHOLE catalogue (items never recommended count as zeros),
    otherwise a model that ignores most of the catalogue scores misleadingly well.
    """
    x = np.asarray(rec_counts, dtype=np.float64).reshape(-1)
    total = x.sum()
    if total <= 0:
        return 0.0
    x = np.sort(x)
    n = x.shape[0]
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * total) - (n + 1.0) / n)


def coverage_at_k(rec_counts):
    """Fraction of the catalogue that appears in at least one user's top-K."""
    x = np.asarray(rec_counts).reshape(-1)
    return float((x > 0).sum() / x.shape[0]) if x.shape[0] else 0.0


def APLT_ATk(sorted_items, tail_mask, k):
    """
    Average Percentage of Long-Tail items. Summed over users (divided by user count
    by the caller, matching RecallPrecisionHR_ATk's convention).
    """
    topk = np.asarray(sorted_items)[:, :k]
    return float(tail_mask[topk].sum() / k)


def TailRecallATk(test_data, sorted_items, k, tail_mask):
    """
    Recall@k restricted to long-tail relevant items: of the tail items a user actually
    wanted, how many surfaced. This is the metric AvgPop@20 pretends to be -- unlike
    AvgPop it cannot be improved by shuffling exposure among head items, and unlike
    APLT it only rewards tail items the user genuinely wanted.

    Users whose ground truth contains no tail item are skipped entirely (they cannot
    contribute signal). Returns (summed_recall, n_users_counted); the caller divides.
    """
    pred = np.asarray(sorted_items)[:, :k]
    total, n_valid = 0.0, 0
    for i, items in enumerate(test_data):
        items = np.asarray(items, dtype=np.int64)
        if items.size == 0:
            continue
        tail_truth = items[tail_mask[items]]
        if tail_truth.size == 0:
            continue
        n_valid += 1
        hits = np.intersect1d(pred[i], tail_truth, assume_unique=False).size
        total += hits / tail_truth.size
    return float(total), n_valid


def per_user_metrics_ATk(test_data, sorted_items, k, tail_mask=None):
    """
    Per-user Recall@k / NDCG@k / Precision@k as vectors, not sums.

    The aggregate helpers above return sums for speed, which is fine for reporting a mean
    but makes a PAIRED significance test impossible -- you cannot pair two methods user by
    user if you only kept the totals. These vectors are what `significance.py` consumes.
    """
    pred = np.asarray(sorted_items)[:, :k]
    n = len(test_data)
    recall = np.zeros(n); ndcg = np.zeros(n); prec = np.zeros(n)
    aplt = np.zeros(n)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    for i, items in enumerate(test_data):
        items = np.asarray(items, dtype=np.int64)
        if items.size == 0:
            continue
        truth = set(items.tolist())
        hits = np.array([1.0 if it in truth else 0.0 for it in pred[i]])
        n_hit = hits.sum()
        recall[i] = n_hit / items.size
        prec[i] = n_hit / k
        dcg = float((hits * discount).sum())
        ideal_n = min(k, items.size)
        idcg = float(discount[:ideal_n].sum())
        ndcg[i] = dcg / idcg if idcg > 0 else 0.0
        if tail_mask is not None:
            aplt[i] = float(tail_mask[pred[i]].sum()) / k
    out = {"recall": recall, "ndcg": ndcg, "precision": prec}
    if tail_mask is not None:
        out["aplt"] = aplt
    return out


def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================
