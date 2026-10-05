from collections import defaultdict

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

VARIANT_EMBEDDING_UPDATE = 1
VARIANT_TRACK_AND_PRUNE = 2
VARIANT_VECTOR_BIAS_ZERO_LAYER_UPDATE = 3
VARIANT_VECTOR_BIAS_ALL_LAYER_UPDATE = 4

# import Procedure, world
# from register import dataset



def compute_global_popularity_score(user_embeddings, item_embeddings, similarity, sample_size=-1):
    """
    user_embeddings: Tensor of shape (num_users, embed_dim)
    item_embeddings: Tensor of shape (num_items, embed_dim)
    similarity: 'dot' or 'cosine'
    sample_size: number of user embeddings to randomly sample (default: -1), if -1 take all users
    """
    num_users = len(user_embeddings)
    popularity_scores = []

    for i, e_i in enumerate(item_embeddings):  # for each item
        # Use item index as seed → ensures reproducible sampling for each item
        g = torch.Generator()
        g.manual_seed(i + 4200)

        if sample_size <= 0:
            sampled_users = user_embeddings
        elif num_users > sample_size:
            indices = torch.randperm(num_users, generator=g)[:sample_size]
            sampled_users = user_embeddings[indices]
        else:
            sampled_users = user_embeddings

        num_sampled = len(sampled_users)
        score_sum = 0.0

        for e_u in sampled_users:  # for sampled users
            if similarity == 'dot':
                sim = torch.dot(e_u, e_i).item()
            elif similarity == 'cosine':
                sim = F.cosine_similarity(e_u.unsqueeze(0), e_i.unsqueeze(0)).item()
            else:
                raise ValueError("Unsupported similarity")

            score_sum += sim

        p_i = score_sum / num_sampled
        popularity_scores.append(p_i)

    popularity_scores = torch.tensor(popularity_scores)

    # Min-max scaling to [0, 1]
    p_min, p_max = popularity_scores.min(), popularity_scores.max()
    if p_max == p_min:
        scaled_popularity_scores = torch.zeros_like(popularity_scores)
    else:
        scaled_popularity_scores = (popularity_scores - p_min) / (p_max - p_min)

    return scaled_popularity_scores


def compute_personalized_preference_scores(item_embeddings, user_interactions, popularity_score_of_items,
                                           similarity, gamma):
    """
    Computes r_ui only for user u's interacted items i ∈ N_u using similarity.

    Args:
        item_embeddings (Tensor): [num_items, dim]
        user_interactions (list of lists): user_interactions[u] = list of item indices
        popularity_score_of_items (list): popularity_score_of_items[i] = popularity score of item i
        similarity (str): 'dot' or 'cosine'
        gamma (float): popularity penalty factor

    Returns:
        rui_scores -> a 2D array containing the personalized scores of interactions for each user
    """
    rui_scores = []

    for u in range(len(user_interactions)):
        items_u = user_interactions[u]
        user_scores = []
        if not items_u:
            rui_scores.append(user_scores)
            continue

        for i in items_u:
            e_i = item_embeddings[i]
            sim_sum = 0.0

            for j in items_u:
                e_j = item_embeddings[j]

                penalty_for_popularity = gamma * popularity_score_of_items[i] * popularity_score_of_items[j]

                if similarity == 'dot':
                    # sim = torch.dot(e_i, e_j).item()
                    sim = torch.sigmoid(torch.dot(e_i, e_j)).item()
                elif similarity == 'cosine':
                    sim = F.cosine_similarity(e_i.unsqueeze(0), e_j.unsqueeze(0)).item()
                else:
                    raise ValueError("Unsupported similarity type")

                sim_sum += (sim - penalty_for_popularity)

            r_ui = sim_sum / len(items_u)
            user_scores.append(r_ui)

        rui_scores.append(user_scores)

    return rui_scores


def normalize_scores_global(score_matrix, eps=1e-8):
    """
    Applies global min-max normalization to a 2D list of scores.

    Args:
        score_matrix (List[List[float]]): score_matrix[u] = list of s_ui for i in N_u
        eps (float): small constant to avoid division by zero

    Returns:
        List[List[float]]: same structure, but each score is min-max normalized to [0, 1]
    """
    # Flatten all scores
    flat_scores = [score for user_scores in score_matrix for score in user_scores]

    if not flat_scores:
        return score_matrix  # return as-is if no scores

    min_val = min(flat_scores)
    max_val = max(flat_scores)

    # Normalize
    normalized = []
    for user_scores in score_matrix:
        normed_user_scores = [
            (s - min_val) / (max_val - min_val + eps) for s in user_scores
        ]
        normalized.append(normed_user_scores)

    return normalized


def compute_bias_scores(popularity_score_of_items, rui_scores_of_user_interactions, user_interactions):
    """
    Computes popularity bias score b_ui = p_i - r_ui for all (u, i ∈ N_u).

    Args:
        popularity_score_of_items (Tensor): shape [num_items], min-max normalized
        rui_scores_of_user_interactions (List[List[float]]): rui_scores[u] = list of r_ui for i in user_interactions[u],
                                                                    min-max normalized
        user_interactions (List[List[int]]): user_interactions[u] = list of item indices

    Returns:
        List[List[float]]: bias_scores[u] = list of b_ui for i in user_interactions[u]
    """
    bias_scores = []

    for u in range(len(user_interactions)):
        user_bias = []
        items_u = user_interactions[u]
        rui_u = rui_scores_of_user_interactions[u]

        for idx, i in enumerate(items_u):
            r_ui = rui_u[idx]
            b_ui = popularity_score_of_items[i] - r_ui
            user_bias.append(b_ui)

        bias_scores.append(user_bias)

    return bias_scores


def compute_pop_bias_scores(user_interactions, Recmodel, best_epoch, variant,
                            dataset, beta, phi, similarity='dot'):
    """
        Computes popularity bias score.

        Args:
            root_path (str): directory path of dataset
            user_interactions (List[List[int]]): user_interactions[u] = list of item indices
            similarity (str): 'dot' or 'cosine'

        Returns:
            List[List[float]]: bias_scores[u] = list of b_ui for i in user_interactions[u]
        """

    # model_state_dict = torch.load(root_path + "/model_state_dict_lgn.pth")
    # user_emb_weight = (model_state_dict['embedding_user.weight']).detach().clone()
    # item_emb_weight = (model_state_dict['embedding_item.weight']).detach().clone()
    user_emb_weight = Recmodel.embed_user.weight
    item_emb_weight = Recmodel.embed_item.weight
    user_emb_weight = nn.functional.normalize(user_emb_weight)
    item_emb_weight = nn.functional.normalize(item_emb_weight)
    # for Coat = 0.30, for KuaiRec = 0.20
    penalty_coefficient_in_personalized_scores = beta
    # penalty_coefficient_in_personalized_scores = 0.05

    popularity_score_of_items = compute_global_popularity_score(user_emb_weight, item_emb_weight, similarity, sample_size=-1)
    rui_scores_of_user_interactions = compute_personalized_preference_scores(item_emb_weight,
                                                                             user_interactions,
                                                                             popularity_score_of_items, similarity,
                                                                             penalty_coefficient_in_personalized_scores)
    rui_scores_of_user_interactions = normalize_scores_global(rui_scores_of_user_interactions)
    bias_scores_of_user_interactions = compute_bias_scores(popularity_score_of_items,
                                                           rui_scores_of_user_interactions, user_interactions)
    bias_scores_of_user_interactions = normalize_scores_global(bias_scores_of_user_interactions)
    # np.savetxt(root_path + "/pop_bias_scores.txt", bias_scores_of_user_interactions, fmt='%.5f', delimiter=' ')
    # plot_bias_score_histogram(bias_scores_of_user_interactions, root_path)

    # popular_item_ids = get_popular_item_ids(popularity_score_of_items, top_k_percent=0.10)
    # bias_direction = F.normalize(item_emb_weight[popular_item_ids].mean(dim=0), p=2, dim=0)

    if variant == VARIANT_EMBEDDING_UPDATE:
        pass
        # Recmodel.Graph = dataloader.getSparseGraphWithWeight(bias_scores_of_user_interactions)
        # results_test = Procedure.Test(dataset, Recmodel, best_epoch, None, world.config['multicore'])
    elif variant == VARIANT_VECTOR_BIAS_ZERO_LAYER_UPDATE:
        d_pop_u, d_pop_i = compute_d_pop_all_normalized(user_emb_weight, item_emb_weight, len(user_emb_weight),
                          len(item_emb_weight), bias_scores_of_user_interactions, user_interactions, phi)
        user_emb_weight = project_and_remove(user_emb_weight, d_pop_u)
        item_emb_weight = project_and_remove(item_emb_weight, d_pop_i)

        with torch.no_grad():
            if 'yahoo' in dataset or 'Yahoo' in dataset:
                eta = 0.3
                Recmodel.embed_user.weight.mul_(eta).add_((1 - eta) * user_emb_weight)
                Recmodel.embed_item.weight.mul_(eta).add_((1 - eta) * item_emb_weight)
            else:
                Recmodel.embed_user.weight.copy_(user_emb_weight)
                Recmodel.embed_item.weight.copy_(item_emb_weight)
                # Recmodel.state_dict['embed_user.weight'] = user_emb_weight
                # Recmodel.state_dict['embed_item.weight'] = item_emb_weight
        # results_test = Procedure.Test(dataset, Recmodel, best_epoch, None, world.config['multicore'],
        #                               posthoc=0, all_updated_embeddings=None)
    elif variant == VARIANT_VECTOR_BIAS_ALL_LAYER_UPDATE:
        pass
        # Recmodel.bias_scores_of_user_interactions = bias_scores_of_user_interactions
        # Recmodel.user_interactions = user_interactions
        # Recmodel.set_item_to_users()
        # all_updated_embeddings = Recmodel.forward_post_hoc_with_bias_vector_estimation(compute_d_pop_all_normalized)

        # results_test = Procedure.Test(dataset, Recmodel, best_epoch, None, world.config['multicore'],
        #                               posthoc=1,
        #                               all_updated_embeddings=all_updated_embeddings)

    print("------ PPD Test Results -----")
    # print(results_test)

    return bias_scores_of_user_interactions, Recmodel


def compute_d_pop_all_normalized(emb_u, emb_i, num_users, num_items, bias_scores_of_user_interactions,
                                 user_interactions, phi):
    device = emb_u.device

    d_pop_u = torch.zeros_like(emb_u)
    d_pop_i = torch.zeros_like(emb_i)

    # ---------- Compute d_pop_u ----------
    for u in range(num_users):
        item_ids = user_interactions[u]
        biases = bias_scores_of_user_interactions[u]

        item_embs = emb_i[item_ids]  # [num_items_u, d]
        bias_tensor = torch.tensor(biases, dtype=emb_i.dtype, device=device).unsqueeze(1)
        one_minus_bias = 1.0 - bias_tensor

        sum_bias = bias_tensor.sum()
        sum_one_minus_bias = one_minus_bias.sum()

        if sum_bias.item() == 0 or sum_one_minus_bias.item() == 0:
            continue

        e_pop_u = (item_embs * bias_tensor).sum(dim=0) / sum_bias
        e_pref_u = (item_embs * one_minus_bias).sum(dim=0) / sum_one_minus_bias

        diff = e_pop_u - phi * e_pref_u
        norm = diff.norm(p=2)

        if norm > 0:
            d_pop_u[u] = diff / norm

    # ---------- Compute d_pop_i ----------
    item_interactions = defaultdict(list)

    for u in range(num_users):
        for idx, i in enumerate(user_interactions[u]):
            b_ui = bias_scores_of_user_interactions[u][idx]
            item_interactions[i].append((u, b_ui))

    for i in range(num_items):
        interactions = item_interactions[i]
        if not interactions:
            continue

        user_ids, biases = zip(*interactions)
        user_ids = list(user_ids)
        biases = torch.tensor(biases, dtype=emb_u.dtype, device=device).unsqueeze(1)
        one_minus_bias = 1.0 - biases

        user_embs = emb_u[user_ids]

        sum_bias = biases.sum()
        sum_one_minus_bias = one_minus_bias.sum()

        if sum_bias.item() == 0 or sum_one_minus_bias.item() == 0:
            continue

        e_pop_i = (user_embs * biases).sum(dim=0) / sum_bias
        e_pref_i = (user_embs * one_minus_bias).sum(dim=0) / sum_one_minus_bias

        diff = e_pop_i - phi * e_pref_i
        norm = diff.norm(p=2)

        if norm > 0:
            d_pop_i[i] = diff / norm

    return d_pop_u, d_pop_i


def project_and_remove(embeddings, d_pops):
    """
    Remove the projection of d_pop from each embedding vector.

    Args:
        embeddings (torch.Tensor): Tensor of shape (N, D), where N is the number of users/items and D is the embedding dim.
        d_pops (torch.Tensor): Tensor of shape (N, D), the popularity direction vectors per user/item.

    Returns:
        torch.Tensor: Debiased embeddings of shape (N, D)
    """
    # Normalize direction vectors (avoid division by zero)
    d_norm_sq = torch.sum(d_pops ** 2, dim=1, keepdim=True) + 1e-8

    # Compute projection component
    projection = (torch.sum(embeddings * d_pops, dim=1, keepdim=True) / d_norm_sq) * d_pops

    # Subtract the projection from the original embeddings
    debiased_embeddings = embeddings - projection

    return debiased_embeddings


def compute_d_pop_all_normalized_1(emb_u, emb_i, num_users, num_items,
                                 bias_scores_of_user_interactions,
                                 user_interactions,
                                 dynamic_pref="history", c=5, lam=2.0):
    """
    Compute d_pop_u and d_pop_i with optional dynamic preference coefficients.

    Args:
        emb_u: user embedding matrix
        emb_i: item embedding matrix
        bias_scores_of_user_interactions: list of lists of b_ui
        user_interactions: list of lists of item indices
        dynamic_pref (str): one of {"none", "history", "variance", "bias"}
        c: smoothing constant for "history" mode
        lam: scaling factor for "bias" mode
    """
    device = emb_u.device
    d_pop_u = torch.zeros_like(emb_u)
    d_pop_i = torch.zeros_like(emb_i)

    # ---------- Compute d_pop_u ----------
    for u in range(num_users):
        item_ids = user_interactions[u]
        biases = bias_scores_of_user_interactions[u]
        if not item_ids:
            continue

        item_embs = emb_i[item_ids]
        bias_tensor = torch.tensor(biases, dtype=emb_i.dtype, device=device).unsqueeze(1)
        one_minus_bias = 1.0 - bias_tensor

        sum_bias = bias_tensor.sum()
        sum_one_minus_bias = one_minus_bias.sum()
        if sum_bias.item() == 0 or sum_one_minus_bias.item() == 0:
            continue

        e_pop_u = (item_embs * bias_tensor).sum(dim=0) / sum_bias
        e_pref_u = (item_embs * one_minus_bias).sum(dim=0) / sum_one_minus_bias

        # ---- Dynamic preference coefficient α_u ----
        if dynamic_pref == "history":
            alpha_u = len(item_ids) / (len(item_ids) + c)
        elif dynamic_pref == "variance":
            p_vals = [bias_scores_of_user_interactions[u][idx] for idx in range(len(item_ids))]
            alpha_u = np.var(p_vals) / (np.var(p_vals) + 1e-6)
        elif dynamic_pref == "bias":
            mean_bias = float(np.mean(biases))
            alpha_u = np.tanh(lam * mean_bias)
        else:
            alpha_u = 1.0  # fixed case (original)

        diff = e_pop_u - alpha_u * e_pref_u
        norm = diff.norm(p=2)
        if norm > 0:
            d_pop_u[u] = diff / norm

    # ---------- Compute d_pop_i ----------
    item_interactions = defaultdict(list)
    for u in range(num_users):
        for idx, i in enumerate(user_interactions[u]):
            b_ui = bias_scores_of_user_interactions[u][idx]
            item_interactions[i].append((u, b_ui))

    for i in range(num_items):
        interactions = item_interactions[i]
        if not interactions:
            continue

        user_ids, biases = zip(*interactions)
        user_embs = emb_u[list(user_ids)]
        biases = torch.tensor(biases, dtype=emb_u.dtype, device=device).unsqueeze(1)
        one_minus_bias = 1.0 - biases

        sum_bias = biases.sum()
        sum_one_minus_bias = one_minus_bias.sum()
        if sum_bias.item() == 0 or sum_one_minus_bias.item() == 0:
            continue

        e_pop_i = (user_embs * biases).sum(dim=0) / sum_bias
        e_pref_i = (user_embs * one_minus_bias).sum(dim=0) / sum_one_minus_bias

        # Here we just use global α (can extend per-item if needed)
        diff = e_pop_i - e_pref_i
        norm = diff.norm(p=2)
        if norm > 0:
            d_pop_i[i] = diff / norm

    return d_pop_u, d_pop_i