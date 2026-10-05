from collections import defaultdict

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

import Procedure, world
from constants import VARIANT_EMBEDDING_UPDATE, VARIANT_TRACK_AND_PRUNE, \
    VARIANT_VECTOR_BIAS_ZERO_LAYER_UPDATE, VARIANT_VECTOR_BIAS_ALL_LAYER_UPDATE
from register import dataset


# def compute_global_popularity_score(user_embeddings, item_embeddings, similarity):
#     """
#     user_embeddings: Tensor of shape (num_users, embed_dim)
#     item_embeddings: Tensor of shape (num_items, embed_dim)
#     similarity: 'dot' or 'cosine'
#     """
#     num_users = len(user_embeddings)
#     popularity_scores = []
#
#     for i, e_i in enumerate(item_embeddings):  # for each item
#         score_sum = 0.0
#         for u, e_u in enumerate(user_embeddings):  # for each user
#             if similarity == 'dot':
#                 sim = torch.dot(e_u, e_i)
#             elif similarity == 'cosine':
#                 sim = F.cosine_similarity(e_u.unsqueeze(0), e_i.unsqueeze(0)).item()
#             else:
#                 raise ValueError("Unsupported similarity")
#
#             # sqrt_degree = torch.sqrt(torch.tensor(len(user_interactions[u]), dtype=torch.float32) + 1e-8)
#             # score_sum += sim / sqrt_degree
#             score_sum += sim
#
#         p_i = score_sum / num_users
#         popularity_scores.append(p_i)
#
#     popularity_scores = torch.tensor(popularity_scores)
#
#     # Min-max scaling to [0, 1]
#     p_min = min(popularity_scores)
#     p_max = max(popularity_scores)
#     if p_max == p_min:
#         scaled_popularity_scores = [0.0 for _ in popularity_scores]
#     else:
#         scaled_popularity_scores = [(x - p_min) / (p_max - p_min) for x in popularity_scores]
#
#     # Bin counts: how many items fall in each range of score
#     # bins = np.linspace(min(scaled_popularity_scores), max(scaled_popularity_scores), 21)  # 20 bins from 0 to 1
#     # hist, bin_edges = np.histogram(scaled_popularity_scores, bins=bins)
#     #
#     # # Plot
#     # bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
#     # plt.figure(figsize=(8, 4))
#     # plt.bar(bin_centers, hist,
#     #         width=(max(scaled_popularity_scores) - min(scaled_popularity_scores)) / len(bins),
#     #         color='royalblue', edgecolor='black')
#     # plt.title("Distribution of Scaled Popularity Scores")
#     # plt.xlabel("Scaled Popularity Score (0–1)")
#     # plt.ylabel("Number of Items")
#     # plt.grid(True, axis='y', linestyle='--')
#     # plt.tight_layout()
#     # plt.show()
#
#     return scaled_popularity_scores


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
        items_u = user_interactions[u].tolist()
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


def get_popular_item_ids(popularity_scores, top_k_percent=0.20):
    """
    Returns the indices of top-k% most popular items.

    Args:
        popularity_scores (list or tensor): normalized popularity scores (length = num_items)
        top_k_percent (float): e.g., 0.10 means top 10%

    Returns:
        List[int]: item indices considered 'popular'
    """
    num_items = len(popularity_scores)
    top_k = int(num_items * top_k_percent)

    # Convert to tensor if needed
    if not isinstance(popularity_scores, torch.Tensor):
        popularity_scores = torch.tensor(popularity_scores)

    # Sort indices in descending order of popularity
    _, sorted_indices = torch.sort(popularity_scores, descending=True)

    # Select top-k item indices
    popular_item_ids = sorted_indices[:top_k].tolist()
    return popular_item_ids


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


def plot_bias_score_histogram(bias_scores, root_path):
    """
    Plots a histogram of bias scores (flattened from 2D list).

    Args:
        bias_scores (List[List[float]]): bias_scores[u] = list of b_ui for i in N_u
        bins (int): number of bins in histogram
    """
    # Flatten scores
    all_b_ui = [score for user_scores in bias_scores for score in user_scores]

    # Set number of bins
    bins = 20  # or any value you need

    # Plot histogram with gaps
    plt.figure(figsize=(14, 10), frameon=True)
    plt.hist(
        all_b_ui,
        bins=bins,
        color='#1f77b4',
        edgecolor='#f0f8ff',
        weights=np.ones_like(all_b_ui) / len(all_b_ui) * 100,
        histtype='bar',  # default, but good to specify
        rwidth=0.98  # set < 1.0 to create gaps between bars
    )

    # Labels and formatting
    plt.xlabel("Popularity bias score (normalized)", fontsize=36)
    plt.ylabel("Percentage of interactions (%)", fontsize=36)
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.tight_layout()
    plt.savefig(root_path + "/pop_bias_scores.pdf")


 # per-entity eta = eta0 * sigmoid(log(deg+1))
    def eta_from_deg(deg, eta0=0.3):
        return eta0 * torch.sigmoid(torch.log1p(deg))


def compute_pop_bias_scores(root_path, user_interactions, Recmodel,
                            dataloader, best_epoch, variant,
                            beta, phi, similarity='dot'):
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
    user_emb_weight = Recmodel.embedding_user.weight
    item_emb_weight = Recmodel.embedding_item.weight
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
        Recmodel.Graph = dataloader.getSparseGraphWithWeight(bias_scores_of_user_interactions)
        results_test = Procedure.Test(dataset, Recmodel, best_epoch, None, world.config['multicore'])
    elif variant == VARIANT_VECTOR_BIAS_ZERO_LAYER_UPDATE:
        d_pop_u, d_pop_i = compute_d_pop_all_normalized(user_emb_weight, item_emb_weight, len(user_emb_weight),
                          len(item_emb_weight), bias_scores_of_user_interactions, user_interactions, phi)
        user_emb_weight = project_and_remove(user_emb_weight, d_pop_u)
        item_emb_weight = project_and_remove(item_emb_weight, d_pop_i)

        # degrees: uncomment for dynamic update of embeddings
        # deg_u = torch.tensor([len(li) for li in user_interactions], device=user_emb_weight.device, dtype=user_emb_weight.dtype)
        # deg_i = torch.zeros(len(item_emb_weight), device=item_emb_weight.device, dtype=item_emb_weight.dtype)
        # for u, items in enumerate(user_interactions):
        #     deg_i[items] += 1

        with torch.no_grad():
            Recmodel.embedding_user.weight.copy_(user_emb_weight)
            Recmodel.embedding_item.weight.copy_(item_emb_weight)
            # uncomment for fixed update of embeddings with previous embeddings
            # eta = 0.3  # Yahoo likes small eta (0.1–0.3)
            # Recmodel.embedding_user.weight.mul_(eta).add_((1 - eta) * user_emb_weight)
            # Recmodel.embedding_item.weight.mul_(eta).add_((1 - eta) * item_emb_weight)

        # uncomment for dynamic update of embeddings
        # with torch.no_grad():
        #     U1 = Recmodel.embedding_user.weight
        #     I1 = Recmodel.embedding_item.weight
        #     eta_u = eta_from_deg(deg_u).unsqueeze(1)
        #     eta_i = eta_from_deg(deg_i).unsqueeze(1)
        #     U1.copy_((1 - eta_u) * user_emb_weight + eta_u * U1)
        #     I1.copy_((1 - eta_i) * item_emb_weight + eta_i * I1)

        results_test = Procedure.Test(dataset, Recmodel, best_epoch, None, world.config['multicore'],
                                      posthoc=0, all_updated_embeddings=None,
                                      dataset_name=world.dataset, model_name="ppd")
    elif variant == VARIANT_VECTOR_BIAS_ALL_LAYER_UPDATE:
        Recmodel.bias_scores_of_user_interactions = bias_scores_of_user_interactions
        Recmodel.user_interactions = user_interactions
        Recmodel.set_item_to_users()
        all_updated_embeddings = Recmodel.forward_post_hoc_with_bias_vector_estimation(compute_d_pop_all_normalized,
                                                                                       phi)

        results_test = Procedure.Test(dataset, Recmodel, best_epoch, None, world.config['multicore'],
                                      posthoc=1,
                                      all_updated_embeddings=all_updated_embeddings)
    else:
        a_k = [1.0, 0.8, 0.6]
        T = 0.10
        K = 2
        all_updated_embeddings = whole_pipeline_of_pop_debiasing(dataloader, user_interactions,
                                        bias_scores_of_user_interactions,
                                        Recmodel, K, a_k, T, user_emb_weight, item_emb_weight)
        results_test = Procedure.Test(dataset, Recmodel, best_epoch, None, world.config['multicore'],
                                      posthoc=0, all_updated_embeddings=None)
        # use it if using all k-hop embeddings update
        results_test = Procedure.Test(dataset, Recmodel, best_epoch, None, world.config['multicore'],
                                      posthoc=1, all_updated_embeddings=all_updated_embeddings)

    print("------ PPD Test Results -----")
    print(results_test)

    return bias_scores_of_user_interactions


def init_influence_tracking_algorithm(num_layers, user_interactions, bias_scores_of_user_interactions):
    """
        Computes popularity bias score.

        Args:
            num_layers = layers of GCN
            a_k = [...]  # pruning ratio for each k
            T = threshold for filtering influence
            user_interactions (List[List[int]]): user_interactions[u] = list of item indices
            bias_scores_of_user_interactions (List[List[int]]): bias_scores_of_user_interactions[u] =
                                                    list of bias scores in the sequence of user_interactions[u]

        Returns:
            # List[List[float]]: bias_scores[u] = list of b_ui for i in user_interactions[u]
        """

    Dr = []  # biased interactions
    for u, items in enumerate(user_interactions):
        for j, i in enumerate(items):
            if bias_scores_of_user_interactions[u][j] > 0:
                Dr.append((u, i))

    # Parameters
    K = num_layers

    s_k = [defaultdict(float) for _ in range(K + 1)]
    alpha_k = [defaultdict(dict) for _ in range(K + 1)]
    I_k = [defaultdict(set) for _ in range(K + 1)]
    S_k = [set() for _ in range(K + 1)]

    return s_k, S_k, I_k, alpha_k, Dr


def prune_and_filter(S_k_layer, s_k_layer, I_k_layer, alpha_k_layer, a_k_val, T):
    top_nodes = sorted(S_k_layer, key=lambda v: s_k_layer[v], reverse=True)
    top_nodes = top_nodes[:int(len(top_nodes) * a_k_val)]
    pruned = set(top_nodes)
    for v in pruned:
        I_k_layer[v] = {(u, i) for (u, i) in I_k_layer[v] if alpha_k_layer[v][(u, i)] >= T}
    return pruned


def pruning_and_tracking(Dr, graph, num_users, s_k, S_k, I_k, alpha_k, a_k, T, K):
    for u, i in Dr:
        for v in [u, i]:
            # deg_v = (graph[v])
            i_with_offset = i + num_users
            if v == u:
                deg_v = sum(1 for x in graph[v] if x != 0)
            else:
                deg_v = sum(1 for x in graph[i_with_offset] if x != 0)
            if v == u:
                s_k[0][v] += 1 / deg_v
                S_k[0].add(v)
                I_k[0][v].add((u, i))
                alpha_k[0][v][(u, i)] = alpha_k[0][v].get((u, i), 0.0) + 1 / deg_v
            else:
                s_k[0][i_with_offset] += 1 / deg_v
                S_k[0].add(i_with_offset)
                I_k[0][i_with_offset].add((u, i))
                alpha_k[0][i_with_offset][(u, i)] = alpha_k[0][i_with_offset].get((u, i), 0.0) + 1 / deg_v

    S_k[0] = prune_and_filter(S_k[0], s_k[0], I_k[0], alpha_k[0], a_k[0], T)
    # print(len(S_k[0]))

    for k in range(1, K + 1):
        for v in S_k[k - 1]:
            # print(graph[v].nonzero()[1])
            indices = graph.indices()
            mask = indices[0] == v
            neighbors_of_v = indices[1][mask]
            # print(neighbors_of_v)
            for v_prime in neighbors_of_v:
                # print(v_prime)
                deg_v_prime = sum(1 for x in graph[v_prime] if x != 0)
                # print(deg_v_prime)
                s_k[k][v_prime] += s_k[k - 1][v] * (1 / deg_v_prime)
                S_k[k].add(v_prime)
                for (u, i) in I_k[k - 1][v]:
                    I_k[k][v_prime].add((u, i))
                    alpha_k[k][v_prime][(u, i)] = alpha_k[k - 1][v][(u, i)] * (1 / deg_v_prime)

        S_k[k] = prune_and_filter(S_k[k], s_k[k], I_k[k], alpha_k[k], a_k[k], T)

    return S_k, I_k, alpha_k, s_k


def update_embeddings_for_pop_bias_correction(user_interactions, Recmodel, bias_scores_of_user_interactions,
                                              K, S_k, I_k, alpha_k, user_emb_weight, item_emb_weight):
    # users_emb = Recmodel.embedding_user.weight
    # items_emb = Recmodel.embedding_item.weight
    all_emb = Recmodel.get_all_layer_embeddings(user_emb_weight, item_emb_weight)
    for k in range(K + 1):
        for v in S_k[k]:
            influence = 0.0
            for (u, i) in I_k[k][v]:
                # j = user_interactions[u].index(i)  # get index j in b_ui
                j = np.where(user_interactions[u] == i)[0][0]
                influence += bias_scores_of_user_interactions[u][j].item() * alpha_k[k][v][(u, i)]
            # if influence > 0:
            #     print(influence)
            influence = min(max(influence, 0.0), 1.0)
            all_emb[v][k] = all_emb[v][k] * (1 - influence)
    print('ended update_embeddings_for_pop_bias_correction')
    return all_emb


def update_embeddings_for_pop_bias_correction_1(user_interactions, Recmodel, bias_scores_of_user_interactions,
                                                  K, S_k, I_k, alpha_k, user_emb_weight, item_emb_weight, s_k):
    # Set of all affected nodes (union of S_k)
    # V_r = set().union(*S_k)
    V_r = S_k[0]

    all_emb = Recmodel.get_all_layer_embeddings(user_emb_weight, item_emb_weight)
    new_all_emb = all_emb.detach().clone()

    # For each affected node, compute total influence across all layers
    for v in V_r:
        total_influence = 0.0
        for k in range(K + 1):
            if v not in I_k[k]:
                continue
            for (u, i) in I_k[k][v]:
                j = np.where(user_interactions[u] == i)[0][0]
                total_influence += bias_scores_of_user_interactions[u][j].item() * alpha_k[k][v][(u, i)]

        # if total_influence > 0:
            # print(total_influence)
        # Apply forgetting only to the 0-th layer embedding
        total_influence = min(max(total_influence, 0.0), 1.0)
        new_all_emb[v][0] = new_all_emb[v][0] * (1 - total_influence)

    # new_all_emb[:][0] = nn.functional.normalize(new_all_emb[:][0])

    with torch.no_grad():
        # Recmodel.embedding_user.weight.copy_(all_emb[:Recmodel.num_users][0])
        # Recmodel.embedding_item.weight.copy_(all_emb[Recmodel.num_users:][0])
        Recmodel.embedding_user.weight.copy_(new_all_emb[:Recmodel.num_users, 0])
        Recmodel.embedding_item.weight.copy_(new_all_emb[Recmodel.num_users:, 0])
    print('ended update_embeddings_for_pop_bias_correction')

    test_forgetting_correctness(user_interactions, all_emb, new_all_emb)
    return new_all_emb


def whole_pipeline_of_pop_debiasing(dataloader, user_interactions, bias_scores_of_user_interactions,
                                    Recmodel, K, a_k, T, user_emb_weight, item_emb_weight):
    s_k, S_k, I_k, alpha_k, Dr = init_influence_tracking_algorithm(K,
                                      user_interactions, bias_scores_of_user_interactions)
    S_k, I_k, alpha_k, s_k = pruning_and_tracking(Dr, dataloader.getOnlyGraphParams(), len(user_interactions),
                                                  s_k, S_k, I_k, alpha_k, a_k, T, K)
    return update_embeddings_for_pop_bias_correction_1(user_interactions, Recmodel, bias_scores_of_user_interactions,
                                              K, S_k, I_k, alpha_k, user_emb_weight, item_emb_weight, s_k)


def test_forgetting_correctness(user_interactions, all_emb_before, new_all_emb):
    u = 10
    i = user_interactions[u][0]
    v = u  # test for user node
    before = all_emb_before[v][0].clone()
    after = new_all_emb[v][0]
    influence = torch.norm(before - after).item()
    print(f"Embedding changed by: {influence}")


def compute_d_pop_all_normalized(emb_u, emb_i, num_users, num_items,
                                 bias_scores_of_user_interactions,
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

