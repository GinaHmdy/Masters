from collections import defaultdict
import time
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
    Exact equivalent to the original when sample_size <= 0 (i.e., use ALL users).
    Much faster and GPU-friendly (no .item() syncs).
    """
    if sample_size > 0:
        # keep your existing sampling behavior for later experiments
        # (paper replication uses sample_size=-1, so this won't trigger)
        num_users = len(user_embeddings)
        popularity_scores = []
        for i, e_i in enumerate(item_embeddings):
            g = torch.Generator(device=user_embeddings.device)
            g.manual_seed(i + 4200)
            if num_users > sample_size:
                indices = torch.randperm(num_users, generator=g, device=user_embeddings.device)[:sample_size]
                sampled_users = user_embeddings[indices]
            else:
                sampled_users = user_embeddings

            if similarity == "dot":
                sims = sampled_users @ e_i
            elif similarity == "cosine":
                sims = torch.nn.functional.cosine_similarity(sampled_users, e_i.unsqueeze(0), dim=1)
            else:
                raise ValueError("Unsupported similarity")

            popularity_scores.append(sims.mean())

        popularity_scores = torch.stack(popularity_scores)
    else:
        print("[PPD] compute_global_popularity_score: FAST PATH (no sampling)")  # in the sample_size<=0 branch
        # exact for your current PPD usage
        if similarity == "dot":
            mean_u = user_embeddings.mean(dim=0)              # [d]
            popularity_scores = item_embeddings @ mean_u      # [num_items]
        elif similarity == "cosine":
            mean_u = user_embeddings.mean(dim=0)
            mean_u = torch.nn.functional.normalize(mean_u, dim=0)
            popularity_scores = item_embeddings @ mean_u
        else:
            raise ValueError("Unsupported similarity")

    # min-max to [0,1]
    p_min, p_max = popularity_scores.min(), popularity_scores.max()
    if (p_max - p_min).abs() < 1e-12:
        return torch.zeros_like(popularity_scores)
    return (popularity_scores - p_min) / (p_max - p_min)


def compute_personalized_preference_scores(
    item_embeddings, user_interactions, popularity_score_of_items,
    similarity, gamma
):
    """
    EXACT same definition as your loop version, but GPU-friendly.

    Old code does for each user u:
      r_ui = (1/|Nu|) * sum_{j in Nu} ( sigmoid(e_i^T e_j) - gamma * p_i * p_j )

    This version computes the same using matrix ops per user.
    Returns list-of-lists like before, so the rest of the pipeline stays identical.
    """
    device = item_embeddings.device
    if not isinstance(popularity_score_of_items, torch.Tensor):
        popularity_score_of_items = torch.tensor(popularity_score_of_items, dtype=torch.float32, device=device)
    else:
        popularity_score_of_items = popularity_score_of_items.to(device)    
    pop = popularity_score_of_items.to(device)  # [num_items] tensor

    rui_scores = []
    for u in range(len(user_interactions)):
        items_u = user_interactions[u]
        if hasattr(items_u, "numel"):
            if items_u.numel() == 0:
                rui_scores.append([])
                continue
            items_u = items_u.to(device)
        else:
            if len(items_u) == 0:
                rui_scores.append([])
                continue
            items_u = torch.tensor(items_u, device=device, dtype=torch.long)

        E = item_embeddings[items_u]  # [m,d]

        if similarity == "dot":
            sims = torch.sigmoid(E @ E.t())  # [m,m]  (matches your sigmoid(dot))
        elif similarity == "cosine":
            En = torch.nn.functional.normalize(E, dim=1)
            sims = En @ En.t()
        else:
            raise ValueError("Unsupported similarity type")

        p = pop[items_u]  # [m]
        penalty = gamma * (p[:, None] * p[None, :])  # [m,m]

        r = (sims - penalty).mean(dim=1)  # [m]
        rui_scores.append(r.detach().tolist())

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
            b_ui = float(popularity_score_of_items[i]) - float(r_ui)
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
    Computes popularity bias score (PPD pipeline).

    Notes:
    - Adds detailed timing logs for each stage.
    - Keeps math unchanged.
    - Ensures no tensor/list mixing issues inside bias score computation by converting
      popularity scores to CPU floats once.
    """
    import time

    print("[PPD] calling compute_global_popularity_score with sample_size=-1")

    # 1) Get normalized embeddings (same as before)
    user_emb_weight = nn.functional.normalize(Recmodel.embedding_user.weight)
    item_emb_weight = nn.functional.normalize(Recmodel.embedding_item.weight)

    print("[PPD] device user:", Recmodel.embedding_user.weight.device)
    print("[PPD] device item:", Recmodel.embedding_item.weight.device)
    print("[PPD] device Graph:", Recmodel.Graph.device if hasattr(Recmodel, "Graph") else "N/A")

    penalty_coefficient_in_personalized_scores = beta

    # 2) Global popularity scores (vectorized fast path already in your function)
    t0 = time.time()
    popularity_score_of_items = compute_global_popularity_score(
        user_emb_weight, item_emb_weight, similarity, sample_size=-1
    )
    print("[PPD] popularity time:", time.time() - t0)

    # Convert pop scores once to CPU floats for downstream list-based code
    # (keeps behavior consistent with your original .item() usage)
    pop_cpu = popularity_score_of_items.detach().float().cpu().numpy().tolist()

    # 3) Personalized preference scores (rui)
    t1 = time.time()
    rui_scores_of_user_interactions = compute_personalized_preference_scores(
        item_emb_weight,
        user_interactions,
        pop_cpu,  # pass CPU float list so penalty and bias are pure Python floats
        similarity,
        penalty_coefficient_in_personalized_scores
    )
    print("[PPD] rui time:", time.time() - t1)

    # 4) Normalize rui
    t2 = time.time()
    rui_scores_of_user_interactions = normalize_scores_global(rui_scores_of_user_interactions)
    print("[PPD] normalize rui time:", time.time() - t2)

    # 5) Bias scores + normalize
    t3 = time.time()
    bias_scores_of_user_interactions = compute_bias_scores(
        pop_cpu,
        rui_scores_of_user_interactions,
        user_interactions
    )
    bias_scores_of_user_interactions = normalize_scores_global(bias_scores_of_user_interactions)
    print("[PPD] bias compute+normalize time:", time.time() - t3)

    # 6) Apply the chosen PPD variant
    if variant == VARIANT_EMBEDDING_UPDATE:
        Recmodel.Graph = dataloader.getSparseGraphWithWeight(bias_scores_of_user_interactions)
        results_test = Procedure.Test(dataset, Recmodel, best_epoch, None, world.config['multicore'])

    elif variant == VARIANT_VECTOR_BIAS_ZERO_LAYER_UPDATE:
        # Time the d_pop step explicitly (often the next bottleneck)
        t4 = time.time()
        d_pop_u, d_pop_i = compute_d_pop_all_normalized(
            user_emb_weight, item_emb_weight,
            len(user_emb_weight), len(item_emb_weight),
            bias_scores_of_user_interactions,
            user_interactions,
            phi
        )
        print("[PPD] d_pop time:", time.time() - t4)

        user_emb_new = project_and_remove(user_emb_weight, d_pop_u)
        item_emb_new = project_and_remove(item_emb_weight, d_pop_i)

        with torch.no_grad():
            Recmodel.embedding_user.weight.copy_(user_emb_new)
            Recmodel.embedding_item.weight.copy_(item_emb_new)

        results_test = Procedure.Test(
            dataset, Recmodel, best_epoch, None, world.config['multicore'],
            posthoc=0, all_updated_embeddings=None,
            dataset_name=world.dataset, model_name="ppd"
        )

    elif variant == VARIANT_VECTOR_BIAS_ALL_LAYER_UPDATE:
        Recmodel.bias_scores_of_user_interactions = bias_scores_of_user_interactions
        Recmodel.user_interactions = user_interactions
        Recmodel.set_item_to_users()

        t4 = time.time()
        all_updated_embeddings = Recmodel.forward_post_hoc_with_bias_vector_estimation(
            compute_d_pop_all_normalized, phi
        )
        print("[PPD] all-layer update time:", time.time() - t4)

        results_test = Procedure.Test(
            dataset, Recmodel, best_epoch, None, world.config['multicore'],
            posthoc=1, all_updated_embeddings=all_updated_embeddings
        )

    else:
        a_k = [1.0, 0.8, 0.6]
        T = 0.10
        K = 2

        t4 = time.time()
        all_updated_embeddings = whole_pipeline_of_pop_debiasing(
            dataloader, user_interactions,
            bias_scores_of_user_interactions,
            Recmodel, K, a_k, T,
            user_emb_weight, item_emb_weight
        )
        print("[PPD] whole pipeline time:", time.time() - t4)

        results_test = Procedure.Test(
            dataset, Recmodel, best_epoch, None, world.config['multicore'],
            posthoc=0, all_updated_embeddings=None
        )
        results_test = Procedure.Test(
            dataset, Recmodel, best_epoch, None, world.config['multicore'],
            posthoc=1, all_updated_embeddings=all_updated_embeddings
        )

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

