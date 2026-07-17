import sys

from torch import utils

sys.path.append('../')
import math

import numpy as np
import torch
import world
from dataloader import BasicDataset
from sklearn.cluster import KMeans
from torch import nn
# from utils import cos_similarity
from multiprocessing.dummy import Pool as ThreadPool
from model import BasicModel

import torch
import torch.nn.functional as F


def cos_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two embedding tensors.

    Args:
        x: Tensor of shape (N, D)
        y: Tensor of shape (M, D)

    Returns:
        Tensor of shape (N, M) with pairwise cosine similarities.
    """
    # Normalize embeddings to unit length
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)

    # Matrix multiply to get cosine similarity
    # return torch.mm(x_norm, y_norm.t()) # previous one
    return torch.sum(x_norm * y_norm, dim=1)  # (590,) # new one

class LightGCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        
        # dynamic ego attention params
        if self.config.get('dynamic_ego', False):
            self.W_att = nn.Parameter(torch.randn(self.latent_dim, 16))
            self.w_att = nn.Parameter(torch.randn(16, 1))
            nn.init.xavier_uniform_(self.W_att)
            nn.init.xavier_uniform_(self.w_att)

        if self.config.get('use_llm_init', False):
            import os
            dataset_path = self.dataset.path
            user_file = os.path.join(dataset_path, "user_llm_init.npy")
            item_file = os.path.join(dataset_path, "item_llm_init.npy")
            loaded = False
            if os.path.exists(user_file) and os.path.exists(item_file):
                try:
                    user_init = np.load(user_file)
                    item_init = np.load(item_file)
                    if user_init.shape == (self.num_users, self.latent_dim) and item_init.shape == (self.num_items, self.latent_dim):
                        self.embedding_user.weight.data.copy_(torch.from_numpy(user_init))
                        self.embedding_item.weight.data.copy_(torch.from_numpy(item_init))
                        print(f"[LLM Init] Successfully loaded LLM initialization from {dataset_path}")
                        loaded = True
                    else:
                        print(f"[LLM Init] Warning: shape mismatch: user file {user_init.shape} vs expected {(self.num_users, self.latent_dim)}, item file {item_init.shape} vs expected {(self.num_items, self.latent_dim)}")
                except Exception as e:
                    print(f"[LLM Init] Error loading LLM files: {e}")
            if not loaded:
                print(f"[LLM Init] Warning: LLM initialization files not found or invalid at '{dataset_path}'. Falling back to standard normal initialization.")
                nn.init.normal_(self.embedding_user.weight, std=0.1)
                nn.init.normal_(self.embedding_item.weight, std=0.1)
                print('use NORMAL distribution initilizer')
        elif self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            print('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        device = values.device
        random_index = torch.rand(len(values), device=device) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse_coo_tensor(index.t(), values, size, device=device)
        return g

    def apply_dynamic_ego(self, embs):
        N, L_plus_1, D = embs.shape
        flat_embs = embs.view(-1, D)
        att_features = torch.tanh(torch.matmul(flat_embs, self.W_att))
        scores = torch.matmul(att_features, self.w_att).view(N, L_plus_1, 1)
        att_weights = torch.softmax(scores, dim=1)
        light_out = torch.sum(embs * att_weights, dim=1)
        return light_out

    def sgl_epoch_init(self):
        self._g1 = self.__dropout_x(self.Graph, 0.9)
        self._g2 = self.__dropout_x(self.Graph, 0.9)

    def sgl_loss(self, users, pos):
        if not hasattr(self, '_g1'):
            self.sgl_epoch_init()
        g1 = self._g1
        g2 = self._g2
        
        all_emb_1 = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        embs1 = [all_emb_1]
        for _ in range(self.n_layers):
            all_emb_1 = torch.sparse.mm(g1, all_emb_1)
            embs1.append(all_emb_1)
        embs1 = torch.stack(embs1, dim=1)
        
        if self.config.get('dynamic_ego', False):
            light_out1 = self.apply_dynamic_ego(embs1)
        else:
            ego_weight = self.config.get('ego_weight', -1.0)
            if ego_weight >= 0.0:
                weights = [ego_weight] + [(1.0 - ego_weight) / self.n_layers] * self.n_layers
                w_tensor = torch.tensor(weights, device=embs1.device).view(1, -1, 1)
                light_out1 = torch.sum(embs1 * w_tensor, dim=1)
            else:
                light_out1 = torch.mean(embs1, dim=1)
        users_emb_1, items_emb_1 = torch.split(light_out1, [self.num_users, self.num_items])
        
        all_emb_2 = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        embs2 = [all_emb_2]
        for _ in range(self.n_layers):
            all_emb_2 = torch.sparse.mm(g2, all_emb_2)
            embs2.append(all_emb_2)
        embs2 = torch.stack(embs2, dim=1)
        
        if self.config.get('dynamic_ego', False):
            light_out2 = self.apply_dynamic_ego(embs2)
        else:
            ego_weight = self.config.get('ego_weight', -1.0)
            if ego_weight >= 0.0:
                weights = [ego_weight] + [(1.0 - ego_weight) / self.n_layers] * self.n_layers
                w_tensor = torch.tensor(weights, device=embs2.device).view(1, -1, 1)
                light_out2 = torch.sum(embs2 * w_tensor, dim=1)
            else:
                light_out2 = torch.mean(embs2, dim=1)
        users_emb_2, items_emb_2 = torch.split(light_out2, [self.num_users, self.num_items])
        
        u_emb1 = nn.functional.normalize(users_emb_1[users], dim=1)
        u_emb2 = nn.functional.normalize(users_emb_2[users], dim=1)
        pos_u = torch.sum(u_emb1 * u_emb2, dim=1)
        ttl_score_u = torch.matmul(u_emb1, u_emb2.t())
        temp = 0.2
        numerator_u = torch.exp(pos_u / temp)
        denominator_u = torch.sum(torch.exp(ttl_score_u / temp), dim=1)
        ssl_loss_u = -torch.log(numerator_u / (denominator_u + 1e-8) + 1e-8).mean()
        
        i_emb1 = nn.functional.normalize(items_emb_1[pos], dim=1)
        i_emb2 = nn.functional.normalize(items_emb_2[pos], dim=1)
        pos_i = torch.sum(i_emb1 * i_emb2, dim=1)
        ttl_score_i = torch.matmul(i_emb1, i_emb2.t())
        numerator_i = torch.exp(pos_i / temp)
        denominator_i = torch.sum(torch.exp(ttl_score_i / temp), dim=1)
        ssl_loss_i = -torch.log(numerator_i / (denominator_i + 1e-8) + 1e-8).mean()
        
        return ssl_loss_u + ssl_loss_i

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        ego_weight = self.config.get('ego_weight', -1.0)
        if self.config.get('dynamic_ego', False):
            light_out = self.apply_dynamic_ego(embs)
        elif ego_weight >= 0.0:
            weights = [ego_weight] + [(1.0 - ego_weight) / self.n_layers] * self.n_layers
            w_tensor = torch.tensor(weights, device=embs.device).view(1, -1, 1)
            light_out = torch.sum(embs * w_tensor, dim=1)
        else:
            light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def computer_DAP(self, weight1, weight2, num_cluster):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        u_degree = torch.tensor(self.dataset.users_D, device=world.device)
        i_degree = torch.tensor(self.dataset.items_D, device=world.device)
        degree = torch.cat([u_degree, i_degree], dim=0)
        all_embs_anchor_higher = torch.zeros([self.num_users + self.num_items, self.latent_dim]).to(world.device)
        all_embs_anchor_lower = torch.zeros([self.num_users + self.num_items, self.latent_dim]).to(world.device)
        embs = [all_emb]
        g_droped = self.Graph

        # weight1 = self.config['alpha1']
        # weight2 = self.config['alpha2']
        # num_cluster = self.config['cluster_num']
        # weight1 = 0.6
        # weight2 = 0.6
        # num_cluster = 5
        print(f'weight1={weight1},weight2={weight2}, num_cluster:{num_cluster}')
        for layer in range(self.n_layers):
            all_emb_i = torch.sparse.mm(g_droped, all_emb)
            kmeans = KMeans(n_clusters=num_cluster, random_state=9)
            print(f'GCN debiasing at {layer + 1}-layer')
            all_emb_cluster = kmeans.fit_predict(all_emb_i.to('cpu'))
            for k_cluster in range(num_cluster):
                index = np.where(all_emb_cluster == k_cluster)
                if len(index[0]) == 0:
                    continue
                idx_tensor = torch.tensor(index[0], dtype=torch.long, device=world.device)
                embs_cluster = all_emb_i[idx_tensor]
                degree_cluster = degree[idx_tensor].unsqueeze(1).to(world.device)

                sorted_degrees, sort_indices = torch.sort(degree_cluster.squeeze(1))
                sorted_embs = embs_cluster[sort_indices]
                
                prefix_sums = torch.cumsum(sorted_embs, dim=0)
                suffix_sums = torch.cumsum(sorted_embs.flip(dims=[0]), dim=0).flip(dims=[0])
                
                total_elements = len(sorted_degrees)
                
                lower_count = torch.searchsorted(sorted_degrees, sorted_degrees, side='left')
                first_greater_idx = torch.searchsorted(sorted_degrees, sorted_degrees, side='right')
                higher_count = total_elements - first_greater_idx
                
                lower_sum = torch.zeros_like(sorted_embs)
                valid_lower = lower_count > 0
                if valid_lower.any():
                    lower_sum[valid_lower] = prefix_sums[lower_count[valid_lower] - 1]
                
                higher_sum = torch.zeros_like(sorted_embs)
                valid_higher = first_greater_idx < total_elements
                if valid_higher.any():
                    higher_sum[valid_higher] = suffix_sums[first_greater_idx[valid_higher]]
                
                mean_lower = lower_sum / torch.clamp(lower_count.unsqueeze(1).float(), min=1.0)
                mean_higher = higher_sum / torch.clamp(higher_count.unsqueeze(1).float(), min=1.0)
                
                unsorted_indices = torch.argsort(sort_indices)
                mean_lower_unsorted = mean_lower[unsorted_indices]
                mean_higher_unsorted = mean_higher[unsorted_indices]
                lower_count_unsorted = lower_count[unsorted_indices]
                higher_count_unsorted = higher_count[unsorted_indices]
                
                all_embs_anchor_higher[idx_tensor] = torch.where(
                    (higher_count_unsorted > 0).unsqueeze(1),
                    mean_higher_unsorted,
                    all_embs_anchor_higher[idx_tensor]
                )
                all_embs_anchor_lower[idx_tensor] = torch.where(
                    (lower_count_unsorted > 0).unsqueeze(1),
                    mean_lower_unsorted,
                    all_embs_anchor_lower[idx_tensor]
                )
            all_embs_anchor_higher = nn.functional.normalize(all_embs_anchor_higher)
            all_embs_anchor_lower = nn.functional.normalize(all_embs_anchor_lower)
            alpha_sim_higher = cos_similarity(all_emb, all_embs_anchor_higher.to(world.device))
            alpha_sim_lower = cos_similarity(all_emb, all_embs_anchor_lower.to(world.device))
            # new one
            # all_emb = all_emb_i \
            #           - weight1 * torch.matmul(alpha_sim_higher, all_embs_anchor_higher) \
            #           - weight2 * torch.matmul(alpha_sim_lower, all_embs_anchor_lower)
            # old one
            all_emb = all_emb_i - weight1 * alpha_sim_higher.unsqueeze(1).to(world.device) * all_embs_anchor_higher.to(
                world.device) \
                      - weight2 * alpha_sim_lower.unsqueeze(1).to(world.device) * all_embs_anchor_lower.to(world.device)
            # all_emb = all_emb_i - weight1 * all_embs_anchor_higher.to(world.device) - weight2 * all_embs_anchor_lower.to(world.device)
            # all_emb = all_emb_i - weight * alpha_sim_lower.unsqueeze(1).to(world.device) * all_embs_anchor_lower.to(world.device)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        if self.config.get('dynamic_ego', False):
            light_out = self.apply_dynamic_ego(embs)
        else:
            light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def computer_one(self):
        print('one-order neighbors')
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        u_degree = torch.tensor(self.dataset.users_D)
        i_degree = torch.tensor(self.dataset.items_D)
        degree = torch.cat([u_degree, i_degree], dim=0)
        all_embs_anchor_higher = torch.zeros([self.num_users + self.num_items, self.latent_dim]).to(world.device)
        all_embs_anchor_lower = torch.zeros([self.num_users + self.num_items, self.latent_dim]).to(world.device)
        degree_higher_num = torch.zeros([self.num_users + self.num_items, 1]).to(world.device)
        degree_lower_num = torch.zeros([self.num_users + self.num_items, 1]).to(world.device)
        embs = [all_emb]

        nodes = [i for i in range(len(all_emb))]
        # weight1 = self.config['alpha1']
        # weight2 = self.config['alpha2']
        weight1 = 0.6
        weight2 = 0.6
        print(f'weight1={weight1}, weight2={weight2}')
        for layer in range(self.n_layers):
            print(f'GCN debiasing at {layer + 1}-layer')
            all_emb_i = torch.sparse.mm(self.Graph, all_emb)
            for node in nodes:
                neighbors = self.Graph[node]._indices()[0]
                if len(neighbors) > 0:
                    embs_cluster = all_emb_i[neighbors]
                    degree_cluster = degree[neighbors].unsqueeze(1)

                    if degree[node] > degree_cluster.min() and degree[node] < degree_cluster.max():
                        degree_higher_index = torch.where(degree_cluster > degree[node])
                        degree_higher_num[node] = degree_cluster[degree_higher_index[0]].sum()
                        degree_lower_index = torch.where(degree_cluster < degree[node])
                        degree_lower_num[node] = degree_cluster[degree_lower_index[0]].sum()
                        embs_higher_mean = embs_cluster[degree_higher_index[0]].mean(0)
                        embs_lower_mean = embs_cluster[degree_lower_index[0]].mean(0)
                        all_embs_anchor_higher[node] = embs_higher_mean
                        all_embs_anchor_lower[node] = embs_lower_mean
                    elif degree[node] == degree_cluster.min():
                        degree_higher_index = torch.where(degree_cluster > degree[node])
                        degree_higher_num[node] = degree_cluster[degree_higher_index[0]].sum()
                        embs_higher_mean = embs_cluster[degree_higher_index[0]].mean(0)
                        all_embs_anchor_higher[node] = embs_higher_mean
                    elif degree[node] == degree_cluster.max():
                        degree_lower_index = torch.where(degree_cluster < degree[node])
                        degree_lower_num[node] = degree_cluster[degree_lower_index[0]].sum()
                        embs_lower_mean = embs_cluster[degree_lower_index[0]].mean(0)
                        all_embs_anchor_lower[node] = embs_lower_mean
            all_embs_anchor_higher = nn.functional.normalize(all_embs_anchor_higher)
            all_embs_anchor_lower = nn.functional.normalize(all_embs_anchor_lower)
            alpha_sim_higher = cos_similarity(all_emb, all_embs_anchor_higher.to(world.device))
            alpha_sim_lower = cos_similarity(all_emb, all_embs_anchor_lower.to(world.device))
            degree_impact = degree_higher_num > degree_lower_num
            all_emb = all_emb_i - weight1 * alpha_sim_higher.unsqueeze(1).to(world.device) * all_embs_anchor_higher.to(
                world.device) \
                      - weight2 * alpha_sim_lower.unsqueeze(1).to(world.device) * all_embs_anchor_lower.to(world.device)
            # all_emb = all_emb_i - weight * alpha_sim_lower.unsqueeze(1).to(world.device) * all_embs_anchor_lower.to(world.device)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getUsersRating__1(self, all_users, all_items, users):
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        if self.config.get('sgl', False):
            sgl_loss_val = self.sgl_loss(users, pos)
            sgl_weight = self.config.get('sgl_weight', 0.1)
            loss = loss + sgl_weight * sgl_loss_val

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


    def getUsersRatingDAP(self, users, weight_1, weight_2, num_cluster):
        cache_key = (weight_1, weight_2, num_cluster)
        if not hasattr(self, '_dap_cached_key') or self._dap_cached_key != cache_key:
            self._dap_cached_users, self._dap_cached_items = self.computer_DAP(weight_1, weight_2, num_cluster)
            self._dap_cached_key = cache_key
        all_users, all_items = self._dap_cached_users, self._dap_cached_items
        # light_out = torch.mean(all_embeddings, dim=1)
        # all_users, all_items = torch.split(light_out, [self.num_users, self.num_items])
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def set_item_to_users(self):
        from collections import defaultdict
        self.item_to_users = defaultdict(list)
        for u, items in enumerate(self.user_interactions):
            for i in items:
                self.item_to_users[i].append(u)

    def aggregate(self, user_emb, item_emb):
        device = user_emb.device

        agg_user_emb = torch.zeros_like(user_emb)

        for u, items in enumerate(self.user_interactions):
            items_tensor = torch.tensor(items, dtype=torch.long, device=device)
            agg_user_emb[u] = item_emb[items_tensor].mean(dim=0)

        agg_item_emb = torch.zeros_like(item_emb)

        for i in range(self.num_items):
            users = self.item_to_users.get(i, [])
            if not users:
                continue
            users_tensor = torch.tensor(users, dtype=torch.long, device=device)
            agg_item_emb[i] = user_emb[users_tensor].mean(dim=0)

        return agg_user_emb, agg_item_emb

    def forward_post_hoc_with_bias_vector_estimation(self, compute_d_pop_all_normalized, phi, beta=1.0):
        print("Inside forward_post_hoc_with_bias_vector_estimation, beta =", beta)
        x_u = self.embedding_user.weight
        x_i = self.embedding_item.weight

        if self.config.get('ppd_seq_v3_v4', False):
            beta_v3 = self.config.get('ppd_seq_beta_v3', None)
            if beta_v3 is None:
                beta_v3 = beta
            
            d_pop_u0, d_pop_i0 = compute_d_pop_all_normalized(
                torch.nn.functional.normalize(x_u), torch.nn.functional.normalize(x_i),
                self.num_users, self.num_items,
                self.bias_scores_of_user_interactions,
                self.user_interactions,
                phi
            )
            eps = 1e-8
            proj_coeff_u0 = (x_u * d_pop_u0).sum(dim=1, keepdim=True)
            proj_coeff_u0_clamped = torch.clamp(proj_coeff_u0, min=0.0)
            x_u = x_u - (proj_coeff_u0_clamped / (d_pop_u0.norm(dim=1, keepdim=True) ** 2 + eps)) * d_pop_u0

            proj_coeff_i0 = (x_i * d_pop_i0).sum(dim=1, keepdim=True)
            proj_coeff_i0_clamped = torch.clamp(proj_coeff_i0, min=0.0)
            x_i = x_i - (proj_coeff_i0_clamped / (d_pop_i0.norm(dim=1, keepdim=True) ** 2 + eps)) * d_pop_i0

        all_user_embs = [x_u]
        all_item_embs = [x_i]

        for layer in range(self.n_layers):
            all_emb = torch.cat([x_u, x_i], dim=0)
            if self.A_split:
                temp_emb = []
                for f in range(len(self.Graph)):
                    temp_emb.append(torch.sparse.mm(self.Graph[f], all_emb))
                all_emb_new = torch.cat(temp_emb, dim=0)
            else:
                all_emb_new = torch.sparse.mm(self.Graph, all_emb)
            x_u_new, x_i_new = torch.split(all_emb_new, [self.num_users, self.num_items])

            x_u, x_i = x_u_new, x_i_new

            # Determine if we should apply PPD at this layer
            apply_ppd = False
            ppd_layers_cfg = self.config.get('ppd_v4_layers', 'all')
            if ppd_layers_cfg == 'all':
                apply_ppd = True
            elif ppd_layers_cfg == 'last':
                if layer == self.n_layers - 1:
                    apply_ppd = True
            else:
                try:
                    target_layers = [int(idx.strip()) for idx in ppd_layers_cfg.split(',') if idx.strip()]
                    if layer in target_layers:
                        apply_ppd = True
                except Exception:
                    apply_ppd = True

            if apply_ppd:
                print(f"[PPD v4] Applying projection removal at layer {layer} (out of {self.n_layers})")
                d_pop_u, d_pop_i = compute_d_pop_all_normalized(
                    x_u, x_i,
                    self.num_users, self.num_items,
                    self.bias_scores_of_user_interactions,
                    self.user_interactions,
                    phi
                )

                beta_l = 1.0
                schedule_cfg = self.config.get('ppd_beta_schedule', None)
                if schedule_cfg is not None:
                    try:
                        schedule = [float(x.strip()) for x in schedule_cfg.split(',') if x.strip()]
                        if layer < len(schedule):
                            beta_l = schedule[layer]
                        else:
                            beta_l = schedule[-1]
                    except Exception:
                        beta_l = 1.0

                eps = 1e-8
                proj_coeff_u = (x_u * d_pop_u).sum(dim=1, keepdim=True)
                proj_coeff_u_clamped = torch.clamp(proj_coeff_u, min=0.0)
                x_u = x_u - (proj_coeff_u_clamped / (d_pop_u.norm(dim=1, keepdim=True) ** 2 + eps)) * d_pop_u

                proj_coeff_i = (x_i * d_pop_i).sum(dim=1, keepdim=True)
                proj_coeff_i_clamped = torch.clamp(proj_coeff_i, min=0.0)
                x_i = x_i - (proj_coeff_i_clamped / (d_pop_i.norm(dim=1, keepdim=True) ** 2 + eps)) * d_pop_i

            all_user_embs.append(x_u)
            all_item_embs.append(x_i)

        tensor1 = torch.stack(all_user_embs, dim=1)
        tensor2 = torch.stack(all_item_embs, dim=1)

        all_emb = torch.cat([tensor1, tensor2], dim=0)

        return all_emb

    def getUsersRatingPostHoc(self, all_embeddings, users):
        ego_weight = self.config.get('ego_weight', -1.0)
        if self.config.get('dynamic_ego', False):
            light_out = self.apply_dynamic_ego(all_embeddings)
        elif ego_weight >= 0.0:
            n_layers = all_embeddings.shape[1] - 1
            weights = [ego_weight] + [(1.0 - ego_weight) / n_layers] * n_layers
            w_tensor = torch.tensor(weights, device=all_embeddings.device).view(1, -1, 1)
            light_out = torch.sum(all_embeddings * w_tensor, dim=1)
        else:
            light_out = torch.mean(all_embeddings, dim=1)
        all_users, all_items = torch.split(light_out, [self.num_users, self.num_items])
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating