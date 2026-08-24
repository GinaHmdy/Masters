"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
from collections import defaultdict

import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.variant = self.config['variant']
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
                world.cprint('use NORMAL distribution initilizer')
        elif self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph(variant=self.variant)
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
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph


    def computer_1(self):
        self.residual_coff = 1.0
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        all_emb = torch.cat([users_emb, items_emb])
        initial_emb = nn.functional.normalize(all_emb)
        embs = [all_emb]

        for layer in range(self.n_layers):
            all_emb = all_emb + self.residual_coff * initial_emb
            all_emb = nn.functional.normalize(all_emb)
            neighbor_emb = torch.sparse.mm(self.Graph, all_emb)
            all_emb = neighbor_emb + self.residual_coff * (all_emb - initial_emb)  # preserve last layer embedding
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        return users, items


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
        """
        propagate methods for lightGCN
        """
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

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
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
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
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
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
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
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma

    def get_all_layer_embeddings(self, users_emb, items_emb):
        """
        propagate methods for lightGCN
        """
        # users_emb = self.embedding_user.weight
        # items_emb = self.embedding_item.weight
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

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        return embs

    def getUsersRatingPostHoc(self, all_embeddings, users):
        # all_users, all_items = self.computer()
        ego_weight = self.config.get('ego_weight', -1.0)
        if ego_weight >= 0.0:
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

    # begin region: this part is for multi layer post-hoc bias vector estimation and embedding update
    def set_item_to_users(self):
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
            # Standard GCN propagation using the actual model's Graph (matrix multiplication)
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
                # Compute popularity bias directions using your normalized function
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

                # Project and remove bias direction for users
                eps = 1e-8
                proj_coeff_u = (x_u * d_pop_u).sum(dim=1, keepdim=True)
                proj_coeff_u_clamped = torch.clamp(proj_coeff_u, min=0.0)
                x_u = x_u - (proj_coeff_u_clamped / (d_pop_u.norm(dim=1, keepdim=True) ** 2 + eps)) * d_pop_u

                # Project and remove bias direction for items
                proj_coeff_i = (x_i * d_pop_i).sum(dim=1, keepdim=True)
                proj_coeff_i_clamped = torch.clamp(proj_coeff_i, min=0.0)
                x_i = x_i - (proj_coeff_i_clamped / (d_pop_i.norm(dim=1, keepdim=True) ** 2 + eps)) * d_pop_i

            all_user_embs.append(x_u)
            all_item_embs.append(x_i)

        tensor1 = torch.stack(all_user_embs, dim=1)  # shape: (300, 3, 256)
        tensor2 = torch.stack(all_item_embs, dim=1)  # shape: (290, 3, 256)

        # Concatenate along dim=0 → final shape: (590, 3, 256)
        all_emb = torch.cat([tensor1, tensor2], dim=0)

        return all_emb

        # final_user_emb = torch.stack(all_user_embs, dim=0).mean(dim=0)
        # final_item_emb = torch.stack(all_item_embs, dim=0).mean(dim=0)
        #
        # return torch.cat([final_user_emb, final_item_emb])
        # end region


class PPDv6ASTD(BasicModel):
    """
    PPD v6 / ASTD-PD:
    Adaptive Spectral-Topological Debiasing with Pareto Dual-Stream Fusion.
    """
    def __init__(self, config: dict, dataset: BasicDataset):
        super(PPDv6ASTD, self).__init__()
        self.config = config
        self.dataset = dataset
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config.get('latent_dim_rec', 64)
        self.n_layers = config.get('lightGCN_n_layers', 2)
        self.device = world.device

        # Hyperparameters with robust defaults
        self.alpha_base = config.get('astd_alpha_base', 0.2)
        if self.alpha_base is None:
            self.alpha_base = 0.2
        self.alpha_min = config.get('astd_alpha_min', 0.0)
        self.density_xi = config.get('astd_density_xi', 1.0)
        self.alpha_pop_exponent = config.get('astd_alpha_pop_exponent', 1.0)
        self.pop_basis_rank = config.get('astd_pop_basis_rank', 2)
        self.pop_basis_ema = config.get('astd_pop_basis_ema', 0.95)
        self.pop_basis_update_steps = config.get('astd_pop_basis_update_steps', 50)
        self.debias_warmup_epochs = config.get('astd_debias_warmup_epochs', 5)

        self.topo_gate_init = config.get('astd_topo_gate_init', 0.5)
        self.topo_gate_learnable = config.get('astd_topo_gate_learnable', True)
        self.topo_pop_damping = config.get('astd_topo_pop_damping', 0.1)
        self.highfreq_gamma = config.get('astd_highfreq_gamma', 0.1)
        self.user_alpha_scale = config.get('astd_user_alpha_scale', 0.3)

        self.fuse_lambda_min = config.get('astd_fuse_lambda_min', 0.0)
        self.fuse_lambda_max = config.get('astd_fuse_lambda_max', 0.5)
        self.tail_phi = config.get('astd_tail_phi', 1.0)
        self.temp_acc = config.get('astd_temperature_acc', 0.1)
        self.temp_deb = config.get('astd_temperature_deb', 0.1)

        self.lambda_final = config.get('astd_lambda_final', 1.0)
        self.lambda_acc = config.get('astd_lambda_acc', 0.5)
        self.lambda_deb = config.get('astd_lambda_deb', 0.5)
        self.lambda_ind = config.get('astd_lambda_ind', 0.1)
        self.lambda_cov = config.get('astd_lambda_cov', 0.1)
        self.decay = config.get('decay', 1e-4)

        self.ego_weight = config.get('ego_weight', -1.0)
        self.inference_pop_penalty = config.get('astd_inference_pop_penalty', 0.0)
        self.simplex_topk = config.get('astd_simplex_topk', 200)

        # 1. Dual-stream embeddings (Accuracy Stream & Debiased Stream)
        self.embedding_user_A = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item_A = nn.Embedding(self.num_items, self.latent_dim)
        self.embedding_user_D = nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item_D = nn.Embedding(self.num_items, self.latent_dim)

        nn.init.normal_(self.embedding_user_A.weight, std=0.1)
        nn.init.normal_(self.embedding_item_A.weight, std=0.1)
        nn.init.normal_(self.embedding_user_D.weight, std=0.1)
        nn.init.normal_(self.embedding_item_D.weight, std=0.1)

        # Backward compatibility attributes for evaluate/inspect code
        self.embedding_user = self.embedding_user_A
        self.embedding_item = self.embedding_item_A

        # 2. Registered Popularity Subspace Basis Buffers per layer (Detached from autograd)
        for l in range(self.n_layers + 1):
            init_basis = torch.zeros(self.latent_dim, self.pop_basis_rank)
            # Initialize with orthonormal columns
            nn.init.orthogonal_(init_basis)
            self.register_buffer(f"popularity_basis_{l}", init_basis)

        # 3. Popularity & Density features store
        from dataloader import PopularityDensityStore
        self.pop_store = PopularityDensityStore(
            self.dataset.UserItemNet,
            self.num_users,
            self.num_items,
            dataset_name=getattr(world, "dataset", "dataset")
        )
        self._init_feature_tensors()

        # Global dataset-density scaled alpha_max
        train_size = getattr(self.dataset, 'trainDataSize', getattr(self.dataset, 'traindataSize', None))
        if train_size is None or callable(train_size):
            try:
                train_size = self.dataset.trainDataSize
            except Exception:
                train_size = self.dataset.UserItemNet.nnz
        avg_item_deg = float(train_size) / max(1, self.num_items)
        d_ref = 20.0
        scaled_alpha = self.alpha_base / (1.0 + self.density_xi * np.log1p(avg_item_deg / d_ref))
        self.effective_alpha_max = float(np.clip(scaled_alpha, self.alpha_min, self.alpha_base))
        print(f"[PPD v6] Global Item Avg Deg = {avg_item_deg:.2f} -> Scaled Alpha Max = {self.effective_alpha_max:.4f}")

        # 4. Neural Routing & Debiasing Modules (MLPs)
        # Topological gate: [pop, density, is_user] -> theta in [0, 1]
        self.gate_mlp_theta = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # Adaptive alpha: [pop, density] -> a in [0, 1]
        self.gate_mlp_alpha = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # High-frequency gamma: [pop, density] -> gamma in [0, gamma_base]
        self.gate_mlp_gamma = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # Fusion lambda gate: [user_pop, user_density, user_diversity] -> lambda in [min, max]
        self.fuse_mlp_lambda = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 5. Graph Adjacency matrices (Standard GCN and Damped Simplex)
        self.Graph = self.dataset.getSparseGraph()
        self._build_topological_adjacency()

        # Step counter for basis EMA update
        self.register_buffer("train_step_count", torch.tensor(0, dtype=torch.long))

    def _init_feature_tensors(self):
        p_u = self.pop_store.user_popularity.view(-1, 1).to(self.device)
        p_i = self.pop_store.item_popularity_rank.view(-1, 1).to(self.device)
        self.register_buffer("node_popularity", torch.cat([p_u, p_i], dim=0))

        d_u = self.pop_store.user_density_rank.view(-1, 1).to(self.device)
        d_i = self.pop_store.item_density_rank.view(-1, 1).to(self.device)
        self.register_buffer("node_density", torch.cat([d_u, d_i], dim=0))

        t_u = torch.ones(self.num_users, 1, device=self.device)
        t_i = torch.zeros(self.num_items, 1, device=self.device)
        self.register_buffer("node_type", torch.cat([t_u, t_i], dim=0))

        div_u = self.pop_store.user_history_decile_diversity.view(-1, 1).to(self.device)
        self.register_buffer("user_diversity", div_u)
        self.register_buffer("item_pop_rank_tensor", self.pop_store.item_popularity_rank.to(self.device))

    def _build_topological_adjacency(self):
        """
        Constructs popularity-damped 2nd-order simplex adjacency matrix S^kappa.
        W_{ij}^kappa = (sum_u R_{ui} R_{uj}) / (sqrt(deg_i * deg_j) * (1+p_i)^{kappa/2} * (1+p_j)^{kappa/2})
        Keeps top-k neighbors per item for memory efficiency.
        """
        import scipy.sparse as sp
        print(f"[PPD v6] Building popularity-damped simplex adjacency (kappa={self.topo_pop_damping})...")
        R = self.dataset.UserItemNet
        if not sp.issparse(R):
            R = sp.csr_matrix(R)
        else:
            R = R.tocsr()

        # Item-item co-occurrence: C = R^T R
        item_deg = np.array(R.sum(axis=0)).flatten().astype(np.float32)
        deg_inv_sqrt = np.zeros_like(item_deg)
        pos_mask = item_deg > 0
        deg_inv_sqrt[pos_mask] = 1.0 / np.sqrt(item_deg[pos_mask])

        pop_rank = self.pop_store.item_popularity_rank.numpy()
        damp_factor = 1.0 / np.power(1.0 + pop_rank, self.topo_pop_damping / 2.0)
        weight_factor = deg_inv_sqrt * damp_factor

        # Symmetrically scaled interaction matrix: R_scaled = R * diag(weight_factor)
        W_diag = sp.diags(weight_factor)
        R_scaled = R.dot(W_diag).tocsr()
        ItemCooc = (R_scaled.T.dot(R_scaled)).tocsr()
        ItemCooc.setdiag(0)
        ItemCooc.eliminate_zeros()

        # Top-K neighbor pruning per item
        top_k = min(self.simplex_topk, self.num_items)
        rows, cols, data = [], [], []

        for i in range(self.num_items):
            start = ItemCooc.indptr[i]
            end = ItemCooc.indptr[i+1]
            if start == end:
                continue
            col_indices = ItemCooc.indices[start:end]
            col_data = ItemCooc.data[start:end]

            if len(col_indices) > top_k:
                top_idx = np.argpartition(col_data, -top_k)[-top_k:]
                col_indices = col_indices[top_idx]
                col_data = col_data[top_idx]

            rows.extend([i] * len(col_indices))
            cols.extend(col_indices)
            data.extend(col_data)

        if len(rows) == 0:
            # Fallback identity
            rows = list(range(self.num_items))
            cols = list(range(self.num_items))
            data = [1.0] * self.num_items

        ItemCooc_pruned = sp.csr_matrix((data, (rows, cols)), shape=(self.num_items, self.num_items), dtype=np.float32)
        
        # Symmetrically normalize S^kappa
        rowsum = np.array(ItemCooc_pruned.sum(axis=1)).flatten()
        d_inv = np.zeros_like(rowsum)
        non_zero = rowsum > 0
        d_inv[non_zero] = np.power(rowsum[non_zero], -0.5)
        d_inv_mat = sp.diags(d_inv)
        Item_S_norm = d_inv_mat.dot(ItemCooc_pruned).dot(d_inv_mat).tocoo()

        idx = torch.tensor(np.vstack([Item_S_norm.row, Item_S_norm.col]), dtype=torch.long)
        val = torch.tensor(Item_S_norm.data, dtype=torch.float32)
        self.Simplex_Item_Graph = torch.sparse_coo_tensor(
            idx, val, torch.Size([self.num_items, self.num_items]), device=self.device
        ).coalesce()
        print(f"[PPD v6] Simplex Item Adjacency built with {self.Simplex_Item_Graph._nnz()} edges.")

    def _get_popularity_basis(self, layer_idx: int) -> torch.Tensor:
        return getattr(self, f"popularity_basis_{layer_idx}")

    @torch.no_grad()
    def update_popularity_basis(self):
        """
        Detached EMA update of the low-rank popularity subspace basis per layer.
        Constructs candidates from 5-quantile bucket means of item embeddings and re-orthonormalizes via QR.
        """
        all_user_A, all_item_A, _, _ = self.get_all_layer_embeddings()
        dev = all_item_A[0].device
        bucket_indices = self.pop_store.item_bucket_indices.to(dev)

        for l in range(len(all_item_A)):
            item_emb_l = all_item_A[l].detach()  # [m_items, d]
            
            # Compute quantile bucket means
            means = []
            for b in range(5):
                mask = (bucket_indices == b)
                if mask.sum() > 0:
                    means.append(item_emb_l[mask].mean(dim=0))
                else:
                    means.append(item_emb_l.mean(dim=0))
            
            mu_global = item_emb_l.mean(dim=0)
            v1 = means[4] - means[0]
            v2 = means[3] - means[1]
            v3 = means[2] - mu_global
            V = torch.stack([v1, v2, v3], dim=1)  # [d, 3]

            # Top-r singular vectors / QR
            Q, _ = torch.linalg.qr(V)
            C_new = Q[:, :self.pop_basis_rank]  # [d, r]

            # EMA update
            current_basis = self._get_popularity_basis(l).to(dev)
            updated_basis = self.pop_basis_ema * current_basis + (1.0 - self.pop_basis_ema) * C_new
            
            # Re-orthonormalize
            Q_ortho, _ = torch.linalg.qr(updated_basis)
            self._get_popularity_basis(l).copy_(Q_ortho[:, :self.pop_basis_rank])

    def _project_popularity(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Projects embeddings x onto the low-rank popularity subspace: Pi(x) = x C C^T
        """
        C = self._get_popularity_basis(layer_idx).to(x.device).detach()  # [d, r]
        # x: [N, d] -> (x @ C) @ C^T
        return torch.matmul(torch.matmul(x, C), C.t())

    def get_all_layer_embeddings(self):
        """
        Runs dual-stream forward propagation across layers.
        Returns:
            all_user_A, all_item_A, all_user_D, all_item_D: list of tensors per layer
        """
        ea_u0 = self.embedding_user_A.weight
        ea_i0 = self.embedding_item_A.weight
        ed_u0 = self.embedding_user_D.weight
        ed_i0 = self.embedding_item_D.weight

        E_A = torch.cat([ea_u0, ea_i0], dim=0)
        E_D = torch.cat([ed_u0, ed_i0], dim=0)

        all_user_A = [ea_u0]
        all_item_A = [ea_i0]
        all_user_D = [ed_u0]
        all_item_D = [ed_i0]

        # Ensure graphs and buffers are on the same device as E_A
        dev = E_A.device
        if self.Graph.device != dev:
            self.Graph = self.Graph.to(dev)
        if self.Simplex_Item_Graph.device != dev:
            self.Simplex_Item_Graph = self.Simplex_Item_Graph.to(dev)
        if self.node_popularity.device != dev:
            self.node_popularity = self.node_popularity.to(dev)
            self.node_density = self.node_density.to(dev)
            self.node_type = self.node_type.to(dev)
            self.user_diversity = self.user_diversity.to(dev)
            self.item_pop_rank_tensor = self.item_pop_rank_tensor.to(dev)

        # Ensure MLPs are on dev
        self.gate_mlp_theta = self.gate_mlp_theta.to(dev)
        self.gate_mlp_alpha = self.gate_mlp_alpha.to(dev)
        self.gate_mlp_gamma = self.gate_mlp_gamma.to(dev)
        self.fuse_mlp_lambda = self.fuse_mlp_lambda.to(dev)

        # Features for MLPs
        feat_3 = torch.cat([self.node_popularity, self.node_density, self.node_type], dim=1)  # [N, 3]
        feat_2 = torch.cat([self.node_popularity, self.node_density], dim=1)                  # [N, 2]

        all_alphas = []
        all_thetas = []

        for layer in range(self.n_layers):
            # 1. Standard GCN propagation M^G = P E^A
            M_G = torch.sparse.mm(self.Graph, E_A)

            # 2. Simplex Topological message M^T
            # Items receive damped simplex message, users receive through bipartite graph
            E_A_items = E_A[self.num_users:]
            M_T_items = torch.sparse.mm(self.Simplex_Item_Graph, E_A_items)
            M_T_all = torch.cat([M_G[:self.num_users], M_T_items], dim=0)

            # 3. Topological Router Gate theta_v
            if self.topo_gate_learnable:
                theta = self.gate_mlp_theta(feat_3)  # [N, 1] in [0, 1]
            else:
                theta = torch.full((self.num_users + self.num_items, 1), self.topo_gate_init, device=dev)
            all_thetas.append(theta.mean().item())

            m = theta * M_T_all + (1.0 - theta) * M_G  # [N, d]

            # 4. High-frequency residual h_v = E^A - m
            h = E_A - m

            # 5. Adaptive alpha and Spectral Popularity Projection
            raw_a = self.gate_mlp_alpha(feat_2)  # [N, 1]
            p_v = self.node_popularity           # [N, 1]
            m_med = 0.5
            rho = (p_v ** self.alpha_pop_exponent) / (p_v ** self.alpha_pop_exponent + (m_med ** self.alpha_pop_exponent) + 1e-8)
            alpha_node = self.effective_alpha_max * raw_a * rho

            # Scale down user alpha
            user_mask = (self.node_type == 1.0).float()
            alpha_node = alpha_node * (user_mask * self.user_alpha_scale + (1.0 - user_mask))
            all_alphas.append(alpha_node.mean().item())

            # Gamma for high-frequency residual
            gamma = self.highfreq_gamma * self.gate_mlp_gamma(feat_2)

            # Popularity projection Pi(m)
            Pi_m = self._project_popularity(m, layer)

            # Debiased message m_D
            m_D = m - alpha_node * Pi_m + gamma * h

            # 6. Ego connection / stream update
            if self.ego_weight >= 0.0:
                E_A = (1.0 - self.ego_weight) * m + self.ego_weight * torch.cat([ea_u0, ea_i0], dim=0)
                E_D = (1.0 - self.ego_weight) * m_D + self.ego_weight * torch.cat([ed_u0, ed_i0], dim=0)
            else:
                E_A = m
                E_D = m_D

            u_A, i_A = torch.split(E_A, [self.num_users, self.num_items])
            u_D, i_D = torch.split(E_D, [self.num_users, self.num_items])

            all_user_A.append(u_A)
            all_item_A.append(i_A)
            all_user_D.append(u_D)
            all_item_D.append(i_D)

        self.last_mean_alpha = float(np.mean(all_alphas)) if all_alphas else 0.0
        self.last_mean_theta = float(np.mean(all_thetas)) if all_thetas else 0.0

        return all_user_A, all_item_A, all_user_D, all_item_D

    def computer(self):
        """
        Computes layer-aggregated dual-stream embeddings.
        Returns:
            user_A, item_A, user_D, item_D (layer averaged)
        """
        all_user_A, all_item_A, all_user_D, all_item_D = self.get_all_layer_embeddings()
        
        user_A = torch.mean(torch.stack(all_user_A, dim=0), dim=0)
        item_A = torch.mean(torch.stack(all_item_A, dim=0), dim=0)
        user_D = torch.mean(torch.stack(all_user_D, dim=0), dim=0)
        item_D = torch.mean(torch.stack(all_item_D, dim=0), dim=0)

        return user_A, item_A, user_D, item_D

    def getUsersRating(self, users):
        """
        Computes full recommendation scores for evaluation using Pareto Fusion.
        """
        user_A, item_A, user_D, item_D = self.computer()
        dev = user_A.device
        users = users.long().to(dev)

        u_A = user_A[users]  # [B, d]
        u_D = user_D[users]  # [B, d]

        # Accuracy scores: [B, I]
        s_A = torch.matmul(u_A, item_A.t()) / max(1e-4, self.temp_acc)

        # Debiased scores: [B, I]
        s_D = torch.matmul(u_D, item_D.t()) / max(1e-4, self.temp_deb)

        # Fusion gate lambda_u
        u_feat = torch.cat([
            self.pop_store.user_popularity[users.cpu()].view(-1, 1).to(dev),
            self.pop_store.user_density_rank[users.cpu()].view(-1, 1).to(dev),
            self.user_diversity[users]
        ], dim=1)
        raw_lambda = self.fuse_mlp_lambda(u_feat)  # [B, 1]
        lambda_u = self.fuse_lambda_min + (self.fuse_lambda_max - self.fuse_lambda_min) * raw_lambda

        # Tail importance t_i = (1 - p_i)^phi_t: [1, I]
        p_i = self.item_pop_rank_tensor.view(1, -1).to(dev)
        t_i = torch.pow(torch.clamp(1.0 - p_i, min=0.0, max=1.0), self.tail_phi)

        # Fused score
        scores = s_A + (lambda_u * t_i) * s_D
        if self.inference_pop_penalty > 0.0:
            scores = scores - self.inference_pop_penalty * p_i

        return scores

    def compute_loss(self, batch_users, batch_pos, batch_neg, batch_neg_logq, epoch=0):
        """
        Computes multi-objective loss:
        L = lambda_final * L_final + lambda_acc * L_A + lambda_deb * L_D,tail + lambda_ind * L_ind + lambda_2 * L_2
        """
        user_A, item_A, user_D, item_D = self.computer()
        dev = user_A.device

        batch_users = batch_users.long().to(dev)
        batch_pos = batch_pos.long().to(dev)
        batch_neg = batch_neg.long().to(dev)

        u_A = user_A[batch_users]
        pos_A = item_A[batch_pos]
        neg_A = item_A[batch_neg]

        u_D = user_D[batch_users]
        pos_D = item_D[batch_pos]
        neg_D = item_D[batch_neg]

        # 1. Accuracy Stream Scores
        pos_s_A = torch.sum(u_A * pos_A, dim=-1) / max(1e-4, self.temp_acc)
        neg_s_A = torch.sum(u_A.unsqueeze(1) * neg_A, dim=-1) / max(1e-4, self.temp_acc)

        # 2. Debiased Stream Scores
        pos_s_D = torch.sum(u_D * pos_D, dim=-1) / max(1e-4, self.temp_deb)
        neg_s_D = torch.sum(u_D.unsqueeze(1) * neg_D, dim=-1) / max(1e-4, self.temp_deb)

        # 3. User Fusion Gate & Tail Factor
        u_feat = torch.cat([
            self.pop_store.user_popularity[batch_users.cpu()].view(-1, 1).to(dev),
            self.pop_store.user_density_rank[batch_users.cpu()].view(-1, 1).to(dev),
            self.user_diversity[batch_users]
        ], dim=1)
        raw_lambda = self.fuse_mlp_lambda(u_feat)
        lambda_u = self.fuse_lambda_min + (self.fuse_lambda_max - self.fuse_lambda_min) * raw_lambda  # [B, 1]

        p_pos = self.item_pop_rank_tensor[batch_pos]
        t_pos = torch.pow(torch.clamp(1.0 - p_pos, min=0.0, max=1.0), self.tail_phi)

        p_neg = self.item_pop_rank_tensor[batch_neg]
        t_neg = torch.pow(torch.clamp(1.0 - p_neg, min=0.0, max=1.0), self.tail_phi)

        # Fused scores
        pos_s_final = pos_s_A + lambda_u.squeeze(-1) * t_pos * pos_s_D
        neg_s_final = neg_s_A + lambda_u * t_neg * neg_s_D

        # LogQ correction on negative scores
        logq = batch_neg_logq.to(dev)
        neg_s_final_corr = neg_s_final - logq
        neg_s_A_corr = neg_s_A - logq
        neg_s_D_corr = neg_s_D - logq

        # Sampled Softmax Losses
        # L_final
        logits_final = torch.cat([pos_s_final.unsqueeze(1), neg_s_final_corr], dim=1)
        loss_final = -torch.log_softmax(logits_final, dim=1)[:, 0].mean()

        # L_acc
        logits_A = torch.cat([pos_s_A.unsqueeze(1), neg_s_A_corr], dim=1)
        loss_acc = -torch.log_softmax(logits_A, dim=1)[:, 0].mean()

        # L_D,tail (weighted by tail factor 1 + beta_t * t_pos)
        logits_D = torch.cat([pos_s_D.unsqueeze(1), neg_s_D_corr], dim=1)
        tail_weights = 1.0 + t_pos
        loss_deb_tail = (tail_weights * (-torch.log_softmax(logits_D, dim=1)[:, 0])).mean()

        # 4. Popularity Independence Loss on item debiased embeddings
        # Projection loss: ||Pi(item_D)||^2
        proj_item_D = self._project_popularity(item_D, self.n_layers)
        loss_proj = torch.mean(torch.sum(proj_item_D ** 2, dim=-1))

        # Covariance loss with rank popularity
        p_items_std = self.item_pop_rank_tensor - self.item_pop_rank_tensor.mean()
        var_p = torch.sum(p_items_std ** 2) + 1e-8
        item_D_centered = item_D - item_D.mean(dim=0, keepdim=True)
        var_D = torch.sum(item_D_centered ** 2, dim=0) + 1e-8
        cov = torch.sum(item_D_centered * p_items_std.unsqueeze(1), dim=0)
        corr_sq = (cov ** 2) / (var_D * var_p + 1e-8)
        loss_cov = torch.mean(corr_sq)

        loss_ind = loss_proj + self.lambda_cov * loss_cov

        # 5. Regularization Loss L2
        loss_reg = (1.0 / 2.0) * (
            self.embedding_user_A.weight.norm(2).pow(2) +
            self.embedding_item_A.weight.norm(2).pow(2) +
            self.embedding_user_D.weight.norm(2).pow(2) +
            self.embedding_item_D.weight.norm(2).pow(2)
        ) / float(len(batch_users))

        total_loss = (
            self.lambda_final * loss_final +
            self.lambda_acc * loss_acc +
            self.lambda_deb * loss_deb_tail +
            self.lambda_ind * loss_ind +
            self.decay * loss_reg
        )

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_final": loss_final.item(),
            "loss_acc": loss_acc.item(),
            "loss_deb_tail": loss_deb_tail.item(),
            "loss_ind": loss_ind.item(),
            "loss_proj": loss_proj.item(),
            "loss_cov": loss_cov.item(),
            "loss_reg": loss_reg.item(),
            "mean_lambda_fuse": lambda_u.mean().item(),
            "mean_alpha": getattr(self, "last_mean_alpha", 0.0),
            "mean_theta": getattr(self, "last_mean_theta", 0.0)
        }

        return total_loss, loss_dict

