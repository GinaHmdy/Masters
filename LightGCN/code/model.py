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
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
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
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
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
        user_deg = torch.zeros(self.num_users, device=device)

        for u, items in enumerate(self.user_interactions):
            # if not items:
            #     continue
            items_tensor = torch.tensor(items, dtype=torch.long, device=device)
            agg_user_emb[u] = item_emb[items_tensor].mean(dim=0)
            user_deg[u] = len(items)

        agg_item_emb = torch.zeros_like(item_emb)
        item_deg = torch.zeros(self.num_items, device=device)

        for i in range(self.num_items):
            users = self.item_to_users.get(i, [])
            if not users:
                continue
            users_tensor = torch.tensor(users, dtype=torch.long, device=device)
            agg_item_emb[i] = user_emb[users_tensor].mean(dim=0)
            item_deg[i] = len(users)

        # Normalize embeddings by degree to avoid scale issues
        user_deg = user_deg.clamp(min=1).unsqueeze(1)
        item_deg = item_deg.clamp(min=1).unsqueeze(1)

        agg_user_emb = agg_user_emb / user_deg
        agg_item_emb = agg_item_emb / item_deg

        return agg_user_emb, agg_item_emb

    def forward_post_hoc_with_bias_vector_estimation(self, compute_d_pop_all_normalized, phi):
        all_user_embs = [self.embedding_user.weight]
        all_item_embs = [self.embedding_item.weight]

        x_u = self.embedding_user.weight
        x_i = self.embedding_item.weight

        for layer in range(self.n_layers):
            # Propagate embeddings by aggregating neighbors from interactions
            x_u_new, x_i_new = self.aggregate(x_u, x_i)

            x_u, x_i = x_u_new, x_i_new

            # Compute popularity bias directions using your normalized function
            d_pop_u, d_pop_i = compute_d_pop_all_normalized(
                x_u, x_i,
                self.num_users, self.num_items,
                self.bias_scores_of_user_interactions,
                self.user_interactions
            )

            # Project and remove bias direction for users
            eps = 1e-8
            proj_coeff_u = (x_u * d_pop_u).sum(dim=1, keepdim=True) / (d_pop_u.norm(dim=1, keepdim=True) ** 2 + eps)
            x_u = x_u - proj_coeff_u * d_pop_u

            # Project and remove bias direction for items
            proj_coeff_i = (x_i * d_pop_i).sum(dim=1, keepdim=True) / (d_pop_i.norm(dim=1, keepdim=True) ** 2 + eps)
            x_i = x_i - proj_coeff_i * d_pop_i

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
