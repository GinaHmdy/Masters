import os
import sys
import argparse
import itertools
import numpy as np
import torch
from torch import nn
from sklearn.cluster import KMeans

# Set up world first to ensure imports work correctly
import world
world.config['bpr_batch_size'] = 2048
world.config['latent_dim_rec'] = 256
world.config['lightGCN_n_layers'] = 2
world.config['A_split'] = False
world.config['variant'] = 'DCD'
world.config['keep_prob'] = 0.6
world.config['multicore'] = 1

import model
import dataloader
import utils
import Procedure
from dataloader import Loader
from run_thesis import apply_TSP

class DCDLightGCN(model.BasicModel):
    def __init__(self, config, dataset):
        super(DCDLightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        
        self.embedding_user_int = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item_int = torch.nn.Embedding(self.num_items, self.latent_dim)
        self.embedding_user_con = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item_con = torch.nn.Embedding(self.num_items, self.latent_dim)
        
        nn.init.normal_(self.embedding_user_int.weight, std=0.1)
        nn.init.normal_(self.embedding_item_int.weight, std=0.1)
        nn.init.normal_(self.embedding_user_con.weight, std=0.1)
        nn.init.normal_(self.embedding_item_con.weight, std=0.1)
        
        self.Graph = self.dataset.getSparseGraph()
        
    def computer_interest(self):
        users_emb = self.embedding_user_int.weight
        items_emb = self.embedding_item_int.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        embs = [all_emb]
        g_droped = self.Graph.to(world.device)
        all_emb = all_emb.to(world.device)
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
            
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
        
    def computer_conformity(self):
        users_emb = self.embedding_user_con.weight
        items_emb = self.embedding_item_con.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        u_degree = torch.tensor(self.dataset.users_D, device=world.device)
        i_degree = torch.tensor(self.dataset.items_D, device=world.device)
        degree = torch.cat([u_degree, i_degree], dim=0)
        
        all_embs_anchor_higher = torch.zeros([self.num_users + self.num_items, self.latent_dim]).to(world.device)
        all_embs_anchor_lower = torch.zeros([self.num_users + self.num_items, self.latent_dim]).to(world.device)
        
        embs = [all_emb]
        g_droped = self.Graph.to(world.device)
        all_emb = all_emb.to(world.device)
        
        num_cluster = 5
        weight1 = 0.6
        weight2 = 0.6
        
        for layer in range(self.n_layers):
            all_emb_i = torch.sparse.mm(g_droped, all_emb)
            
            # Subsample for faster kmeans
            all_emb_cpu = all_emb_i.cpu().detach().numpy()
            kmeans = KMeans(n_clusters=num_cluster, random_state=9, n_init='auto', max_iter=20)
            all_emb_cluster = kmeans.fit_predict(all_emb_cpu)
            
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
                
            all_embs_anchor_higher = nn.functional.normalize(all_embs_anchor_higher, p=2, dim=1)
            all_embs_anchor_lower = nn.functional.normalize(all_embs_anchor_lower, p=2, dim=1)
            
            all_emb_i = all_emb_i + weight1 * all_embs_anchor_higher + weight2 * all_embs_anchor_lower
            embs.append(all_emb_i)
            all_emb = all_emb_i
            
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        users_int, items_int = self.computer_interest()
        users_con, items_con = self.computer_conformity()
        
        u_int = users_int[users]
        u_con = users_con[users]
        
        score_int = torch.matmul(u_int, items_int.t())
        score_con = torch.matmul(u_con, items_con.t())
        
        lambda_w = getattr(self, 'lambda_weight', 0.0)
        return score_int + lambda_w * score_con

def Sample_DCD(dataset, batch_size):
    users = np.random.randint(0, dataset.n_users, batch_size)
    pos_items = []
    neg_int_items = []
    neg_con_items = []
    
    items_D = dataset.items_D
    m_items = dataset.m_items
    allPos = dataset.allPos
    
    for u in users:
        pos_items_u = allPos[u]
        if len(pos_items_u) == 0:
            pos_items.append(0)
            neg_int_items.append(0)
            neg_con_items.append(0)
            continue
            
        pos = np.random.choice(pos_items_u)
        pos_items.append(pos)
        pos_deg = items_D[pos]
        
        neg_int, neg_con = -1, -1
        
        for _ in range(50):
            if neg_int == -1:
                ni = np.random.randint(0, m_items)
                if ni not in pos_items_u and items_D[ni] > pos_deg:
                    neg_int = ni
            if neg_con == -1:
                nc = np.random.randint(0, m_items)
                if nc not in pos_items_u and items_D[nc] < pos_deg:
                    neg_con = nc
            if neg_int != -1 and neg_con != -1:
                break
                
        if neg_int == -1:
            while True:
                ni = np.random.randint(0, m_items)
                if ni not in pos_items_u:
                    neg_int = ni
                    break
        if neg_con == -1:
            while True:
                nc = np.random.randint(0, m_items)
                if nc not in pos_items_u:
                    neg_con = nc
                    break
                    
        neg_int_items.append(neg_int)
        neg_con_items.append(neg_con)
        
    return torch.tensor(users), torch.tensor(pos_items), torch.tensor(neg_int_items), torch.tensor(neg_con_items)

def train_dcd_epoch(dataset, model, optimizer, batch_size, gamma, discrepancy_weight):
    model.train()
    users, pos, neg_int, neg_con = Sample_DCD(dataset, batch_size)
    users, pos, neg_int, neg_con = users.to(world.device), pos.to(world.device), neg_int.to(world.device), neg_con.to(world.device)
    
    users_int, items_int = model.computer_interest()
    users_con, items_con = model.computer_conformity()
    
    u_int, p_int, n_int = users_int[users], items_int[pos], items_int[neg_int]
    u_con, p_con, n_con = users_con[users], items_con[pos], items_con[neg_con]
    
    pos_scores_int = torch.mul(u_int, p_int).sum(dim=1)
    neg_scores_int = torch.mul(u_int, n_int).sum(dim=1)
    loss_int = -torch.mean(torch.nn.functional.logsigmoid(pos_scores_int - neg_scores_int))
    
    pos_scores_con = torch.mul(u_con, p_con).sum(dim=1)
    neg_scores_con = torch.mul(u_con, n_con).sum(dim=1)
    loss_con = -torch.mean(torch.nn.functional.logsigmoid(pos_scores_con - neg_scores_con))
    
    discrepancy = torch.mean(torch.cosine_similarity(u_int, u_con)) + torch.mean(torch.cosine_similarity(p_int, p_con))
                  
    S = torch.matmul(u_con, p_con.t())
    try:
        _, S_val, _ = torch.svd(S)
        spectral_reg = S_val[0] if S_val.numel() > 0 else 0
    except:
        spectral_reg = 0
    
    reg_loss = (1/2)*(u_int.norm(2).pow(2) + p_int.norm(2).pow(2) + n_int.norm(2).pow(2) +
                      u_con.norm(2).pow(2) + p_con.norm(2).pow(2) + n_con.norm(2).pow(2)) / float(len(users))
                      
    loss = loss_int + loss_con + discrepancy_weight * discrepancy + gamma * spectral_reg + 1e-4 * reg_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate_dcd(dataset, model, lambda_weight, apply_tsp=False):
    model.eval()
    
    if apply_tsp:
        # Save original weights
        orig_user_int = model.embedding_user_int.weight.data.clone()
        orig_item_int = model.embedding_item_int.weight.data.clone()
        
        # We need a small hack to apply TSP only to interest.
        # apply_TSP expects model.embedding_user and model.embedding_item
        model.embedding_user = model.embedding_user_int
        model.embedding_item = model.embedding_item_int
        
        apply_TSP(model, theta=0.1, top_m=10, K=2, beta=0.5, alpha=0.5, layers=1, semantic_batch=1024, max_edges=1000000, use_hodge=False)
    
    model.lambda_weight = lambda_weight
    
    with torch.no_grad():
        results = Procedure.Test(dataset, model, epoch=0, w=None, multicore=1)
        
    if apply_tsp:
        model.embedding_user_int.weight.data = orig_user_int
        model.embedding_item_int.weight.data = orig_item_int
        
    return results

def main():
    world.dataset = 'Yahoo'
    dataset = Loader(path="../../data/Yahoo")
    
    lambdas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    gammas = [0.0, 0.001, 0.01, 0.1]
    disc_weights = [0.01, 0.1, 1.0]
    seeds = [2020, 2021, 2022]
    
    # Tiny subset for testing
    is_test = os.environ.get('TEST_MODE') == '1'
    if is_test:
        lambdas, gammas, disc_weights, seeds = [0.1], [0.01], [0.1], [2020]
        epochs = 2
    else:
        epochs = 100
        
    configs = list(itertools.product(lambdas, gammas, disc_weights, seeds))
    print(f"Total configs: {len(configs)}")
    
    results_log = []
    
    for l_idx, (lam, gam, disc, seed) in enumerate(configs):
        print(f"Running [{l_idx+1}/{len(configs)}] lam={lam}, gam={gam}, disc={disc}, seed={seed}")
        utils.set_seed(seed)
        model = DCDLightGCN(world.config, dataset).to(world.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for ep in range(epochs):
            loss = train_dcd_epoch(dataset, model, optimizer, world.config['bpr_batch_size'], gam, disc)
            if ep % 10 == 0:
                print(f"Epoch {ep} Loss: {loss}")
                
        # Eval
        res = evaluate_dcd(dataset, model, lam, apply_tsp=False)
        print(f"Eval results: {res}")
        results_log.append({
            'lam': lam, 'gam': gam, 'disc': disc, 'seed': seed,
            'recall': res['recall'][0], 'ndcg': res['ndcg'][0], 'precision': res['precision'][0]
        })
        
    # Summarize
    if not is_test:
        with open('dcd_sweep_results.txt', 'w') as f:
            for r in results_log:
                f.write(f"{r}\n")

if __name__ == '__main__':
    main()
