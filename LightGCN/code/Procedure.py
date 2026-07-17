'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score


CORES = multiprocessing.cpu_count() // 2

import lightgbm as lgb
import random

def get_eval_embeddings(Recmodel, posthoc, all_updated_embeddings, weight_1, weight_2, num_cluster):
    if posthoc == 0:
        all_users, all_items = Recmodel.computer()
    elif posthoc == 1:
        ego_weight = Recmodel.config.get('ego_weight', -1.0)
        if Recmodel.config.get('dynamic_ego', False):
            light_out = Recmodel.apply_dynamic_ego(all_updated_embeddings)
        elif ego_weight >= 0.0:
            n_layers = all_updated_embeddings.shape[1] - 1
            weights = [ego_weight] + [(1.0 - ego_weight) / n_layers] * n_layers
            w_tensor = torch.tensor(weights, device=all_updated_embeddings.device).view(1, -1, 1)
            light_out = torch.sum(all_updated_embeddings * w_tensor, dim=1)
        else:
            light_out = torch.mean(all_updated_embeddings, dim=1)
        all_users, all_items = torch.split(light_out, [Recmodel.num_users, Recmodel.num_items])
    else: # DAP
        all_users, all_items = Recmodel.computer_DAP(weight_1, weight_2, num_cluster)
    return all_users, all_items

def train_lgbm_ranker(dataset, Recmodel, all_users, all_items):
    print("[LightGBM] Training LightGBM ranker on training sample...")
    train_dict = dataset.allPos
    train_users = []
    train_items = []
    labels = []
    
    sampled_users = list(range(len(train_dict)))
    random.seed(42)
    random.shuffle(sampled_users)
    sampled_users = sampled_users[:2000]
    
    for u in sampled_users:
        pos_items = train_dict[u]
        if len(pos_items) == 0:
            continue
        sampled_pos = pos_items[:5]
        for pos_i in sampled_pos:
            train_users.append(u)
            train_items.append(pos_i)
            labels.append(1)
            
            for _ in range(1):
                while True:
                    neg_i = random.randint(0, dataset.m_items - 1)
                    if neg_i not in pos_items:
                        break
                train_users.append(u)
                train_items.append(neg_i)
                labels.append(0)
                
    train_users = np.array(train_users)
    train_items = np.array(train_items)
    labels = np.array(labels)
    
    u_emb = all_users[train_users].detach().cpu().numpy()
    i_emb = all_items[train_items].detach().cpu().numpy()
    
    dot_prod = (u_emb * i_emb).sum(axis=1, keepdims=True)
    cos_sim = dot_prod / (np.linalg.norm(u_emb, axis=1, keepdims=True) * np.linalg.norm(i_emb, axis=1, keepdims=True) + 1e-8)
    elem_prod = u_emb * i_emb
    diff = np.abs(u_emb - i_emb)
    
    X = np.hstack([dot_prod, cos_sim, elem_prod, diff])
    y = labels
    
    train_data = lgb.Dataset(X, label=y)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 15,
        'max_depth': 4,
        'verbosity': -1,
        'random_state': 42
    }
    model = lgb.train(params, train_data, num_boost_round=50)
    print("[LightGBM] Training complete.")
    return model

def predict_lgbm(model, u_emb, all_items_emb):
    num_users = u_emb.shape[0]
    num_items = all_items_emb.shape[0]
    
    preds_all = []
    # Predict in user batches of 50 to speed up loop while keeping memory safe (~1GB)
    user_batch_size = 50
    for i in range(0, num_users, user_batch_size):
        u_batch = u_emb[i:i+user_batch_size]
        n_u = u_batch.shape[0]
        
        u_emb_rep = np.repeat(u_batch, num_items, axis=0)
        i_emb_rep = np.tile(all_items_emb, (n_u, 1))
        
        dot_prod = (u_emb_rep * i_emb_rep).sum(axis=1, keepdims=True)
        cos_sim = dot_prod / (np.linalg.norm(u_emb_rep, axis=1, keepdims=True) * np.linalg.norm(i_emb_rep, axis=1, keepdims=True) + 1e-8)
        elem_prod = u_emb_rep * i_emb_rep
        diff = np.abs(u_emb_rep - i_emb_rep)
        
        X = np.hstack([dot_prod, cos_sim, elem_prod, diff])
        preds = model.predict(X).reshape(n_u, num_items)
        preds_all.append(preds)
        
    return np.vstack(preds_all)


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    if world.config.get('sgl', False) and hasattr(Recmodel, 'sgl_epoch_init'):
        Recmodel.sgl_epoch_init()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg, hr = [], [], [], []
    for k in world.topks:
        ret = utils.RecallPrecisionHR_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
        hr.append(ret['hr'])
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg),
            'hr':np.array(hr)}
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0, posthoc=0,
         all_updated_embeddings=None, weight_1=None, weight_2=None, num_cluster=None,
         dataset_name=None, model_name=None):
    # posthoc=0, all_updated_embeddings=None is added by Aminul, if posthoc=1, use all_updated_embeddings
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks)),
               'hr': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            if world.config.get('use_lgbm_ranker', False):
                if not hasattr(Recmodel, 'lgbm_model') or Recmodel.lgbm_model is None:
                    all_users, all_items = get_eval_embeddings(Recmodel, posthoc, all_updated_embeddings, weight_1, weight_2, num_cluster)
                    Recmodel.lgbm_model = train_lgbm_ranker(dataset, Recmodel, all_users, all_items)
                all_users, all_items = get_eval_embeddings(Recmodel, posthoc, all_updated_embeddings, weight_1, weight_2, num_cluster)
                u_emb = all_users[batch_users_gpu].detach().cpu().numpy()
                all_items_emb = all_items.detach().cpu().numpy()
                preds = predict_lgbm(Recmodel.lgbm_model, u_emb, all_items_emb)
                rating = torch.tensor(preds, dtype=torch.float32, device=world.device)
            else:
                if posthoc == 0:
                    rating = Recmodel.getUsersRating(batch_users_gpu)
                elif posthoc == 1:
                    rating = Recmodel.getUsersRatingPostHoc(all_updated_embeddings, batch_users_gpu)
                else: # DAP
                    rating = Recmodel.getUsersRatingDAP(batch_users_gpu, weight_1, weight_2, num_cluster)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            results['hr'] += result['hr']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['hr'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/HR@{world.topks}',
                          {str(world.topks[i]): results['hr'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        # print(results)
        if dataset_name is not None:
            store_top_k_predicted_items(users_list, rating_list, dataset_name, model_name=model_name)
        return results


def Valid(dataset, Recmodel, epoch, w=None, multicore=0, posthoc=0,
         all_updated_embeddings=None, weight_1=None, weight_2=None, num_cluster=None):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    validDict: dict = dataset.validDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks)),
               'hr': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(validDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"valid_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [validDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            if world.config.get('use_lgbm_ranker', False):
                if not hasattr(Recmodel, 'lgbm_model') or Recmodel.lgbm_model is None:
                    all_users, all_items = get_eval_embeddings(Recmodel, posthoc, all_updated_embeddings, weight_1, weight_2, num_cluster)
                    Recmodel.lgbm_model = train_lgbm_ranker(dataset, Recmodel, all_users, all_items)
                all_users, all_items = get_eval_embeddings(Recmodel, posthoc, all_updated_embeddings, weight_1, weight_2, num_cluster)
                u_emb = all_users[batch_users_gpu].detach().cpu().numpy()
                all_items_emb = all_items.detach().cpu().numpy()
                preds = predict_lgbm(Recmodel.lgbm_model, u_emb, all_items_emb)
                rating = torch.tensor(preds, dtype=torch.float32, device=world.device)
            else:
                if posthoc == 0:
                    rating = Recmodel.getUsersRating(batch_users_gpu)
                elif posthoc == 1:
                    rating = Recmodel.getUsersRatingPostHoc(all_updated_embeddings, batch_users_gpu)
                else: # DAP
                    rating = Recmodel.getUsersRatingDAP(batch_users_gpu, weight_1, weight_2, num_cluster)

            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   valid_data) for i, valid_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            results['hr'] += result['hr']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['hr'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Valid/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Valid/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Valid/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Valid/HR@{world.topks}',
                          {str(world.topks[i]): results['hr'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        # print(results)
        return results


def store_top_k_predicted_items(users_list, rating_list, dataset_name, model_name):
    if model_name is None:
        file = ("../../data/" + dataset_name + "/topk_predict_lightgcn.txt")
    else:
        file = ("../../data/" + dataset_name + "/topk_predict_" + model_name + ".txt")

    with open(file, "w") as f:
        users_batches = users_list.copy()
        top_k_items_batches = rating_list.copy()
        current_batch = 0
        for users_one_batch in users_batches:
            current_index = 0
            for user in users_one_batch:
                top_k_item_for_this_user = top_k_items_batches[current_batch][current_index].tolist()
                line = f"{user} " + " ".join(str(i) for i in top_k_item_for_this_user) + "\n"
                f.write(line)
                current_index += 1
            current_batch += 1