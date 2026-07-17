'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=256,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=2,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like Gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='Coat',
                        help="available datasets: [lastfm, Gowalla, yelp2018, amazon-book, Epinions, KuaiRand, KuaiRec, Coat, Yahoo]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[5, 10, 15, 20, 50]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=0,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    # 331 for Coat with emb_dim=256, 581 for KuaiRec with emb_dim=256
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    parser.add_argument('--variant', type=str, default='lgn', help='rec-model, support [lgn, navip]')
    parser.add_argument('--beta', type=float, default=0.20, help='popularity penalty coefficient')
    parser.add_argument('--phi', type=float, default=1.0, help='preference centroid coefficient')
    parser.add_argument('--ppd_variant', type=int, default=3, help='PPD variant (1: EMBEDDING_UPDATE, 2: TRACK_AND_PRUNE, 3: VECTOR_BIAS_ZERO_LAYER_UPDATE, 4: VECTOR_BIAS_ALL_LAYER_UPDATE)')
    parser.add_argument('--ppd_v4_layers', type=str, default='all', help='GCN layers to apply PPD Variant 4 (all, last, or comma-separated indices e.g. 0,1)')
    parser.add_argument('--ppd_beta_schedule', type=str, default=None, help='comma-separated floats for layer-wise beta scheduling in Variant 4 (e.g. 0.2,2.0)')
    parser.add_argument('--ppd_seq_v3_v4', action='store_true', help='enable sequential PPD v3-then-v4')
    parser.add_argument('--ppd_seq_beta_v3', type=float, default=None, help='beta value for PPD v3 step in sequential mode')
    parser.add_argument('--ego_weight', type=float, default=-1.0, help='weight for layer-0 ego embeddings in LightGCN (set between 0 and 1, defaults to uniform)')
    
    # Phase III flags
    parser.add_argument('--sgl', action='store_true', help='enable Self-Supervised Graph Contrastive Learning')
    parser.add_argument('--sgl_weight', type=float, default=0.1, help='weight of SGL loss')
    parser.add_argument('--dynamic_ego', action='store_true', help='enable dynamic attention ego weighting')
    parser.add_argument('--denoise_edges', action='store_true', help='enable IsolationForest edge denoising')
    parser.add_argument('--use_lgbm_ranker', action='store_true', help='enable LightGBM ranker during evaluation')
    parser.add_argument('--use_llm_init', action='store_true', help='enable loading LLM pre-computed initialization')

    # post-hoc toggles for thesis experiments
    parser.add_argument('--tsp', action='store_true', help='apply TSP posthoc')
    parser.add_argument('--tsp_hodge', action='store_true', help='apply Hodge-Laplacian TSP posthoc')
    parser.add_argument('--ppd', action='store_true', help='apply PPD posthoc')
    parser.add_argument('--dap', action='store_true', help='apply DAP (Degree-Aware Propagation)')
    parser.add_argument('--dap_weight1', type=float, default=0.6, help='DAP high-degree anchor weight')
    parser.add_argument('--dap_weight2', type=float, default=0.6, help='DAP low-degree anchor weight')
    parser.add_argument('--dap_num_clusters', type=int, default=5, help='DAP number of clusters')
    parser.add_argument('--fairness', action='store_true', help='print fairness metrics')
    parser.add_argument('--save_ckpt', action='store_true', help='save posthoc checkpoint + results file')
    parser.add_argument('--out_dir', type=str, default='./checkpoints', help='where to save posthoc ckpts/results')

    # TSP params
    parser.add_argument('--theta_tsp', type=float, default=0.845)
    parser.add_argument('--top_m_tsp', type=int, default=25)

    parser.add_argument('--k_tsp', type=int, default=1)
    parser.add_argument('--alpha_tsp', type=float, default=1.0)
    parser.add_argument('--beta_tsp', type=float, default=0.01)
    parser.add_argument('--layers_tsp', type=int, default=1)
    parser.add_argument('--semantic_batch', type=int, default=256)
    parser.add_argument('--max_edges_tsp', type=int, default=450000)

    parser.add_argument('--ckpt', type=str, default="", help='explicit checkpoint path to load (posthoc experiments)')
    return parser.parse_args()

