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

    # PPD v6 / ASTD-PD CLI flags
    parser.add_argument('--method', type=str, default='lightgcn', choices=['lightgcn', 'tsp', 'ppd_v3', 'ppd_v4', 'ppd_v6'],
                        help='Method selection: lightgcn, tsp, ppd_v3, ppd_v4, ppd_v6')
    parser.add_argument('--use_astd', action='store_true', help='Enable ASTD debiasing')
    parser.add_argument('--astd_alpha_base', type=float, default=0.2, help='Base alpha debiasing coefficient')
    parser.add_argument('--astd_alpha_min', type=float, default=0.0, help='Minimum alpha debiasing coefficient')
    parser.add_argument('--astd_density_xi', type=float, default=1.0, help='Density adaptation scaling parameter xi')
    parser.add_argument('--astd_alpha_pop_exponent', type=float, default=1.0, help='Popularity modulation exponent eta')
    parser.add_argument('--astd_pop_basis_rank', type=int, default=2, help='Popularity subspace rank r')
    parser.add_argument('--astd_pop_basis_ema', type=float, default=0.95, help='EMA coefficient for popularity basis update')
    parser.add_argument('--astd_pop_basis_update_steps', type=int, default=50, help='Steps between popularity basis updates')
    parser.add_argument('--astd_debias_warmup_epochs', type=int, default=5, help='Epochs of debiasing warmup')
    parser.add_argument('--astd_topo_gate_init', type=float, default=0.5, help='Initial / fixed topological gate theta')
    parser.add_argument('--astd_topo_gate_learnable', type=int, default=1, help='Whether topological gate theta is learnable')
    parser.add_argument('--astd_topo_pop_damping', type=float, default=0.1, help='Topological popularity damping kappa')
    parser.add_argument('--astd_highfreq_gamma', type=float, default=0.1, help='High-frequency residual weight gamma')
    parser.add_argument('--astd_user_alpha_scale', type=float, default=0.3, help='User alpha scale relative to item alpha')
    parser.add_argument('--astd_fuse_lambda_min', type=float, default=0.0, help='Minimum Pareto fusion gate lambda')
    parser.add_argument('--astd_fuse_lambda_max', type=float, default=0.5, help='Maximum Pareto fusion gate lambda')
    parser.add_argument('--astd_tail_phi', type=float, default=1.0, help='Tail indicator exponent phi_t')
    parser.add_argument('--astd_tail_reweight', type=int, default=1,
                        help='Enable tail reweighting (inference t_i and the training tail_weights). '
                             '0 disables BOTH -- required for the mechanism-isolation ablation. '
                             'Note --astd_tail_phi 0 does NOT disable it: (1-p)^0 = 1, which makes '
                             'the term uniform rather than absent.')
    parser.add_argument('--astd_temperature_acc', type=float, default=0.1, help='Temperature for accuracy stream')
    parser.add_argument('--astd_temperature_deb', type=float, default=0.1, help='Temperature for debiased stream')
    parser.add_argument('--astd_lambda_final', type=float, default=1.0, help='Weight of final fused loss')
    parser.add_argument('--astd_lambda_acc', type=float, default=0.5, help='Weight of accuracy stream loss')
    parser.add_argument('--astd_lambda_deb', type=float, default=0.5, help='Weight of debiased stream tail loss')
    parser.add_argument('--astd_lambda_ind', type=float, default=0.1, help='Weight of independence loss')
    parser.add_argument('--astd_lambda_cov', type=float, default=0.1, help='Weight of covariance loss')
    parser.add_argument('--astd_neg_uniform_ratio', type=float, default=0.4, help='Uniform negative sampling ratio')
    parser.add_argument('--astd_neg_matched_ratio', type=float, default=0.4, help='Popularity-matched negative sampling ratio')
    parser.add_argument('--astd_neg_hard_ratio', type=float, default=0.2, help='Hard negative sampling ratio')
    parser.add_argument('--astd_use_logq', type=int, default=1, help='Whether to apply logQ correction')
    parser.add_argument('--astd_simplex_topk', type=int, default=200, help='Top-k neighbors for simplex adjacency')
    parser.add_argument('--astd_use_simplex_topk', type=int, default=1, help='Whether to prune simplex adjacency to top-k')
    parser.add_argument('--astd_inference_pop_penalty', type=float, default=0.0, help='Post-hoc popularity penalty mu_cal')
    parser.add_argument('--astd_seed_protocol', type=str, default='three_seed', help='Seed protocol')
    parser.add_argument('--dump_per_user', type=str, default='',
                        help='Path to write per-user Recall/NDCG/Precision vectors (.npz). '
                             'Required for paired significance testing.')
    parser.add_argument('--select_on', type=str, default='val', choices=['val', 'test'],
                        help='Which split sweep drivers should select configurations on. '
                             'Default val. Selecting on test invalidates the result.')
    parser.add_argument('--restrict_candidates', type=int, default=None,
                        help='Restrict eval candidates to the test-split item universe. '
                             'Default: on for KuaiRec, off elsewhere. 1=force on, 0=force off.')
    parser.add_argument('--smoke_test', action='store_true', help='Run fast smoke test (few batches)')

    args, _ = parser.parse_known_args()
    return args


