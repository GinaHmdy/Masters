'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['lastfm', 'Gowalla', 'yelp2018', 'amazon-book', 'Epinions', 'KuaiRand', 'KuaiRec', 'Coat', 'Yahoo', 'KuaiRec_Gen']
all_models  = ['mf', 'lgn', 'dap', 'ppd_v6']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['variant'] = args.variant
config['beta'] = args.beta
config['phi'] = args.phi
config['ego_weight'] = args.ego_weight
config['ppd_v4_layers'] = args.ppd_v4_layers
config['ppd_beta_schedule'] = args.ppd_beta_schedule
config['ppd_seq_v3_v4'] = args.ppd_seq_v3_v4
config['ppd_seq_beta_v3'] = args.ppd_seq_beta_v3
config['sgl'] = args.sgl
config['sgl_weight'] = args.sgl_weight
config['dynamic_ego'] = args.dynamic_ego
config['denoise_edges'] = args.denoise_edges
config['use_lgbm_ranker'] = args.use_lgbm_ranker
config['use_llm_init'] = args.use_llm_init

# ASTD / PPD v6 flags
config['method'] = args.method
config['use_astd'] = args.use_astd or (args.method == 'ppd_v6')
config['astd_alpha_base'] = args.astd_alpha_base
config['astd_alpha_min'] = args.astd_alpha_min
config['astd_density_xi'] = args.astd_density_xi
config['astd_alpha_pop_exponent'] = args.astd_alpha_pop_exponent
config['astd_pop_basis_rank'] = args.astd_pop_basis_rank
config['astd_pop_basis_ema'] = args.astd_pop_basis_ema
config['astd_pop_basis_update_steps'] = args.astd_pop_basis_update_steps
config['astd_debias_warmup_epochs'] = args.astd_debias_warmup_epochs
config['astd_topo_gate_init'] = args.astd_topo_gate_init
config['astd_topo_gate_learnable'] = bool(args.astd_topo_gate_learnable)
config['astd_topo_pop_damping'] = args.astd_topo_pop_damping
config['astd_highfreq_gamma'] = args.astd_highfreq_gamma
config['astd_user_alpha_scale'] = args.astd_user_alpha_scale
config['astd_fuse_lambda_min'] = args.astd_fuse_lambda_min
config['astd_fuse_lambda_max'] = args.astd_fuse_lambda_max
config['astd_tail_phi'] = args.astd_tail_phi
config['astd_tail_reweight'] = bool(args.astd_tail_reweight)
config['astd_temperature_acc'] = args.astd_temperature_acc
config['astd_temperature_deb'] = args.astd_temperature_deb
config['astd_lambda_final'] = args.astd_lambda_final
config['astd_lambda_acc'] = args.astd_lambda_acc
config['astd_lambda_deb'] = args.astd_lambda_deb
config['astd_lambda_ind'] = args.astd_lambda_ind
config['astd_lambda_cov'] = args.astd_lambda_cov
config['astd_neg_uniform_ratio'] = args.astd_neg_uniform_ratio
config['astd_neg_matched_ratio'] = args.astd_neg_matched_ratio
config['astd_neg_hard_ratio'] = args.astd_neg_hard_ratio
config['astd_use_logq'] = bool(args.astd_use_logq)
config['astd_simplex_topk'] = args.astd_simplex_topk
config['astd_use_simplex_topk'] = bool(args.astd_use_simplex_topk)
config['astd_inference_pop_penalty'] = args.astd_inference_pop_penalty
config['astd_seed_protocol'] = args.astd_seed_protocol
config['smoke_test'] = args.smoke_test

# Datasets whose test split covers only part of the item catalogue must be
# scored against that sub-catalogue; ranking over all items otherwise lets the
# model spend its top-K on items that can never be a hit. KuaiRec's test split
# is the dense "small matrix" (~30% of items) and needs this; every other
# dataset here covers >94% of the catalogue in test and is left unrestricted.
_RESTRICTED_CANDIDATE_DATASETS = {'KuaiRec', 'KuaiRec_Gen'}
config['dump_per_user'] = args.dump_per_user
config['restrict_candidates'] = (
    bool(args.restrict_candidates)
    if getattr(args, 'restrict_candidates', None) is not None
    else args.dataset in _RESTRICTED_CANDIDATE_DATASETS
)

# Knobs that are parsed and stored but have NO effect on the computation. They are kept
# so existing command lines keep working, but they must announce themselves -- two thirds
# of a 108-config Yahoo sweep measured nothing because --astd_topo_gate_init is
# unreachable while --astd_topo_gate_learnable defaults to 1.
_DEAD_KNOBS = {
    'ppd_beta_schedule': 'beta_l is computed in model.py but never applied to the projection',
    'ppd_seq_beta_v3':   'beta_v3 is computed in model.py but never applied to the projection',
}
for _k, _why in _DEAD_KNOBS.items():
    if config.get(_k) is not None:
        print(f"[WARNING] --{_k} has NO EFFECT: {_why}. Do not sweep over it.")
if getattr(args, 'astd_topo_gate_learnable', 1) and \
        abs(getattr(args, 'astd_topo_gate_init', 0.5) - 0.5) > 1e-12:
    print("[WARNING] --astd_topo_gate_init is ignored while --astd_topo_gate_learnable=1 "
          "(theta comes from gate_mlp_theta). Pass --astd_topo_gate_learnable 0 to use it.")

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = 'ppd_v6' if args.method == 'ppd_v6' else args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")





TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
