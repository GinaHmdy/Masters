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
    return parser.parse_args()
