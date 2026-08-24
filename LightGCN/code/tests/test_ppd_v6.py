import unittest
import numpy as np
import scipy.sparse as sp
import torch
import os
import sys

# Add code dir to sys.path
code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from dataloader import PopularityDensityStore, StratifiedPopularityNegativeSampler, BasicDataset
from model import PPDv6ASTD, LightGCN
from utilities import is_pareto_dominant, compute_pareto_hypervolume, format_pareto_report


class DummyDataset(BasicDataset):
    def __init__(self, n_users=50, m_items=100, num_interactions=300):
        super(DummyDataset, self).__init__()
        self._n_users = n_users
        self._m_items = m_items
        self._traindataSize = num_interactions
        self.path = "/tmp/dummy_data"

        np.random.seed(42)
        users = np.random.randint(0, n_users, size=num_interactions)
        items = np.random.randint(0, m_items, size=num_interactions)
        
        # Unique pairs
        pairs = list(set(zip(users, items)))
        self._traindataSize = len(pairs)
        self.trainUser = np.array([p[0] for p in pairs], dtype=np.int64)
        self.trainItem = np.array([p[1] for p in pairs], dtype=np.int64)

        self._allPos = {u: [] for u in range(n_users)}
        for u, i in pairs:
            self._allPos[u].append(i)

        row = np.array([p[0] for p in pairs])
        col = np.array([p[1] for p in pairs])
        data = np.ones(len(pairs), dtype=np.float32)
        self.UserItemNet = sp.csr_matrix((data, (row, col)), shape=(n_users, m_items))

        # Adjacency graph
        adj_mat = sp.bmat([
            [sp.csr_matrix((n_users, n_users)), self.UserItemNet],
            [self.UserItemNet.T, sp.csr_matrix((m_items, m_items))]
        ], format='csr')
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv = np.zeros_like(rowsum)
        mask = rowsum > 0
        d_inv[mask] = np.power(rowsum[mask], -0.5)
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocoo()
        idx = torch.tensor(np.vstack([norm_adj.row, norm_adj.col]), dtype=torch.long)
        val = torch.tensor(norm_adj.data, dtype=torch.float32)
        self.Graph = torch.sparse_coo_tensor(idx, val, torch.Size([n_users + m_items, n_users + m_items])).coalesce()

    @property
    def n_users(self):
        return self._n_users

    @property
    def m_items(self):
        return self._m_items

    @property
    def trainDataSize(self):
        return self._traindataSize

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self, variant="lgn"):
        return self.Graph

    def getUserPosItems(self, users):
        return [self.allPos[u] for u in users]


class TestPPDv6ASTD(unittest.TestCase):
    def setUp(self):
        self.n_users = 40
        self.m_items = 80
        self.dataset = DummyDataset(n_users=self.n_users, m_items=self.m_items, num_interactions=250)
        self.config = {
            'latent_dim_rec': 32,
            'lightGCN_n_layers': 2,
            'bpr_batch_size': 64,
            'lr': 0.001,
            'decay': 1e-4,
            'method': 'ppd_v6',
            'astd_alpha_base': 0.2,
            'astd_pop_basis_rank': 2,
            'astd_pop_basis_ema': 0.9,
            'astd_pop_basis_update_steps': 5,
            'astd_topo_pop_damping': 0.1,
            'astd_highfreq_gamma': 0.1,
            'astd_fuse_lambda_min': 0.0,
            'astd_fuse_lambda_max': 0.5,
            'astd_tail_phi': 1.0,
            'astd_temperature_acc': 0.1,
            'astd_temperature_deb': 0.1,
            'astd_lambda_final': 1.0,
            'astd_lambda_acc': 0.5,
            'astd_lambda_deb': 0.5,
            'astd_lambda_ind': 0.1,
            'astd_lambda_cov': 0.1,
            'astd_neg_uniform_ratio': 0.4,
            'astd_neg_matched_ratio': 0.4,
            'astd_neg_hard_ratio': 0.2,
            'astd_simplex_topk': 50,
            'ego_weight': -1.0
        }

    def test_popularity_density_store(self):
        store = PopularityDensityStore(self.dataset.UserItemNet, self.n_users, self.m_items, cache_dir="/tmp/test_cache", dataset_name="dummy_test")
        
        self.assertEqual(store.item_popularity_rank.shape[0], self.m_items)
        self.assertTrue(torch.all(store.item_popularity_rank >= 0.0) and torch.all(store.item_popularity_rank <= 1.0))
        
        self.assertEqual(store.user_popularity.shape[0], self.n_users)
        self.assertTrue(torch.all(store.user_popularity >= 0.0) and torch.all(store.user_popularity <= 1.0))

        self.assertEqual(store.item_density_rank.shape[0], self.m_items)
        self.assertTrue(torch.all(store.item_density_rank >= 0.0) and torch.all(store.item_density_rank <= 1.0))

        self.assertEqual(store.user_density_rank.shape[0], self.n_users)
        self.assertTrue(torch.all(store.user_density_rank >= 0.0) and torch.all(store.user_density_rank <= 1.0))

        self.assertEqual(store.user_history_decile_diversity.shape[0], self.n_users)
        self.assertTrue(torch.all(store.user_history_decile_diversity >= 0.0) and torch.all(store.user_history_decile_diversity <= 1.0))

    def test_stratified_negative_sampler(self):
        store = PopularityDensityStore(self.dataset.UserItemNet, self.n_users, self.m_items, cache_dir="/tmp/test_cache", dataset_name="dummy_test")
        sampler = StratifiedPopularityNegativeSampler(
            store.item_popularity_rank,
            store.item_popularity_decile,
            self.dataset.allPos,
            self.m_items,
            uniform_ratio=0.4,
            matched_ratio=0.4,
            hard_ratio=0.2
        )
        batch_users = [0, 1, 2, 3]
        batch_pos = [self.dataset.allPos[0][0], self.dataset.allPos[1][0], self.dataset.allPos[2][0], self.dataset.allPos[3][0]]
        
        neg_items, neg_logq = sampler.sample(batch_users, batch_pos, num_negatives=2)
        
        self.assertEqual(neg_items.shape, (4, 2))
        self.assertEqual(neg_logq.shape, (4, 2))
        self.assertTrue(np.all(np.isfinite(neg_logq)))
        
        # Ensure sampled items are valid indices
        self.assertTrue(np.all(neg_items >= 0) and np.all(neg_items < self.m_items))

    def test_ppd_v6_model_propagation_and_orthonormality(self):
        model = PPDv6ASTD(self.config, self.dataset)
        
        # Check dual stream embeddings
        user_A, item_A, user_D, item_D = model.computer()
        self.assertEqual(user_A.shape, (self.n_users, self.config['latent_dim_rec']))
        self.assertEqual(item_A.shape, (self.m_items, self.config['latent_dim_rec']))
        self.assertEqual(user_D.shape, (self.n_users, self.config['latent_dim_rec']))
        self.assertEqual(item_D.shape, (self.m_items, self.config['latent_dim_rec']))

        # Check popularity subspace basis orthonormalization
        model.update_popularity_basis()
        for l in range(self.config['lightGCN_n_layers'] + 1):
            C = getattr(model, f"popularity_basis_{l}")
            CtC = torch.matmul(C.t(), C)
            identity = torch.eye(self.config['astd_pop_basis_rank'], device=C.device)
            self.assertTrue(torch.allclose(CtC, identity, atol=1e-4))
            self.assertFalse(C.requires_grad)

    def test_ppd_v6_scoring_and_loss(self):
        model = PPDv6ASTD(self.config, self.dataset)
        batch_users = torch.tensor([0, 1, 2], dtype=torch.long, device=model.device)
        batch_pos = torch.tensor([10, 20, 30], dtype=torch.long, device=model.device)
        batch_neg = torch.tensor([[15], [25], [35]], dtype=torch.long, device=model.device)
        batch_neg_logq = torch.tensor([[-4.0], [-4.0], [-4.0]], dtype=torch.float32, device=model.device)

        # Recommendation ratings
        ratings = model.getUsersRating(batch_users)
        self.assertEqual(ratings.shape, (3, self.m_items))
        self.assertTrue(torch.all(torch.isfinite(ratings)))

        # Loss computation
        loss, loss_dict = model.compute_loss(batch_users, batch_pos, batch_neg, batch_neg_logq, epoch=0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_final", loss_dict)
        self.assertIn("loss_acc", loss_dict)
        self.assertIn("loss_deb_tail", loss_dict)
        self.assertIn("loss_ind", loss_dict)
        self.assertIn("loss_proj", loss_dict)
        self.assertIn("loss_cov", loss_dict)
        
        # Test backward pass
        loss.backward()
        self.assertIsNotNone(model.embedding_user_A.weight.grad)
        self.assertIsNotNone(model.embedding_user_D.weight.grad)

    def test_pareto_dominance_and_hypervolume(self):
        cand = {'recall@20': 0.150, 'ndcg@20': 0.120, 'precision@20': 0.050, 'avgpop@20': 200.0}
        base_worse = {'recall@20': 0.140, 'ndcg@20': 0.110, 'precision@20': 0.045, 'avgpop@20': 250.0}
        base_better_acc = {'recall@20': 0.160, 'ndcg@20': 0.110, 'precision@20': 0.045, 'avgpop@20': 250.0}
        
        dom, _ = is_pareto_dominant(cand, base_worse)
        self.assertTrue(dom)

        dom_fail, _ = is_pareto_dominant(cand, base_better_acc)
        self.assertFalse(dom_fail)

        hv_cand = compute_pareto_hypervolume(cand)
        hv_worse = compute_pareto_hypervolume(base_worse)
        self.assertGreater(hv_cand, hv_worse)

        report = format_pareto_report("TestDataset", "PPDv6", cand, {"Baseline": base_worse})
        self.assertIn("✅ YES", report)


if __name__ == '__main__':
    unittest.main()
