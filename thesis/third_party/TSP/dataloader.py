import os

import numpy as np
import scipy.sparse as sp

DATASET_STATS = {
    "adressa": (13485, 744, 116321),
    "globo": (158323, 12005, 2520171),
    "ml10m": (69166, 8790, 5000415),
    "yelp2018": (31668, 38048, 1561406),
    "gowalla": (29858, 40981, 1027370),
}

DATASETS = list(DATASET_STATS.keys())


def _parse_interaction_file(path):
    users, items = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            for it in parts[1:]:
                users.append(u)
                items.append(int(it))
    return np.asarray(users, dtype=np.int64), np.asarray(items, dtype=np.int64)


def generate_synthetic(name, seed=42, user_scale=1 / 30, min_items=200,
                       avg_degree=30, zipf_alpha=0.9, n_topics=16,
                       topic_conc=0.8):
    rng = np.random.default_rng(seed + sum(map(ord, name)))
    n_u_real, n_i_real, _ = DATASET_STATS[name]
    n_users = max(int(n_u_real * user_scale), 200)
    n_items = max(int(n_i_real * user_scale), min_items)
    n_topics = min(n_topics, n_items)

    item_topic = rng.integers(0, n_topics, size=n_items)
    items_by_topic = [np.where(item_topic == t)[0] for t in range(n_topics)]
    # Zipf popularity inside each topic
    topic_probs = []
    for items_t in items_by_topic:
        p = 1.0 / np.power(np.arange(1, len(items_t) + 1), zipf_alpha)
        topic_probs.append(p / p.sum())

    users, items = [], []
    for u in range(n_users):
        my_topics = rng.choice(n_topics, size=min(3, n_topics), replace=False)
        deg = int(rng.integers(max(2, avg_degree // 2), avg_degree + 1))
        for _ in range(min(deg, n_items)):
            t = rng.choice(my_topics) if rng.random() < topic_conc else rng.integers(n_topics)
            items_t = items_by_topic[t]
            items.append(items_t[rng.choice(len(items_t), p=topic_probs[t])])
            users.append(u)
    pair = np.unique(np.stack([users, items], axis=1), axis=0)  # deduplicate
    return pair[:, 0], pair[:, 1], n_users, n_items


class RecDataset:
    def __init__(self, name, data_dir=None, seed=42, sample=False):
        assert name in DATASET_STATS, f"unknown dataset {name}, choose from {DATASETS}"
        self.name = name
        self.seed = seed
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        self.dir = os.path.join(data_dir, name)
        os.makedirs(self.dir, exist_ok=True)

        if self._has_real_files() and not sample:
            self._load_real()
        else:
            print(f"[data] no real files found in {self.dir}; generating a synthetic sample.")
            self._generate_sample()

        self.n_interactions = len(self.train_users)
        self.train_csr = self._to_csr(self.train_users, self.train_items)
        self.valid_csr = self._to_csr(self.valid_users, self.valid_items)
        self.test_csr = self._to_csr(self.test_users, self.test_items)
        self.eval_users = np.unique(self.test_users)

        # 20% least popular items (popularity measured on the training set)
        popularity = np.asarray(self.train_csr.sum(axis=0)).flatten()
        order = np.argsort(popularity, kind="stable")
        n_tail = max(1, int(0.2 * self.n_items))
        self.tail_item_mask = np.zeros(self.n_items, dtype=bool)
        self.tail_item_mask[order[:n_tail]] = True

        print(f"[data] {name}: {self.n_users} users, {self.n_items} items, "
              f"{self.train_csr.nnz} train / {self.valid_csr.nnz} valid / "
              f"{self.test_csr.nnz} test interactions, {n_tail} tail items")

    def _has_real_files(self):
        return os.path.exists(os.path.join(self.dir, "interactions.txt")) or \
            os.path.exists(os.path.join(self.dir, "train.txt"))

    def _load_real(self):
        inter_path = os.path.join(self.dir, "interactions.txt")
        if os.path.exists(inter_path):
            users, items = _parse_interaction_file(inter_path)
            self.n_users = int(users.max()) + 1
            self.n_items = int(items.max()) + 1
            self._uniform_split(users, items)
        else:
            self.train_users, self.train_items = _parse_interaction_file(
                os.path.join(self.dir, "train.txt"))
            self.valid_users, self.valid_items = _parse_interaction_file(
                os.path.join(self.dir, "valid.txt"))
            self.test_users, self.test_items = _parse_interaction_file(
                os.path.join(self.dir, "test.txt"))
            all_u = np.concatenate([self.train_users, self.valid_users, self.test_users])
            all_i = np.concatenate([self.train_items, self.valid_items, self.test_items])
            self.n_users = int(all_u.max()) + 1
            self.n_items = int(all_i.max()) + 1

    def _uniform_split(self, users, items):
        """Unbiased 80/10/10 split of MACR: interactions are sampled uniformly
        at random, regardless of item popularity."""
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(users))
        users, items = users[perm], items[perm]
        n = len(users)
        n_test = int(0.1 * n)
        n_valid = int(0.1 * n)
        self.test_users, self.test_items = users[:n_test], items[:n_test]
        self.valid_users, self.valid_items = users[n_test:n_test + n_valid], items[n_test:n_test + n_valid]
        self.train_users, self.train_items = users[n_test + n_valid:], items[n_test + n_valid:]

    def _generate_sample(self):
        users, items, self.n_users, self.n_items = generate_synthetic(self.name, seed=self.seed)
        path = os.path.join(self.dir, "interactions.txt")
        with open(path, "w") as f:
            for u, i in zip(users, items):
                f.write(f"{u} {i}\n")
        print(f"[data] synthetic sample saved to {path} "
              f"({len(users)} interactions); delete it to regenerate.")
        self._uniform_split(users, items)

    def _to_csr(self, users, items):
        return sp.csr_matrix(
            (np.ones(len(users), dtype=np.float32), (users, items)),
            shape=(self.n_users, self.n_items))

    def user_pos_items(self):
        """List of training item arrays per user (for negative sampling)."""
        csr = self.train_csr
        return [csr.indices[csr.indptr[u]:csr.indptr[u + 1]] for u in range(self.n_users)]
