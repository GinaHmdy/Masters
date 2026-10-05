"""
Stratified popularity negative sampler with exact logQ (port of
StratifiedPopularityNegativeSampler, LightGCN/code/dataloader.py).

Each negative is drawn from a mixture: uniform over the catalogue (r_u), uniform within the
positive's popularity decile (r_m), or uniform within the top decile ("hard", r_h). Rejection of
the user's training positives is capped at 50 tries with the original fallbacks. The returned
log q(j) is the exact mixture probability of the drawn item, used to correct sampled softmax.

The per-sample loop and the order of np.random calls are kept as in the original so a given
seed produces the same draws; only membership tests were made O(1).
"""
import numpy as np


class StratifiedNegativeSampler:
    def __init__(self, item_decile, train, uniform_ratio=0.4, matched_ratio=0.4, hard_ratio=0.2):
        self.num_items = train.shape[1]
        self.pos_sets = [set(train.indices[train.indptr[u]:train.indptr[u + 1]].tolist())
                         for u in range(train.shape[0])]
        total = uniform_ratio + matched_ratio + hard_ratio
        if total <= 0:
            uniform_ratio, matched_ratio, hard_ratio, total = 1.0, 0.0, 0.0, 1.0
        self.r_u, self.r_m, self.r_h = uniform_ratio / total, matched_ratio / total, hard_ratio / total

        self.deciles = np.asarray(item_decile)
        self.decile_items = [np.where(self.deciles == d)[0] for d in range(10)]
        self.decile_items = [p if len(p) else np.arange(self.num_items) for p in self.decile_items]
        self.hard_items = self.decile_items[9]
        self.is_hard = np.zeros(self.num_items, dtype=bool)
        self.is_hard[self.hard_items] = True
        self.decile_sizes = np.array([len(p) for p in self.decile_items], dtype=np.float32)
        self.hard_size = float(len(self.hard_items))

    def _draw(self, pool, pos_set):
        """Up to 50 rejection draws from `pool` (None = whole catalogue); -1 if all were positives."""
        for _ in range(50):
            c = np.random.randint(0, self.num_items) if pool is None else np.random.choice(pool)
            if c not in pos_set:
                return c
        return -1

    def sample(self, users, pos_items, num_negatives=1):
        """Returns (negatives [B, n], log q [B, n])."""
        B = len(users)
        neg = np.zeros((B, num_negatives), dtype=np.int64)
        logq = np.zeros((B, num_negatives), dtype=np.float32)
        for b in range(B):
            pos_set = self.pos_sets[int(users[b])]
            pos_decile = self.deciles[int(pos_items[b])]
            matched = self.decile_items[pos_decile]
            for k in range(num_negatives):
                r = np.random.rand()
                if r < self.r_u:                                   # uniform
                    j = self._draw(None, pos_set)
                    if j == -1:
                        j = np.random.randint(0, self.num_items)
                elif r < self.r_u + self.r_m:                      # popularity-matched
                    j = self._draw(matched, pos_set)
                    if j == -1:
                        j = self._draw(None, pos_set)
                        if j == -1:
                            j = np.random.randint(0, self.num_items)
                else:                                              # hard: top decile
                    j = self._draw(self.hard_items, pos_set)
                    if j == -1:
                        c = np.random.choice(matched)
                        j = c if c not in pos_set else np.random.randint(0, self.num_items)
                neg[b, k] = j
                q_unif = 1.0 / max(1, self.num_items)
                q_matched = (1.0 / max(1.0, self.decile_sizes[pos_decile])) if self.deciles[j] == pos_decile else 0.0
                q_hard = (1.0 / max(1.0, self.hard_size)) if self.is_hard[j] else 0.0
                logq[b, k] = np.log(max(self.r_u * q_unif + self.r_m * q_matched + self.r_h * q_hard, 1e-8))
        return neg, logq
