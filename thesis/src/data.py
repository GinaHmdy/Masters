"""Dataset splits, read exactly as the previous harness (LightGCN/code/dataloader.py) read them.

File format, one line per user: `uid item item ...` (space separated, 0-based ids).
- n_users / n_items = 1 + max id seen in ANY of train/valid/test (as the old Loader).
- Lines with no items are skipped (the old Loader did this for valid.txt only; train/test
  contain no such lines, so this is equivalent).
- KuaiRec's train.txt repeats some (user, item) pairs. `train` is binarised (used for masking and
  positive sets); `train_counts` keeps the multiplicity exactly as the official/old Loader builds
  UserItemNet (csr_matrix((ones, (u, i))) sums duplicates), and is what graphs, degrees and
  popularity features are built from. `item_pop` is the counted degree (UserItemNet column sum),
  as used by the old harness for AvgPop and the long-tail split. Other datasets have no repeats,
  so train == train_counts there.
- KuaiRec is ranked only against the items of the evaluated split itself (its valid/test come
  from the dense "small matrix", ~30% of the catalogue). The universe is per split, so test
  items never leak into validation. Every other dataset ranks over all items (None).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DATASETS = ("Gowalla", "Yahoo", "KuaiRec", "Coat")
RESTRICTED_CANDIDATES = {"KuaiRec"}


@dataclass
class Split:
    name: str
    n_users: int
    n_items: int
    train: sp.csr_matrix                  # users x items, binary float32
    valid: dict[int, np.ndarray]          # user -> item ids
    test: dict[int, np.ndarray]
    item_pop: np.ndarray                  # counted train degree per item (UserItemNet col sum), int64
    valid_candidates: np.ndarray | None   # item ids allowed in ranking for valid (None = all)
    test_candidates: np.ndarray | None
    train_counts: sp.csr_matrix | None = None   # users x items, multiplicity of each train pair


def _read(path: Path) -> dict[int, np.ndarray]:
    out: dict[int, list[int]] = {}
    with open(path) as f:
        for line in f:
            tok = line.split()
            if len(tok) <= 1:
                continue
            out.setdefault(int(tok[0]), []).extend(int(t) for t in tok[1:])
    return {u: np.asarray(v, dtype=np.int64) for u, v in out.items()}


def _universe(d: dict[int, np.ndarray]) -> np.ndarray:
    return np.unique(np.concatenate(list(d.values()))).astype(np.int64)


def load(name: str, root: str | Path = DATA_ROOT) -> Split:
    root = Path(root) / name
    tr, va, te = (_read(root / f"{s}.txt") for s in ("train", "valid", "test"))
    parts = (tr, va, te)
    n_users = 1 + max(max(d) for d in parts)
    n_items = 1 + max(int(v.max()) for d in parts for v in d.values())

    rows = np.concatenate([np.full(len(v), u, dtype=np.int64) for u, v in tr.items()])
    cols = np.concatenate(list(tr.values()))
    train_counts = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                 shape=(n_users, n_items))   # sums duplicates, as the official Loader
    train_counts.sum_duplicates()
    train = train_counts.copy()
    train.data[:] = 1.0
    item_pop = np.asarray(train_counts.sum(axis=0)).ravel().astype(np.int64)

    restrict = name in RESTRICTED_CANDIDATES
    return Split(
        name=name, n_users=n_users, n_items=n_items, train=train,
        valid=va, test=te, item_pop=item_pop,
        valid_candidates=_universe(va) if restrict else None,
        test_candidates=_universe(te) if restrict else None,
        train_counts=train_counts,
    )


if __name__ == "__main__":
    for ds in DATASETS:
        s = load(ds)
        zero = int((np.diff(s.train.indptr) == 0).sum())
        print(f"{ds:8s} users={s.n_users} items={s.n_items} train_nnz={s.train.nnz} "
              f"train_interactions={int(s.train_counts.sum())} "
              f"valid_users={len(s.valid)} test_users={len(s.test)} users_without_train={zero} "
              f"cand(valid/test)={None if s.valid_candidates is None else len(s.valid_candidates)}"
              f"/{None if s.test_candidates is None else len(s.test_candidates)}")
