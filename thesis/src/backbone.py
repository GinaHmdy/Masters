"""Train the official LightGCN (third_party/LightGCN-PyTorch/code) on a data.Split.

The official modules are imported unmodified. Their import-time side effects are handled here:
  * world.py parses sys.argv and only accepts its own dataset names -> argv is set to the run's
    hyperparameters with a placeholder `--dataset gowalla` (only used for naming there);
  * world.py creates code/checkpoints/ and Python would write __pycache__ -> both suppressed so
    the vendored tree stays byte-identical;
  * utils.py compiles sources/sampling.cpp with cppimport if it can (writing the .so into the
    vendored tree, and crashing on users without training items, e.g. one KuaiRec user). We
    block cppimport so the official fallback `UniformSample_original_python` is always used --
    deterministic under np.random.seed and identical on every machine.
The data is handed over through `SplitDataset`, a subclass of the official dataloader.Loader that
fills its attributes from a Split instead of reading files, so the official getSparseGraph
(normalised adjacency) and getUserPosItems are used as-is.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml

from . import data as D

THESIS = Path(__file__).resolve().parents[1]
OFFICIAL = THESIS / "third_party" / "LightGCN-PyTorch" / "code"
RUNS = THESIS / "runs"
CONFIG = THESIS / "configs" / "backbone.yaml"

_official = None


def hparams(ds: str) -> dict:
    cfg = yaml.safe_load(open(CONFIG))
    return {**cfg["defaults"], **cfg[ds]}


def _argv(hp: dict, seed: int) -> list[str]:
    return ["lightgcn", "--dataset", "gowalla", "--model", "lgn", "--tensorboard", "0",
            "--layer", str(hp["layer"]), "--recdim", str(hp["recdim"]), "--lr", str(hp["lr"]),
            "--decay", str(hp["decay"]), "--bpr_batch", str(hp["bpr_batch"]),
            "--dropout", str(hp["dropout"]), "--keepprob", str(hp["keepprob"]),
            "--epochs", str(hp["epochs"]), "--seed", str(seed), "--topks", "[20]"]


def official(hp: dict, seed: int):
    """Import (once) and configure the official modules; returns a namespace of them.

    world.config is the official config dict; it is refreshed from the official parser on every
    call so that one process can build models for several datasets.
    """
    global _official
    argv, sys.argv = sys.argv, _argv(hp, seed)
    try:
        if _official is None:
            real_makedirs, dont_write = os.makedirs, sys.dont_write_bytecode

            def makedirs(p, *a, **k):  # world.py: os.makedirs(code/checkpoints)
                if not str(Path(p).resolve()).startswith(str(OFFICIAL)):
                    real_makedirs(p, *a, **k)

            sys.path.insert(0, str(OFFICIAL))
            sys.modules["cppimport"] = None  # -> ImportError -> official Python sampler
            os.makedirs, sys.dont_write_bytecode = makedirs, True
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    import world, dataloader, model, utils, Procedure  # noqa: E401
            finally:
                os.makedirs, sys.dont_write_bytecode = real_makedirs, dont_write
            assert not utils.sample_ext
            _official = argparse.Namespace(world=world, dataloader=dataloader, model=model,
                                           utils=utils, Procedure=Procedure)
        w = _official.world
        args = w.parse_args()  # official parser, fed the argv above
        w.config.update(bpr_batch_size=args.bpr_batch, latent_dim_rec=args.recdim,
                        lightGCN_n_layers=args.layer, dropout=args.dropout,
                        keep_prob=args.keepprob, lr=args.lr, decay=args.decay)
        w.seed, w.TRAIN_epochs, w.tensorboard = args.seed, args.epochs, args.tensorboard
    finally:
        sys.argv = argv
    return _official


def make_dataset(off, split: D.Split):
    """An official dataloader.Loader populated from `split` (its __init__ reads files; skipped)."""

    class SplitDataset(off.dataloader.Loader):
        def __init__(self, s: D.Split):
            # Counted matrix, exactly as the official Loader builds UserItemNet from train.txt:
            # csr_matrix((ones, (trainUser, trainItem))) sums repeated pairs (KuaiRec) into weights.
            counts = s.train_counts if s.train_counts is not None else s.train
            R = counts.tocsr().astype(np.float64)
            self.n_user, self.m_item = s.n_users, s.n_items
            self.UserItemNet = R
            coo = R.tocoo()
            reps = coo.data.astype(np.int64)   # one (user, item) row per occurrence in train.txt
            self.trainUser = np.repeat(coo.row.astype(np.int64), reps)
            self.trainItem = np.repeat(coo.col.astype(np.int64), reps)
            self.trainUniqueUsers = np.unique(self.trainUser)
            self.traindataSize = len(self.trainUser)
            self.split, self.folds = off.world.config["A_split"], off.world.config["A_n_fold"]
            self.mode_dict, self.mode = {"train": 0, "test": 1}, 0
            self.Graph = None
            # official getSparseGraph caches the normalised adjacency at self.path
            h = hashlib.sha256(R.indptr.tobytes() + R.indices.tobytes() + R.data.tobytes()).hexdigest()[:16]
            self.path = str(RUNS / ".cache" / "adj" / f"{s.name}-{h}")
            os.makedirs(self.path, exist_ok=True)
            self.users_D = np.maximum(np.asarray(R.sum(1)).ravel(), 1)
            self.items_D = np.maximum(np.asarray(R.sum(0)).ravel(), 1)
            self._allPos = self.getUserPosItems(list(range(self.n_user)))
            self._Loader__testDict = {u: list(v) for u, v in s.test.items()}

    with contextlib.redirect_stdout(sys.stderr):
        return SplitDataset(split)


def set_seed(seed: int):
    random.seed(seed)
    _official.utils.set_seed(seed)  # official: numpy, torch, cuda
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build(ds: str, seed: int, split: D.Split | None = None):
    hp = hparams(ds)
    off = official(hp, seed)
    split = split or D.load(ds)
    set_seed(seed)  # before model init: embeddings are drawn from torch's RNG
    dataset = make_dataset(off, split)
    with contextlib.redirect_stdout(sys.stderr):
        net = off.model.LightGCN(off.world.config, dataset).to(off.world.device)
    return off, hp, split, dataset, net


@torch.no_grad()
def scorer(net):
    """score_fn for evaluate(): raw LightGCN inner products (same ranking as the official
    sigmoid in getUsersRating, without fp32 saturation ties)."""
    net.eval()
    users, items = net.computer()
    return lambda u: users[u.to(users.device)] @ items.T


def _valid_recall20(net, split) -> float:
    try:
        from .evaluate import evaluate
    except ImportError:
        evaluate = None
    if evaluate is not None:
        return evaluate(scorer(net), split, "valid", ks=(20,))["recall@20"]
    raise RuntimeError("src/evaluate.py missing")


def train(ds: str, seed: int, out: Path | None = None) -> dict:
    t0 = time.time()
    off, hp, split, dataset, net = build(ds, seed)
    bpr = off.utils.BPRLoss(net, off.world.config)
    best, best_epoch, best_state, curve, bad = -1.0, 0, None, [], 0
    for epoch in range(1, hp["epochs"] + 1):
        te = time.time()
        info = off.Procedure.BPR_train_original(dataset, net, bpr, epoch, neg_k=1, w=None)
        if epoch % hp["eval_every"] and epoch != hp["epochs"]:
            continue
        r = _valid_recall20(net, split)
        curve.append((epoch, r))
        print(f"[{ds} s{seed}] epoch {epoch} {info} epoch_time={time.time() - te:.1f}s "
              f"valid R@20={r:.4f}", flush=True)
        if r > best:
            best, best_epoch, bad = r, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    out = out or RUNS / ds / f"seed{seed}" / "lightgcn.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset=ds, seed=seed, config=hp, best_epoch=best_epoch, best_valid_recall20=best,
                valid_curve=curve, sampler="UniformSample_original_python",
                wall_time_s=time.time() - t0, torch=torch.__version__)
    torch.save({"state_dict": best_state, **meta}, out)
    print(json.dumps({k: v for k, v in meta.items() if k != "valid_curve"}), flush=True)
    return meta


def load_backbone(ds: str, seed: int, device="cuda"):
    """(official model.LightGCN in eval mode with the selected weights, Split)."""
    ckpt = torch.load(RUNS / ds / f"seed{seed}" / "lightgcn.pt", map_location="cpu",
                      weights_only=False)
    off, hp, split, _, net = build(ds, seed)
    net.load_state_dict(ckpt["state_dict"])
    net = net.to(device)
    net.Graph = net.Graph.to(device)  # plain attribute, not moved by .to()
    return net.eval(), split


def checkpoint_id(ds: str, seed: int) -> str:
    """Content hash of runs/<ds>/seed<seed>/lightgcn.pt; downstream outputs record it."""
    return hashlib.sha256((RUNS / ds / f"seed{seed}" / "lightgcn.pt").read_bytes()).hexdigest()[:12]


def is_trained(ds: str, seed: int) -> bool:
    """A checkpoint is reusable if it was trained with the current hyperparameters, or with the
    same ones except a smaller epoch cap that it never reached (it stopped early, so a larger
    cap replays the identical run). A run that hit its cap is retrained."""
    f = RUNS / ds / f"seed{seed}" / "lightgcn.pt"
    if not f.exists():
        return False
    ck = torch.load(f, map_location="cpu", weights_only=False)
    hp, old = hparams(ds), dict(ck["config"])
    if old == hp:
        return True
    last_epoch = ck["valid_curve"][-1][0]
    return ({**old, "epochs": hp["epochs"]} == hp and last_epoch < old["epochs"]
            and last_epoch <= hp["epochs"])


def evaluate_backbone(ds: str, seed: int, device="cuda") -> dict:
    """Valid + test metrics of the selected backbone -> lightgcn.json, lightgcn_per_user_test.npz."""
    from .evaluate import evaluate
    t0 = time.time()
    net, split = load_backbone(ds, seed, device)
    ck = torch.load(RUNS / ds / f"seed{seed}" / "lightgcn.pt", map_location="cpu", weights_only=False)
    fn = scorer(net)
    valid = evaluate(fn, split, "valid", ks=(10, 20), device=device)
    test = evaluate(fn, split, "test", ks=(10, 20), return_per_user=True, device=device)
    out = RUNS / ds / f"seed{seed}"
    np.savez_compressed(out / "lightgcn_per_user_test.npz", **test.pop("per_user"))
    res = dict(method="LightGCN", dataset=ds, seed=seed, backbone_id=checkpoint_id(ds, seed),
               config=ck["config"], best_epoch=ck["best_epoch"],
               epochs_run=ck["valid_curve"][-1][0], best_valid_recall20=ck["best_valid_recall20"],
               train_wall_time_s=ck["wall_time_s"], eval_time_s=time.time() - t0,
               valid=valid, test=test)
    json.dump(res, open(out / "lightgcn.json", "w"), indent=2)
    print(f"[{ds} s{seed}] LightGCN best epoch {res['best_epoch']}/{res['epochs_run']} | "
          f"test R@20 {test['recall@20']:.5f} N@20 {test['ndcg@20']:.5f} "
          f"AvgPop@20 {test['avgpop@20']:.2f}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=D.DATASETS)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    train(a.dataset, a.seed)
