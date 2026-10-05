"""
Train and evaluate ASTD-PD for one dataset and seed (port of main.py's PPD v6 branch and
Procedure.train_ppd_v6).

    python -m src.astdpd.train --dataset Coat --seed 2024        # run from thesis/

Writes runs/<dataset>/seed<seed>/astdpd.pt (best-epoch state_dict), astdpd.json (config,
training history, valid + test metrics) and astdpd_per_user_test.npz (for significance tests).
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src import data
from src.evaluate import evaluate
from src.astdpd.model import ASTDPD
from src.astdpd.sampler import StratifiedNegativeSampler

ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


SELECTED = ROOT / "configs" / "astdpd_selected.yaml"   # written by src/astdpd/tune.py


def load_config(dataset, path=ROOT / "configs" / "astdpd.yaml", selected=True):
    """The dataset's configuration: the validation-retuned one (configs/astdpd_selected.yaml)
    when src/astdpd/tune.py has written one for `dataset`, otherwise the shipped astdpd.yaml.
    selected=False always returns the shipped configuration."""
    cfg = yaml.safe_load(open(path))
    if selected and SELECTED.exists():
        sel = yaml.safe_load(open(SELECTED)) or {}
        if dataset in sel:
            return {**cfg["defaults"], **sel[dataset]["config"]}
    return {**cfg["defaults"], **cfg[dataset]}


def pareto_hypervolume(m, max_pop_ref=1000.0):
    """Validation selection criterion: volume dominated by (Recall, NDCG, Precision, 1/(1+AvgPop/ref))
    above the origin (utilities.compute_pareto_hypervolume)."""
    inv_pop = 1.0 / (1.0 + m["avgpop@20"] / max_pop_ref)
    return float(m["recall@20"] * m["ndcg@20"] * m["precision@20"] * inv_pop)


def train_epoch(model, optimizer, sampler, users, items, batch_size, device):
    model.train()
    perm = np.random.permutation(len(users))
    users, items = users[perm], items[perm]
    sums, n = {}, len(users)
    for s in range(0, n, batch_size):
        bu, bp = users[s:s + batch_size], items[s:s + batch_size]
        bn, logq = sampler.sample(bu, bp, num_negatives=1)
        t = lambda a, dt=torch.long: torch.as_tensor(a, dtype=dt, device=device)
        optimizer.zero_grad()
        loss, stats = model.loss(t(bu), t(bp), t(bn), t(logq, torch.float32))
        loss.backward()
        optimizer.step()
        model.train_step_count += 1
        if model.train_step_count % max(1, model.hp["pop_basis_update_steps"]) == 0:
            model.update_popularity_basis()
        for k, v in {"loss": loss.item(), **stats}.items():
            sums[k] = sums.get(k, 0.0) + v * len(bu)
    return {k: v / max(1, n) for k, v in sums.items()}


def run(dataset, seed, out_dir=None, epochs=None, device=None, split=None, verbose=True, cfg=None):
    """cfg: full configuration overriding load_config(dataset) (used by src/astdpd/tune.py)."""
    t_start = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = cfg or load_config(dataset)
    epochs = epochs or cfg["epochs"]
    set_seed(seed)
    split = split or data.load(dataset)
    train = split.train.tocsr()   # binary: positive sets of the sampler
    # counted (UserItemNet of the old harness): graphs, degrees, features and training pairs
    counts = (split.train_counts if split.train_counts is not None else split.train).tocsr()

    model = ASTDPD(counts, device, latent_dim=cfg["latent_dim"], n_layers=cfg["n_layers"],
                   decay=cfg["decay"], **cfg["model"])
    sampler = StratifiedNegativeSampler(model.item_decile.numpy(), train, **cfg["sampler"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    coo = counts.tocoo()  # one training pair per occurrence in train.txt (row-major order)
    reps = coo.data.astype(np.int64)
    users, items = np.repeat(coo.row.astype(np.int64), reps), np.repeat(coo.col.astype(np.int64), reps)

    warmup = min(10, epochs // 2)
    patience, best = cfg["patience"], dict(hv=-1.0, epoch=-1, state=None, valid=None)
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        stats = train_epoch(model, optimizer, sampler, users, items, cfg["batch_size"], device)
        t_train = time.time() - t0
        val = evaluate(model.scorer(), split, "valid", ks=(20,), device=device)
        hv = pareto_hypervolume(val)
        history.append(dict(epoch=epoch, train_seconds=t_train, eval_seconds=time.time() - t0 - t_train,
                            hv=hv, **{k: val[k] for k in ("recall@20", "ndcg@20", "precision@20", "avgpop@20")},
                            **stats))
        if verbose:
            print(f"epoch {epoch:3d} loss {stats['loss']:.4f} | valid R@20 {val['recall@20']:.5f} "
                  f"N@20 {val['ndcg@20']:.5f} AvgPop {val['avgpop@20']:.1f} HV {hv:.3e} "
                  f"| {t_train:.1f}s train", flush=True)
        # during warm-up the latest epoch is always kept; afterwards only improvements
        if epoch < warmup or hv > best["hv"]:
            best.update(hv=hv, epoch=epoch, state=copy.deepcopy(model.state_dict()), valid=val)
            if epoch >= warmup:
                patience = cfg["patience"]
        else:
            patience -= 1
            if patience == 0:
                break

    model.load_state_dict(best["state"])
    score = model.scorer()
    valid = evaluate(score, split, "valid", device=device)
    test = evaluate(score, split, "test", device=device, return_per_user=True)
    per_user = test.pop("per_user")

    if out_dir is None:
        out_dir = ROOT / "runs" / dataset / f"seed{seed}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], out_dir / "astdpd.pt")
    np.savez_compressed(out_dir / "astdpd_per_user_test.npz", **per_user)
    result = dict(method="ASTD-PD", dataset=dataset, seed=seed, config=cfg, epochs_run=len(history),
                  best_epoch=best["epoch"], best_valid_hypervolume=best["hv"],
                  valid=valid, test=test, wall_time_s=time.time() - t_start, history=history)
    with open(out_dir / "astdpd.json", "w") as f:
        json.dump(result, f, indent=2)
    if verbose:
        print(f"best epoch {best['epoch']} | test R@20 {test['recall@20']:.5f} N@20 {test['ndcg@20']:.5f} "
              f"AvgPop@20 {test['avgpop@20']:.2f} -> {out_dir}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--epochs", type=int, default=None, help="override the configured epoch budget")
    ap.add_argument("--out", default=None, help="output directory (default runs/<dataset>/seed<seed>)")
    args = ap.parse_args()
    run(args.dataset, args.seed, args.out, args.epochs)
