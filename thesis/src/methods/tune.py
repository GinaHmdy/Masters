"""Validation grid search for a post-hoc baseline on one (dataset, seed) backbone.

    python -m src.methods.tune --dataset Coat --seed 2024 --method ppd     # run from thesis/

Every configuration of configs/grids.yaml[<method>] is evaluated on VALIDATION only; the winner is
the highest valid Recall@20 (ties -> NDCG@20; remaining ties -> first in grid order). The winner is
then evaluated once on test. A configuration that raises inside the official code (e.g. DAP's
Kmeans on NaN anchors) is recorded with its error and excluded from selection.
Each grid point's validation row is cached in runs/<ds>/seed<seed>/grid/<method>__<hash>.json,
tagged with the backbone checkpoint id; `only` evaluates a subset of points (no selection), so
a grid can be split over processes and finished by a final call that finds every row cached.
Writes runs/<ds>/seed<seed>/<method>_selected.json (full grid table, selection, valid + test
metrics) and <method>_per_user_test.npz (per-user test vectors for src/significance.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback

import numpy as np
import torch

from . import KS, METHODS, RUNS, grid, hp_hash, method_module


def grid_id(configs) -> str:
    return hashlib.sha256(json.dumps(configs, sort_keys=True).encode()).hexdigest()[:10]


def is_done(dataset: str, seed: int, method: str) -> bool:
    """<method>_selected.json exists for the current backbone checkpoint and grid."""
    from src.backbone import checkpoint_id
    f = RUNS / dataset / f"seed{seed}" / f"{method}_selected.json"
    if not f.exists():
        return False
    r = json.load(open(f))
    return (r.get("backbone_id") == checkpoint_id(dataset, seed)
            and r.get("grid_id") == grid_id(grid(method, dataset)))


def tune(dataset: str, seed: int, method: str, limit: int | None = None, only: dict | None = None,
         device: str | None = None, verbose=True) -> dict | None:
    from src.backbone import checkpoint_id, load_backbone
    from src.evaluate import evaluate
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    bid = checkpoint_id(dataset, seed)
    net, split = load_backbone(dataset, seed, device)
    mod = method_module(method)
    configs = grid(method, dataset)[:limit]
    todo = [hp for hp in configs if all(hp.get(k) == v for k, v in (only or {}).items())]
    out_dir = RUNS / dataset / f"seed{seed}"
    (out_dir / "grid").mkdir(parents=True, exist_ok=True)
    table, t_all = [], time.time()
    for i, hp in enumerate(configs):
        cache = out_dir / "grid" / f"{method}__{hp_hash(method, hp)}.json"
        if cache.exists():
            row = json.load(open(cache))
            if row.get("backbone_id") == bid:
                table.append(row)
                continue
        if hp not in todo:
            continue
        row = {"hparams": hp, "backbone_id": bid}
        t0 = time.time()
        try:
            fn = mod.build_scorer(net, split, device, **hp)
            row["build_time_s"] = time.time() - t0
            v = evaluate(fn, split, "valid", ks=KS, device=device)
            row.update({k: v[k] for k in ("recall@20", "ndcg@20", "precision@20", "avgpop@20",
                                          "aplt@20", "coverage@20", "tail_recall@20")})
            for extra in ("last_info", "last_timing"):
                if hasattr(mod, extra) and getattr(mod, extra):
                    row[extra] = dict(getattr(mod, extra))
            del fn
        except Exception as e:  # recorded, never selected
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc(limit=3)
        row["time_s"] = time.time() - t0
        json.dump(row, open(cache, "w"), indent=2, default=float)
        table.append(row)
        if verbose:
            msg = row.get("error") or (f"R@20 {row['recall@20']:.5f} N@20 {row['ndcg@20']:.5f} "
                                       f"AvgPop {row['avgpop@20']:.2f}")
            print(f"[{method} {dataset} s{seed}] {i + 1}/{len(configs)} {hp} -> {msg} "
                  f"({row['time_s']:.1f}s)", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if only is not None:
        return None  # partial run: rows cached, selection happens in the final call

    ok = [r for r in table if "error" not in r and np.isfinite(r["recall@20"])]
    if not ok:
        raise RuntimeError(f"every {method} configuration failed on {dataset}")
    rank = {hp_hash(method, hp): j for j, hp in enumerate(configs)}  # grid order for ties
    table.sort(key=lambda r: rank[hp_hash(method, r["hparams"])])
    ok.sort(key=lambda r: rank[hp_hash(method, r["hparams"])])
    best = max(ok, key=lambda r: (r["recall@20"], r["ndcg@20"]))  # max keeps the first of ties
    hp = best["hparams"]
    t0 = time.time()
    fn = mod.build_scorer(net, split, device, **hp)
    valid = evaluate(fn, split, "valid", ks=KS, device=device)
    test = evaluate(fn, split, "test", ks=KS, return_per_user=True, device=device)
    per_user = test.pop("per_user")
    np.savez_compressed(out_dir / f"{method}_per_user_test.npz", **per_user)
    res = dict(method=method, dataset=dataset, seed=seed, backbone_id=bid,
               grid_id=grid_id(configs), selection="valid recall@20, ties -> valid ndcg@20",
               selected_hparams=hp, valid=valid, test=test, n_configs=len(configs),
               n_failed=len(table) - len(ok),
               # compute time of the whole grid (sum over points, wherever they ran) + final eval
               grid_time_s=sum(r["time_s"] for r in table) + (time.time() - t0),
               wall_time_this_call_s=time.time() - t_all, grid=table)
    with open(out_dir / f"{method}_selected.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    if verbose:
        print(f"[{method} {dataset} s{seed}] selected {hp} | valid R@20 {valid['recall@20']:.5f} | "
              f"test R@20 {test['recall@20']:.5f} N@20 {test['ndcg@20']:.5f} "
              f"AvgPop@20 {test['avgpop@20']:.2f} -> {out_dir / f'{method}_selected.json'}",
              flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--limit", type=int, default=None, help="only the first N grid points (debug)")
    a = ap.parse_args()
    tune(a.dataset, a.seed, a.method, a.limit)
