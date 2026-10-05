"""Post-hoc / inference-time baselines applied to the trained official LightGCN backbone.

Each method module (ppd, tsp, dap) exposes

    build_scorer(model, split, device, **hparams) -> score_fn      # score_fn(user_ids) -> [B, n_items]

where `model` is the official LightGCN-PyTorch model returned by src.backbone.load_backbone (it is
never modified: methods work on copies of its weights) and score_fn is what src.evaluate.evaluate
expects. `python -m src.methods.<m> --dataset D --seed S [--hparams k=v ...]` evaluates one
configuration on valid + test and writes runs/<D>/seed<S>/<m>__<hash>.json.

The vendored repositories ship modules with clashing top-level names (world, utils, model,
dataloader, Procedure, ...). `isolated_import` imports a vendored module with its own siblings
visible, then restores sys.modules/sys.path, so the LightGCN-PyTorch modules already imported by
src.backbone are untouched. Imported functions keep references to their own module globals, so
they keep working after the swap. Nothing is written into third_party (no bytecode, no dirs).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

THESIS = Path(__file__).resolve().parents[2]
THIRD = THESIS / "third_party"
RUNS = THESIS / "runs"
GRIDS = THESIS / "configs" / "grids.yaml"
METHODS = ("ppd", "tsp", "dap")
KS = (10, 20)


@contextlib.contextmanager
def isolated_modules(path: Path | None, names, stubs: dict | None = None):
    """Inside the block, `names` resolve to fresh imports from `path` (or to `stubs`); on exit the
    previous sys.modules entries, sys.path and sys.dont_write_bytecode are restored."""
    names = set(names) | set(stubs or {})
    saved = {n: sys.modules.pop(n) for n in list(sys.modules)
             if n in names or n.split(".")[0] in names}
    path_before, dwb = list(sys.path), sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # keep third_party byte-identical (no __pycache__)
    if path is not None:
        sys.path.insert(0, str(path))
    sys.modules.update(stubs or {})
    try:
        yield
    finally:
        for n in list(sys.modules):
            if n in names or n.split(".")[0] in names:
                del sys.modules[n]
        sys.modules.update(saved)
        sys.path[:] = path_before
        sys.dont_write_bytecode = dwb


def load_file(path: Path, module_name: str):
    """Import a single vendored file under a private module name."""
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hp_hash(method: str, hp: dict) -> str:
    return hashlib.sha256(json.dumps({"method": method, **hp}, sort_keys=True).encode()).hexdigest()[:10]


def grid(method: str, dataset: str) -> list[dict]:
    """Cartesian product of configs/grids.yaml[method] (+ per-dataset overrides)."""
    import itertools
    cfg = yaml.safe_load(open(GRIDS))[method]
    space = {**cfg["grid"], **(cfg.get("overrides", {}) or {}).get(dataset, {})}
    fixed = {**cfg.get("fixed", {}), **(cfg.get("fixed_overrides", {}) or {}).get(dataset, {})}
    keys = list(space)
    return [{**fixed, **dict(zip(keys, v))} for v in itertools.product(*(space[k] for k in keys))]


def method_module(method: str):
    return importlib.import_module(f"src.methods.{method}")


def evaluate_config(method: str, net, split, hp: dict, parts=("valid", "test"), device="cuda"):
    from src.evaluate import evaluate
    mod = method_module(method)
    t0 = time.time()
    score_fn = mod.build_scorer(net, split, device, **hp)
    t_build = time.time() - t0
    out = {"method": method, "dataset": split.name, "hparams": hp, "build_time_s": t_build}
    for p in parts:
        t1 = time.time()
        out[p] = evaluate(score_fn, split, p, ks=KS)
        out[f"{p}_eval_time_s"] = time.time() - t1
    return out


def parse_hparams(items) -> dict:
    hp = {}
    for it in items or []:
        k, v = it.split("=", 1)
        hp[k] = yaml.safe_load(v)
    return hp


def main(method: str, argv=None):
    ap = argparse.ArgumentParser(description=f"Evaluate {method} on a trained LightGCN backbone")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hparams", nargs="*", default=[], help="k=v pairs (YAML values)")
    a = ap.parse_args(argv)
    from src.backbone import load_backbone
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, split = load_backbone(a.dataset, a.seed, device)
    mod = method_module(method)
    hp = {**mod.DEFAULTS, **parse_hparams(a.hparams)}
    res = evaluate_config(method, net, split, hp, device=device)
    res["seed"] = a.seed
    out = RUNS / a.dataset / f"seed{a.seed}" / f"{method}__{hp_hash(method, hp)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps({"out": str(out), "hparams": hp, "build_time_s": round(res["build_time_s"], 2),
                      **{f"{p}_{m}": round(res[p][m], 5) for p in ("valid", "test")
                         for m in ("recall@20", "ndcg@20", "avgpop@20")}}), flush=True)
    return res
