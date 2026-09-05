"""
Phase 5: re-select PPD v6's hyperparameters on VALIDATION (remediation item 5.7).

Why this exists. The configurations in `experiment_configs.py` are not all validation-selected.
`sweep_yahoo_ppd_v6.py` did rank its 108 configurations by validation hypervolume, but the
configuration actually shipped came from `refine_yahoo_ppd_v6.py`, whose `score_fn` sorts
candidates by **seed-2024 test metrics** (`refine_yahoo_ppd_v6.py:171-176`). The evidence is
direct: the validation winner of the recorded sweep is alpha_base 0.15 / damping 0.05, while the
shipped Yahoo config is 0.10 / 0.10. Every v6 number therefore rests on hyperparameters that saw
the test set. Same class of defect as findings F and X, one level up.

What this does differently:

  1. **Selection reads the `[Best Valid Results]` block only.** Test metrics are parsed and
     recorded for reporting, never compared. The selection statistic is the same 4-D Pareto
     hypervolume `main.py:144` uses for checkpoint selection, so config selection and checkpoint
     selection agree on what "better" means.
  2. **Dead knobs are not swept.** `astd_topo_gate_init` has no effect while
     `astd_topo_gate_learnable=1` (item 5.6) -- sweeping it is what wasted two thirds of the
     original Yahoo sweep. It is fixed here, not varied.
  3. **The incumbent configuration is a grid point** on every dataset, so the sweep can confirm
     or overturn the shipped values rather than merely proposing different ones.

One seed (2024) per configuration for the search -- the winner is then re-run at 5 seeds by
`run_phase1_v6.py` before any number is reported.

Usage:
    python run_phase5_hpo.py Yahoo
    python run_phase5_hpo.py Gowalla --shard 0 --nshards 6 --out-suffix _shard0
"""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_configs import DATASET_CONFIGS
from run_phase1_baselines import RESULTS_DIR
from run_phase2_ablation import PYTHON, CODE_DIR, CKPT_DIR

# Live knobs only. Each grid contains the dataset's incumbent values, marked below, so the
# result is a re-selection over a space that includes what is currently shipped.
GRIDS = {
    # incumbent: alpha 0.40, phi 1.5, damping 0.05
    "Gowalla": {"astd_alpha_base": [0.2, 0.4, 0.6],
                "astd_tail_phi": [1.0, 1.5],
                "astd_topo_pop_damping": [0.05, 0.10]},
    # incumbent: alpha 0.10, phi 1.0, damping 0.10, fuse_max 0.5
    "Yahoo":   {"astd_alpha_base": [0.05, 0.10, 0.15],
                "astd_tail_phi": [0.5, 1.0, 1.5],
                "astd_topo_pop_damping": [0.05, 0.10],
                "astd_fuse_lambda_max": [0.3, 0.5]},
    # incumbent: alpha 0.10, phi 1.5, damping 0.05, fuse_max 0.30
    "Coat":    {"astd_alpha_base": [0.05, 0.10, 0.20],
                "astd_tail_phi": [1.0, 1.5, 2.0],
                "astd_topo_pop_damping": [0.05, 0.10],
                "astd_fuse_lambda_max": [0.30, 0.50]},
    # incumbent: see experiment_configs; KuaiRec runs ~50 min so the grid is the small one
    "KuaiRec": {"astd_alpha_base": [0.2, 0.35, 0.5],
                "astd_tail_phi": [1.0, 1.5],
                "astd_topo_pop_damping": [0.05, 0.10]},
}


def parse_valid_block(out):
    """The ONLY block selection is allowed to read."""
    if "Best Valid Results" not in out:
        return None
    sec = out.split("Best Valid Results", 1)[1][:800]
    m = {}
    for key in ("recall@20", "ndcg@20", "precision@20", "avgpop@20"):
        hit = re.search(rf"'{key}':\s*(?:np\.float64\()?(-?[\d.eE+-]+)", sec)
        if hit:
            m[key] = float(hit.group(1))
    return m if len(m) == 4 else None


def parse_test_block(out):
    """Recorded for reporting only. main.py prints arrays over world.topks = [5,10,15,20,50]."""
    if "Best Test Results" not in out:
        return {}
    sec = out.split("Best Test Results", 1)[1]
    m = {}
    for name in ("precision", "recall", "ndcg", "gini", "coverage", "aplt", "tail_recall"):
        hit = re.search(rf"'{name}':\s*array\(\[(.*?)\]\)", sec, re.S)
        if not hit:
            continue
        vals = [float(x) for x in re.split(r"[\s,]+", hit.group(1).strip()) if x]
        if vals:
            m[f"{name}@20"] = vals[3] if len(vals) >= 4 else vals[0]
    hit = re.search(r"Final Test Average Popularity@20:\s*([\d.]+)", out)
    if hit:
        m["avgpop@20"] = float(hit.group(1))
    return m


def hypervolume(m, max_pop_ref=1000.0):
    """Identical to utilities.compute_pareto_hypervolume; see run_phase4 for why it is copied."""
    inv_pop = max(0.0, 1.0 / (1.0 + m.get("avgpop@20", 0.0) / max_pop_ref))
    return (max(0.0, m.get("recall@20", 0.0)) * max(0.0, m.get("ndcg@20", 0.0))
            * max(0.0, m.get("precision@20", 0.0)) * inv_pop)


def run_config(dataset, cfg, overrides, seed):
    params = dict(cfg["ppd_v6_params"])
    params.update(overrides)
    cmd = [PYTHON, "main.py", "--method", "ppd_v6", "--dataset", dataset,
           "--layer", str(cfg["layer"]), "--recdim", str(cfg["recdim"]),
           "--epochs", str(cfg["epochs"]), "--bpr_batch", str(cfg["bpr_batch"]),
           "--lr", str(cfg["lr"]), "--seed", str(seed)]
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]
    res = subprocess.run(cmd, cwd=CODE_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        return None, None, f"train exit {res.returncode}: {(res.stderr or '').strip().splitlines()[-2:]}"
    valid = parse_valid_block(res.stdout)
    if not valid:
        return None, None, "no parsable [Best Valid Results] block"
    return valid, parse_test_block(res.stdout), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    ds = args.dataset
    cfg = DATASET_CONFIGS[ds]
    grid = GRIDS[ds]
    keys = sorted(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    mine = [c for i, c in enumerate(combos) if i % args.nshards == args.shard]
    print(f"{ds}: {len(combos)} configs total, {len(mine)} in shard "
          f"{args.shard}/{args.nshards}", flush=True)

    path = os.path.join(RESULTS_DIR, f"phase5_hpo_{ds.lower()}{args.out_suffix}.json")
    out = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                out = json.load(f)
        except (ValueError, OSError):
            out = {}

    incumbent = {k: cfg["ppd_v6_params"].get(k) for k in keys}
    failures = []
    for i, combo in enumerate(mine, 1):
        key = json.dumps(combo, sort_keys=True)
        if key in out:
            print(f"  [{i}/{len(mine)}] cached {key}", flush=True)
            continue
        t0 = time.time()
        valid, test, err = run_config(ds, cfg, combo, args.seed)
        if err:
            print(f"  [{i}/{len(mine)}] ERROR {key}: {err}", flush=True)
            failures.append((key, err))
            continue
        hv = hypervolume(valid)
        is_inc = all(abs(float(combo[k]) - float(incumbent[k])) < 1e-9 for k in keys
                     if incumbent.get(k) is not None)
        out[key] = {"params": combo, "seed": args.seed, "valid": valid,
                    "valid_hypervolume": hv, "test": test, "is_incumbent": is_inc}
        print(f"  [{i}/{len(mine)}] {key}{' <- INCUMBENT' if is_inc else ''}\n"
              f"        valid HV={hv:.8f} R@20={valid['recall@20']:.6f} "
              f"AvgPop={valid['avgpop@20']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    print(f"[saved] {path}", flush=True)
    if failures:
        print(f"!!!! {len(failures)} CONFIG(S) FAILED !!!!")
        for row in failures:
            print("   ", row)
        sys.exit(1)
    print("hpo shard complete")


if __name__ == "__main__":
    main()
