"""
Phase 5b: re-run the validation-selected configuration at 5 seeds and report it.

`run_phase5_hpo.py` picks a configuration per dataset using validation only. A winner chosen on
one seed is not a result, so this retrains it across the full seed set and evaluates under the
same frozen protocol as Phase 1, writing to `phase5_final_<dataset>.json`.

It deliberately does NOT edit `experiment_configs.py`. The shipped configuration stays where it
is so that every earlier result keeps meaning what it meant, and the validation-selected numbers
sit beside it as a separate, comparable row. Whether to promote the winner is a decision for the
write-up, made with both columns visible.

Per-user dumps are written as `<Dataset>_PPDv6val_seed<seed>.npz` so `significance.py` can test
the re-selected model against LightGCN exactly as it tests the shipped one.

Usage:
    python run_phase5_final.py Coat Yahoo --seeds 2024,2025,2026,2027,2028
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_configs import DATASET_CONFIGS
from run_phase1_baselines import parse_eval_block, summarize, PER_USER_DIR, RESULTS_DIR
from run_phase2_ablation import PYTHON, CODE_DIR, CKPT_DIR, _TAG


def winner(dataset):
    """Highest validation hypervolume in the completed sweep. Ties broken deterministically."""
    path = os.path.join(RESULTS_DIR, f"phase5_hpo_{dataset.lower()}.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        d = json.load(f)
    if not d:
        return None, None
    best = max(d.values(), key=lambda v: (v["valid_hypervolume"],
                                          json.dumps(v["params"], sort_keys=True)))
    return best["params"], best["valid_hypervolume"]


def params_for(cfg, overrides):
    p = dict(cfg["ppd_v6_params"])
    p.update(overrides)
    out = []
    for k, v in p.items():
        out += [f"--{k}", str(v)]
    return out


def run_seed(dataset, cfg, overrides, seed):
    layer, recdim = cfg["layer"], cfg["recdim"]
    ckpt = os.path.join(CKPT_DIR, f"ppd_v6-{dataset}-{layer}-{recdim}-valsel-seed{seed}-best.pth.tar")
    params = params_for(cfg, overrides)

    if not os.path.exists(ckpt):
        cmd = [PYTHON, "main.py", "--method", "ppd_v6", "--dataset", dataset,
               "--layer", str(layer), "--recdim", str(recdim),
               "--epochs", str(cfg["epochs"]), "--bpr_batch", str(cfg["bpr_batch"]),
               "--lr", str(cfg["lr"]), "--seed", str(seed)] + params
        import time
        t0 = time.time()
        res = subprocess.run(cmd, cwd=CODE_DIR, capture_output=True, text=True)
        if res.returncode != 0:
            return None, f"train exit {res.returncode}: {(res.stderr or '').strip().splitlines()[-2:]}"
        default_best = os.path.join(CKPT_DIR, f"ppd_v6-{dataset}-{layer}-{recdim}{_TAG}-best.pth.tar")
        if not os.path.exists(default_best):
            return None, "training produced no checkpoint"
        if os.path.getmtime(default_best) < t0:
            return None, "stale checkpoint: training saved nothing this run"
        shutil.copyfile(default_best, ckpt)

    dump = os.path.join(PER_USER_DIR, f"{dataset}_PPDv6val_seed{seed}.npz")
    cmd = [PYTHON, "run_thesis.py", "--dataset", dataset, "--load", "1",
           "--model", "ppd_v6", "--method", "ppd_v6",
           "--layer", str(layer), "--recdim", str(recdim), "--topks", "[20]",
           "--ckpt", ckpt, "--seed", str(seed), "--fairness",
           "--dump_per_user", dump] + params
    res = subprocess.run(cmd, cwd=CODE_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        return None, f"eval exit {res.returncode}: {(res.stderr or '').strip().splitlines()[-2:]}"
    try:
        return parse_eval_block(res.stdout), None
    except ValueError as exc:
        return None, str(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=[])
    ap.add_argument("--seeds", default="2024,2025,2026,2027,2028")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    os.makedirs(PER_USER_DIR, exist_ok=True)
    failures = []
    for dataset in (args.datasets or list(DATASET_CONFIGS)):
        overrides, hv = winner(dataset)
        if overrides is None:
            print(f"[skip] {dataset}: no completed sweep", flush=True)
            continue
        cfg = DATASET_CONFIGS[dataset]
        shipped = {k: cfg["ppd_v6_params"].get(k) for k in overrides}
        print(f"\n===== {dataset} validation-selected config =====", flush=True)
        print(f"  selected: {overrides}  (valid HV {hv:.8f})", flush=True)
        print(f"  shipped : {shipped}", flush=True)

        path = os.path.join(RESULTS_DIR, f"phase5_final_{dataset.lower()}{args.out_suffix}.json")
        per_seed = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    per_seed = json.load(f).get("seeds", {})
            except (ValueError, OSError):
                per_seed = {}
        for seed in seeds:
            if str(seed) in per_seed:
                continue
            m, err = run_seed(dataset, cfg, overrides, seed)
            if err:
                print(f"  seed {seed} ERROR: {err}", flush=True)
                failures.append((dataset, seed, err))
                continue
            per_seed[str(seed)] = m
            print(f"  seed {seed}: R@20={m['recall@20']:.6f} AvgPop={m['avgpop@20']:.2f} "
                  f"APLT={m.get('aplt@20', float('nan')):.4f}", flush=True)
            mean, std = summarize(per_seed)
            with open(path, "w") as f:
                json.dump({"selected_params": overrides, "shipped_params": shipped,
                           "valid_hypervolume": hv, "seeds": per_seed,
                           "mean": mean, "std": std}, f, indent=2)
        print(f"[saved] {path}", flush=True)

    if failures:
        print("=" * 70)
        print(f"!!!! {len(failures)} RUN(S) FAILED !!!!")
        for row in failures:
            print("   ", row)
        sys.exit(1)
    print("\nphase 5b complete")


if __name__ == "__main__":
    main()
