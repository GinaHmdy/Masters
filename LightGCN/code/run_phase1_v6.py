"""
Phase 1: train PPD v6 / ASTD-PD per seed and evaluate under the frozen Phase 0 protocol.

The baseline methods are post-hoc transforms of a trained LightGCN, so run_phase1_baselines.py
only has to evaluate existing checkpoints. PPD v6 has its own weights and must be trained per
seed before it can be compared, which is why it lives in a separate driver.

Hyperparameters are taken verbatim from run_all_remaining_datasets.py so this is a re-run of the
SAME configuration under the corrected protocol, not a re-tune. Re-selecting them honestly on
validation is Phase 3 -- keeping them fixed here is what makes the before/after comparable.

Usage:
    python run_phase1_v6.py                  # all datasets, all seeds
    python run_phase1_v6.py Coat --seeds 2024,2025,2026
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_configs import DATASET_CONFIGS  # single source of hyperparameters
from run_phase1_baselines import parse_eval_block, summarize, PER_USER_DIR, RESULTS_DIR

# Interpreter used for child runs. Override with THESIS_PYTHON when the conda env lives
# elsewhere (e.g. after moving to another machine); defaults to whatever is running this.
# Matches utils.checkpoint_tag(): the private default-checkpoint suffix for this worker.
_TAG = ("-" + os.environ["THESIS_CKPT_TAG"]) if os.environ.get("THESIS_CKPT_TAG") else ""

PYTHON = os.environ.get("THESIS_PYTHON", sys.executable)
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(CODE_DIR, "checkpoints")


def train_seed(dataset, cfg, seed):
    """Train one v6 seed. Returns the seed checkpoint path, or None on failure."""
    layer, recdim = cfg["layer"], cfg["recdim"]
    seed_ckpt = os.path.join(CKPT_DIR, f"ppd_v6-{dataset}-{layer}-{recdim}-seed{seed}-best.pth.tar")
    if os.path.exists(seed_ckpt):
        print(f"  [skip training] {os.path.basename(seed_ckpt)} exists")
        return seed_ckpt

    p = cfg["ppd_v6_params"]
    cmd = [PYTHON, "main.py", "--method", "ppd_v6", "--dataset", dataset,
           "--layer", str(layer), "--recdim", str(recdim),
           "--epochs", str(cfg["epochs"]), "--bpr_batch", str(cfg["bpr_batch"]),
           "--lr", str(cfg["lr"]), "--seed", str(seed)]
    for key, val in p.items():
        cmd += [f"--{key}", str(val)]

    print(f"  [train] seed {seed} ...", flush=True)
    res = subprocess.run(cmd, cwd=CODE_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        tail = (res.stderr or "").strip().splitlines()[-3:]
        print(f"  ERROR training exit {res.returncode}: {tail}")
        return None

    default_best = os.path.join(CKPT_DIR, f"ppd_v6-{dataset}-{layer}-{recdim}{_TAG}-best.pth.tar")
    if not os.path.exists(default_best):
        print(f"  ERROR: training produced no checkpoint at {default_best}")
        return None
    shutil.copyfile(default_best, seed_ckpt)
    return seed_ckpt


def evaluate_seed(dataset, cfg, seed, ckpt):
    layer, recdim = cfg["layer"], cfg["recdim"]
    dump = os.path.join(PER_USER_DIR, f"{dataset}_PPDv6_seed{seed}.npz")
    p = cfg["ppd_v6_params"]
    cmd = [PYTHON, "run_thesis.py", "--dataset", dataset, "--load", "1",
           "--model", "ppd_v6", "--method", "ppd_v6",
           "--layer", str(layer), "--recdim", str(recdim), "--topks", "[20]",
           "--ckpt", ckpt, "--seed", str(seed), "--fairness", "--dump_per_user", dump]
    for key, val in p.items():
        cmd += [f"--{key}", str(val)]

    res = subprocess.run(cmd, cwd=CODE_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        return None, f"eval exit {res.returncode}: {(res.stderr or '').strip().splitlines()[-2:]}"
    try:
        metrics = parse_eval_block(res.stdout)
    except ValueError as exc:
        return None, str(exc)
    missing = {"recall@20", "ndcg@20", "precision@20", "avgpop@20"} - set(metrics)
    if missing:
        return None, f"missing metrics {sorted(missing)}"
    return metrics, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=[])
    ap.add_argument("--seeds", default="2024,2025,2026")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    wanted = args.datasets or list(DATASET_CONFIGS)
    unknown = [d for d in wanted if d not in DATASET_CONFIGS]
    if unknown:
        raise SystemExit(f"unknown dataset(s): {', '.join(unknown)}")

    os.makedirs(PER_USER_DIR, exist_ok=True)
    failures = []
    for dataset in wanted:
        cfg = DATASET_CONFIGS[dataset]
        print(f"\n===== PPD v6: {dataset} =====", flush=True)
        path = os.path.join(RESULTS_DIR, f"phase1_{dataset.lower()}_v6.json")
        # Merge, never clobber -- see the same note in run_phase1_baselines.py. Adding
        # seeds 2027/2028 later must extend the existing result, not shrink n back to 2.
        seed_metrics = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    seed_metrics = dict(json.load(f).get("PPD v6", {}).get("seeds", {}))
            except (ValueError, OSError):
                seed_metrics = {}
        for seed in seeds:
            ckpt = train_seed(dataset, cfg, seed)
            if ckpt is None:
                failures.append((dataset, seed, "training failed"))
                continue
            metrics, err = evaluate_seed(dataset, cfg, seed, ckpt)
            if err:
                print(f"  ERROR: {err}", flush=True)
                failures.append((dataset, seed, err))
                continue
            seed_metrics[str(seed)] = metrics
            print(f"  seed {seed}: R@20={metrics['recall@20']:.6f} "
                  f"AvgPop={metrics['avgpop@20']:.2f} "
                  f"Cov={metrics.get('coverage@20', float('nan')):.4f}", flush=True)

        if seed_metrics:
            mean, std = summarize(seed_metrics)
            out = {"PPD v6": {"seeds": seed_metrics, "mean": mean, "std": std,
                              "params": cfg["ppd_v6_params"]}}
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[saved] {path} (n={len(seed_metrics)} seeds: "
                  f"{', '.join(sorted(seed_metrics))})", flush=True)

    if failures:
        print("=" * 70)
        print(f"!!!! {len(failures)} RUN(S) FAILED - RESULTS ARE INCOMPLETE !!!!")
        for row in failures:
            print("   ", row)
        sys.exit(1)
    print("\nall v6 runs completed")


if __name__ == "__main__":
    main()
