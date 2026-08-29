"""
Phase 1: re-baseline every post-hoc method under the frozen Phase 0 protocol.

What changed under these numbers since the last generation:
  * AvgPop@20 is computed over TEST users, not all users (finding L). This does not move
    Gowalla or Coat (test users are ~all users) but it moves every Yahoo and KuaiRec value.
  * Gini@20 / Coverage@20 / APLT@20 / Tail-Recall@20 are reported alongside AvgPop@20,
    because AvgPop alone is gameable -- a Random ranker wins it on every dataset.
  * Per-user metric vectors are dumped for paired significance testing (significance.py).

This driver evaluates trained checkpoints post-hoc; it does not train. PPD v6 is handled
separately because it needs its own trained weights per seed.

Usage:
    python run_phase1_baselines.py                 # all datasets
    python run_phase1_baselines.py Gowalla KuaiRec # a subset
"""
import shutil
import time
import argparse
import json
import os
import re
import subprocess
import sys

from experiment_configs import DATASET_CONFIGS as EXPERIMENT_CONFIGS

# Interpreter used for child runs. Override with THESIS_PYTHON when the conda env lives
# elsewhere (e.g. after moving to another machine); defaults to whatever is running this.
# Matches utils.checkpoint_tag(): the private default-checkpoint suffix for this worker.
_TAG = ("-" + os.environ["THESIS_CKPT_TAG"]) if os.environ.get("THESIS_CKPT_TAG") else ""

PYTHON = os.environ.get("THESIS_PYTHON", sys.executable)
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CODE_DIR, "results")
PER_USER_DIR = os.path.join(RESULTS_DIR, "per_user")

DATASETS = {
    "Gowalla": {"layer": 3, "recdim": 64,  "beta": 0.20, "seeds": [2024, 2025, 2026]},
    "KuaiRec": {"layer": 2, "recdim": 256, "beta": 2.00, "seeds": [2024, 2025, 2026]},
    "Yahoo":   {"layer": 2, "recdim": 256, "beta": 0.07, "seeds": [2024, 2025, 2026]},
    "Coat":    {"layer": 2, "recdim": 64,  "beta": 0.20, "seeds": [2024, 2025, 2026]},
}

METHODS = {
    "LightGCN": [],
    "TSP":      ["--tsp", "--alpha_tsp", "0.1", "--theta_tsp", "0.90",
                 "--top_m_tsp", "15", "--k_tsp", "1"],
    "PPD v3":   ["--ppd", "--ppd_variant", "3"],
    "PPD v4":   ["--ppd", "--ppd_variant", "4"],
}

# Metrics printed inside the [Eval] block as numpy arrays.
ARRAY_METRICS = ("precision", "recall", "ndcg", "gini", "coverage", "aplt", "tail_recall")


def parse_eval_block(output):
    """
    Parse the [Eval] section only.

    run_thesis.py now prints [Eval] AND [Valid]; a bare re.search would silently pick
    whichever came first if the order ever changed. Slicing to the section first makes
    the intent explicit and the failure loud.
    """
    if "[Eval]" not in output:
        raise ValueError("no [Eval] block in output")
    section = output.split("[Eval]", 1)[1]
    for stop in ("[Valid]", "[Average Popularity]"):
        if stop in section:
            section = section.split(stop, 1)[0]

    metrics = {}
    for name in ARRAY_METRICS:
        m = re.search(rf"'{name}':\s*array\(\[(.*?)\]\)", section, re.S)
        if not m:
            continue
        vals = [float(x) for x in re.split(r"[\s,]+", m.group(1).strip()) if x]
        if vals:
            metrics[f"{name}@20"] = vals[-1] if len(vals) > 1 else vals[0]

    # AvgPop comes from the test block, which precedes [Valid Average Popularity]
    pop_section = output.split("[Average Popularity]", 1)
    if len(pop_section) > 1:
        head = pop_section[1].split("[Valid", 1)[0]
        m = re.search(r"Average Popularity@20:\s*([\d.]+)", head)
        if m:
            metrics["avgpop@20"] = float(m.group(1))
    return metrics


def ensure_baseline_checkpoint(dataset, cfg, seed):
    """Train the seed's LightGCN checkpoint if it does not exist yet.

    All four "baselines" share ONE trained LightGCN: TSP, PPD v3 and PPD v4 are post-hoc
    inference-time transforms of the same weights, not separate training runs. So a new seed
    needs exactly one training run, and then four evaluations.

    Previously this function did not exist and run_one simply failed with "missing checkpoint",
    which is how `--seeds 2027,2028` came to extend PPD v6 to 5 seeds while silently leaving the
    baselines at 3 -- a comparison across different n that the summary tables would have shown
    as if it were like-for-like.
    """
    ckpt_dir = os.path.join(CODE_DIR, "checkpoints")
    seed_ckpt = os.path.join(ckpt_dir,
                             f"lgn-{dataset}-{cfg['layer']}-{cfg['recdim']}-seed{seed}-best.pth.tar")
    if os.path.exists(seed_ckpt):
        return seed_ckpt, None

    ecfg = EXPERIMENT_CONFIGS.get(dataset, {})
    cmd = [PYTHON, "main.py", "--method", "lightgcn", "--dataset", dataset,
           "--layer", str(cfg["layer"]), "--recdim", str(cfg["recdim"]),
           "--epochs", str(ecfg.get("epochs", 100)),
           "--bpr_batch", str(ecfg.get("bpr_batch", 2048)),
           "--lr", str(ecfg.get("lr", 0.001)), "--seed", str(seed)]
    print(f"  [train] LightGCN {dataset} seed {seed} (no checkpoint yet)", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=CODE_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        return None, f"train exit {res.returncode}: {(res.stderr or '').strip().splitlines()[-2:]}"
    default_best = os.path.join(ckpt_dir,
                                f"lgn-{dataset}-{cfg['layer']}-{cfg['recdim']}{_TAG}-best.pth.tar")
    if not os.path.exists(default_best):
        return None, "training produced no checkpoint"
    if os.path.getmtime(default_best) < t0:
        return None, ("stale checkpoint: training saved nothing, refusing to copy an older "
                      "seed's weights under this seed's name")
    shutil.copyfile(default_best, seed_ckpt)
    return seed_ckpt, None


def run_one(dataset, cfg, method, flags, seed):
    ckpt, err = ensure_baseline_checkpoint(dataset, cfg, seed)
    if err:
        return None, err

    tag = method.replace(" ", "").replace("/", "")
    dump = os.path.join(PER_USER_DIR, f"{dataset}_{tag}_seed{seed}.npz")
    cmd = [PYTHON, "run_thesis.py",
           "--dataset", dataset, "--load", "1", "--model", "lgn",
           "--layer", str(cfg["layer"]), "--recdim", str(cfg["recdim"]),
           "--topks", "[20]", "--ckpt", ckpt, "--seed", str(seed),
           "--fairness", "--dump_per_user", dump] + flags
    if "--ppd" in flags:
        cmd += ["--beta", str(cfg["beta"]), "--phi", "1.0"]

    res = subprocess.run(cmd, cwd=CODE_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        return None, f"exit {res.returncode}: {res.stderr.strip().splitlines()[-1:] or ''}"
    try:
        metrics = parse_eval_block(res.stdout)
    except ValueError as exc:
        return None, str(exc)
    required = {"recall@20", "ndcg@20", "precision@20", "avgpop@20"}
    missing = required - set(metrics)
    if missing:
        return None, f"missing metrics {sorted(missing)}"
    return metrics, None


def summarize(seed_metrics):
    keys = sorted({k for m in seed_metrics.values() for k in m})
    mean, std = {}, {}
    for k in keys:
        vals = [m[k] for m in seed_metrics.values() if k in m]
        n = len(vals)
        mu = sum(vals) / n
        mean[k] = mu
        std[k] = (sum((v - mu) ** 2 for v in vals) / n) ** 0.5
    return mean, std


def main():
    ap = argparse.ArgumentParser(description="Phase 1 baselines under the frozen protocol.")
    ap.add_argument("datasets", nargs="*", default=[])
    ap.add_argument("--seeds", default="",
                    help="Comma-separated seeds. Default: each dataset's configured seeds. "
                         "Results MERGE into any existing phase1_<ds>.json, so extra seeds can "
                         "be added later without re-running the ones already done.")
    args = ap.parse_args()

    os.makedirs(PER_USER_DIR, exist_ok=True)
    wanted = args.datasets or list(DATASETS)
    unknown = [d for d in wanted if d not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset(s): {', '.join(unknown)}")
    cli_seeds = [int(x) for x in args.seeds.split(",") if x.strip()] if args.seeds else None

    failures = []
    for dataset in wanted:
        cfg = DATASETS[dataset]
        seeds = cli_seeds if cli_seeds is not None else cfg["seeds"]
        path = os.path.join(RESULTS_DIR, f"phase1_{dataset.lower()}.json")

        # Merge, never clobber. Re-running with --seeds 2027,2028 must extend the existing
        # 3-seed result rather than replace it with a 2-seed one -- silently shrinking n is
        # exactly how a study ends up quoting error bars from fewer runs than it claims.
        out = {}
        if os.path.exists(path):
            with open(path) as f:
                out = json.load(f)

        for method, flags in METHODS.items():
            seed_metrics = dict(out.get(method, {}).get("seeds", {}))
            for seed in seeds:
                print(f"[run] {dataset} | {method} | seed {seed}", flush=True)
                metrics, err = run_one(dataset, cfg, method, flags, seed)
                if err:
                    print(f"  ERROR: {err}", flush=True)
                    failures.append((dataset, method, seed, err))
                    continue
                seed_metrics[str(seed)] = metrics
                print(f"  R@20={metrics['recall@20']:.6f} "
                      f"AvgPop={metrics['avgpop@20']:.2f} "
                      f"Cov={metrics.get('coverage@20', float('nan')):.4f}", flush=True)
            if seed_metrics:
                mean, std = summarize(seed_metrics)
                out[method] = {"seeds": seed_metrics, "mean": mean, "std": std}
                print(f"  -> {method}: n={len(seed_metrics)} seeds "
                      f"({', '.join(sorted(seed_metrics))})", flush=True)

        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {path}\n", flush=True)

    if failures:
        print("=" * 70)
        print(f"!!!! {len(failures)} RUN(S) FAILED - RESULTS ARE INCOMPLETE !!!!")
        for row in failures:
            print("   ", row)
        sys.exit(1)
    print("all runs completed")


if __name__ == "__main__":
    main()
