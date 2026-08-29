"""
Phase 4: close Gowalla's AvgPop gap with a validation-selected inference penalty.

Remediation item 5.4 -- "the one run that could convert a 3/4 near-miss into the stated goal".
Under the frozen protocol PPD v6 beats every baseline on Gowalla's three accuracy metrics but
loses AvgPop@20 to PPD v4 (141.60 +/- 2.00 vs 138.44 +/- 2.79), so the four-metric claim fails
on one axis by ~3 points.

`--astd_inference_pop_penalty mu` subtracts mu * p_i from the fused score at ranking time
(model.py:1035). It touches no weights, so this sweep re-ranks the five ALREADY TRAINED v6
checkpoints rather than retraining anything -- minutes, not GPU-days.

Two things make it an honest experiment rather than a test-set fit:

  1. **mu is selected on VALIDATION**, by the same 4-D Pareto hypervolume that main.py uses for
     checkpoint selection (utilities.compute_pareto_hypervolume). The test block is parsed only
     to report the selected configuration, never to choose it. run_thesis.py's `[Valid]` block
     exists precisely so drivers can select this way (see its comment at run_thesis.py:336).
  2. It is **methodologically symmetric** with the baselines it is compared against: PPD v3/v4
     and TSP are themselves post-hoc inference-time transforms of trained LightGCN weights. A
     post-hoc penalty on v6 is the same class of intervention, not an extra advantage.

Usage:
    python run_phase4_inference_penalty.py                    # full sweep, 5 seeds
    python run_phase4_inference_penalty.py --mus 0.0,0.5 --seeds 2024
"""
import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_configs import DATASET_CONFIGS
from run_phase1_baselines import parse_eval_block, summarize, PER_USER_DIR, RESULTS_DIR
from run_phase2_ablation import PYTHON, CODE_DIR, CKPT_DIR

DATASET = "Gowalla"
# Score-space grid. p_i is a popularity rank in [0, 1], so mu is measured in score units;
# 0.0 is the trained default and the upper end is deliberately past the point of usefulness
# so the validation curve is seen to turn over rather than being cut off at its edge.
DEFAULT_MUS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
ARRAY_METRICS = ("precision", "recall", "ndcg", "gini", "coverage", "aplt", "tail_recall")


def parse_valid_block(output):
    """Metrics from the [Valid] block -- the ONLY block selection may look at."""
    if "[Valid]" not in output:
        raise ValueError("no [Valid] block in output")
    section = output.split("[Valid]", 1)[1].split("[Valid Average Popularity]", 1)[0]
    metrics = {}
    for name in ARRAY_METRICS:
        m = re.search(rf"'{name}':\s*array\(\[(.*?)\]\)", section, re.S)
        if not m:
            continue
        vals = [float(x) for x in re.split(r"[\s,]+", m.group(1).strip()) if x]
        if vals:
            metrics[f"{name}@20"] = vals[-1] if len(vals) > 1 else vals[0]

    if "[Valid Average Popularity]" in output:
        head = output.split("[Valid Average Popularity]", 1)[1].split("[Average Popularity]", 1)[0]
        m = re.search(r"Average Popularity@20:\s*([\d.]+)", head)
        if m:
            metrics["avgpop@20"] = float(m.group(1))
    return metrics


def hypervolume(metrics, max_pop_ref=1000.0):
    """4-D Pareto hypervolume, identical to utilities.compute_pareto_hypervolume.

    Reimplemented rather than imported because utilities.py imports `world` and `register`,
    which parse sys.argv and instantiate a dataset at import time -- that would hijack this
    driver's own arguments. Keep the two in step: recall * ndcg * precision * inverted-AvgPop.
    """
    inv_pop = max(0.0, 1.0 / (1.0 + metrics.get("avgpop@20", 0.0) / max_pop_ref))
    return (max(0.0, metrics.get("recall@20", 0.0))
            * max(0.0, metrics.get("ndcg@20", 0.0))
            * max(0.0, metrics.get("precision@20", 0.0))
            * inv_pop)


def build_params(cfg, mu):
    out = []
    for key, val in cfg["ppd_v6_params"].items():
        out += [f"--{key}", str(val) if key != "astd_inference_pop_penalty" else str(mu)]
    if "astd_inference_pop_penalty" not in cfg["ppd_v6_params"]:
        out += ["--astd_inference_pop_penalty", str(mu)]
    return out


def evaluate(cfg, mu, seed, dump=None):
    layer, recdim = cfg["layer"], cfg["recdim"]
    ckpt = os.path.join(CKPT_DIR, f"ppd_v6-{DATASET}-{layer}-{recdim}-seed{seed}-best.pth.tar")
    if not os.path.exists(ckpt):
        return None, f"missing checkpoint {os.path.basename(ckpt)}"
    cmd = [PYTHON, "run_thesis.py", "--dataset", DATASET, "--load", "1",
           "--model", "ppd_v6", "--method", "ppd_v6",
           "--layer", str(layer), "--recdim", str(recdim), "--topks", "[20]",
           "--ckpt", ckpt, "--seed", str(seed), "--fairness"] + build_params(cfg, mu)
    if dump:
        cmd += ["--dump_per_user", dump]
    res = subprocess.run(cmd, cwd=CODE_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        return None, f"eval exit {res.returncode}: {(res.stderr or '').strip().splitlines()[-2:]}"
    try:
        return {"test": parse_eval_block(res.stdout),
                "valid": parse_valid_block(res.stdout)}, None
    except ValueError as exc:
        return None, str(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mus", default=",".join(str(m) for m in DEFAULT_MUS))
    ap.add_argument("--seeds", default="2024,2025,2026,2027,2028")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--dump-per-user", action="store_true",
                    help="Also write per-user dumps (only needed for the selected mu).")
    args = ap.parse_args()

    mus = [float(x) for x in args.mus.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    cfg = DATASET_CONFIGS[DATASET]
    os.makedirs(PER_USER_DIR, exist_ok=True)

    path = os.path.join(RESULTS_DIR, f"phase4_inference_penalty_gowalla{args.out_suffix}.json")
    out = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                out = json.load(f)
        except (ValueError, OSError):
            out = {}

    failures = []
    for mu in mus:
        key = f"mu={mu}"
        print(f"\n===== {key} =====", flush=True)
        per_seed = {}
        for seed in seeds:
            dump = (os.path.join(PER_USER_DIR, f"{DATASET}_PPDv6mu{mu}_seed{seed}.npz")
                    if args.dump_per_user else None)
            m, err = evaluate(cfg, mu, seed, dump)
            if err:
                print(f"  seed {seed} ERROR: {err}", flush=True)
                failures.append((mu, seed, err))
                continue
            m["valid"]["hypervolume"] = hypervolume(m["valid"])
            per_seed[str(seed)] = m
            print(f"  seed {seed}: valid R@20={m['valid']['recall@20']:.6f} "
                  f"AvgPop={m['valid']['avgpop@20']:.2f} HV={m['valid']['hypervolume']:.8f} "
                  f"|| test R@20={m['test']['recall@20']:.6f} "
                  f"AvgPop={m['test']['avgpop@20']:.2f}", flush=True)
        if per_seed:
            test_mean, test_std = summarize({s: v["test"] for s, v in per_seed.items()})
            val_mean, val_std = summarize({s: v["valid"] for s, v in per_seed.items()})
            out[key] = {"mu": mu, "seeds": per_seed,
                        "mean": test_mean, "std": test_std,
                        "valid_mean": val_mean, "valid_std": val_std}
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {path}", flush=True)

    if failures:
        print("=" * 70)
        print(f"!!!! {len(failures)} EVALUATION(S) FAILED !!!!")
        for row in failures:
            print("   ", row)
        sys.exit(1)
    print("\nsweep complete")


if __name__ == "__main__":
    main()
