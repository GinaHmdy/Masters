"""
Phase 2: mechanism-isolation ablation for PPD v6 / ASTD-PD.

The audit's finding D is that all three headline mechanisms are inert -- across all 108 Yahoo
sweep configurations theta collapsed to 0.0007, alpha to 0.0003, and the "personalized" fusion
gate lambda never moved off its 0.4998 initialization. If that holds under ablation, the honest
claim is that the tail reweighting does the work and the spectral/topological machinery does
not, which is a cleaner contribution than the one currently written down -- but only if measured.

Arms:
  (a) full        - defaults
  (b) no_spectral - alpha_base=0, alpha_min=0     -> removes the spectral popularity projection
  (c) no_simplex  - topo gate fixed at 0          -> removes the popularity-damped simplex stream
  (d) no_tail     - astd_tail_reweight=0          -> removes BOTH tail-reweighting sites
  (e) recipe_only - (b)+(c)+(d)                   -> sampled-softmax + logQ + stratified negatives
                                                     with the three explicit mechanisms off
  (f) no_sampler  - (e) + uniform negatives        -> isolates the popularity-stratified sampler,
                                                     which arm (e) revealed is itself a debiaser

Arm (e) is the one that answers finding E: v6 changes the loss, the sampler AND the parameter
count alongside the architecture, so today no gain is attributable to the debiasing itself.

Note --astd_tail_phi 0 does NOT disable tail reweighting: (1-p)^0 = 1, which makes the term
uniform rather than absent. That is why arm (d) needs its own flag.

Usage:
    python run_phase2_ablation.py                    # all datasets, all arms
    python run_phase2_ablation.py Coat --seeds 2024
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_configs import DATASET_CONFIGS
from run_phase1_baselines import parse_eval_block, summarize, PER_USER_DIR, RESULTS_DIR

# Interpreter used for child runs. Override with THESIS_PYTHON when the conda env lives
# elsewhere (e.g. after moving to another machine); defaults to whatever is running this.
# Matches utils.checkpoint_tag(): the private default-checkpoint suffix for this worker.
_TAG = ("-" + os.environ["THESIS_CKPT_TAG"]) if os.environ.get("THESIS_CKPT_TAG") else ""

PYTHON = os.environ.get("THESIS_PYTHON", sys.executable)
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(CODE_DIR, "checkpoints")

ARMS = {
    "a_full":        [],
    "b_no_spectral": ["--astd_alpha_base", "0.0", "--astd_alpha_min", "0.0"],
    "c_no_simplex":  ["--astd_topo_gate_learnable", "0", "--astd_topo_gate_init", "0.0"],
    "d_no_tail":     ["--astd_tail_reweight", "0"],
    "e_recipe_only": ["--astd_alpha_base", "0.0", "--astd_alpha_min", "0.0",
                      "--astd_topo_gate_learnable", "0", "--astd_topo_gate_init", "0.0",
                      "--astd_tail_reweight", "0"],
    # Arm (f) exists because arm (e) turned out NOT to be a clean no-debiasing control:
    # it still trains through StratifiedPopularityNegativeSampler, which draws negatives by
    # popularity decile and is therefore a debiaser in its own right. On Yahoo, arm (e) alone
    # reproduced ~96% of full v6's AvgPop reduction with all three explicit mechanisms off.
    # (f) additionally forces uniform negative sampling, so the sampler's contribution can be
    # separated from the rest of the recipe. Without it, no claim of the form "mechanism X
    # accounts for the popularity reduction" is safe. See THESIS_AUDIT.md finding AA.
    "f_no_sampler":  ["--astd_alpha_base", "0.0", "--astd_alpha_min", "0.0",
                      "--astd_topo_gate_learnable", "0", "--astd_topo_gate_init", "0.0",
                      "--astd_tail_reweight", "0",
                      "--astd_neg_uniform_ratio", "1.0",
                      "--astd_neg_matched_ratio", "0.0",
                      "--astd_neg_hard_ratio", "0.0"],
}

# Arm flags must win over the dataset defaults, so record which keys they override.
def build_params(cfg, arm_flags):
    overridden = {arm_flags[i].lstrip("-") for i in range(0, len(arm_flags), 2)}
    out = []
    for key, val in cfg["ppd_v6_params"].items():
        if key in overridden:
            continue
        out += [f"--{key}", str(val)]
    return out + arm_flags


def parse_gates(output):
    """Last-epoch alpha / theta / lambda, the diagnostics finding D rests on."""
    hits = re.findall(r"Alpha=([\d.]+), Theta=([\d.]+), MeanLambda=([\d.]+)", output)
    if not hits:
        return {}
    a, t, l = hits[-1]
    return {"mean_alpha_final": float(a), "mean_theta_final": float(t),
            "mean_lambda_final": float(l)}


def run_arm(dataset, cfg, arm, arm_flags, seed):
    layer, recdim = cfg["layer"], cfg["recdim"]
    tag = f"ablate-{arm}"
    seed_ckpt = os.path.join(CKPT_DIR, f"ppd_v6-{dataset}-{layer}-{recdim}-{tag}-seed{seed}-best.pth.tar")
    params = build_params(cfg, arm_flags)

    gates = {}
    if not os.path.exists(seed_ckpt):
        cmd = [PYTHON, "main.py", "--method", "ppd_v6", "--dataset", dataset,
               "--layer", str(layer), "--recdim", str(recdim),
               "--epochs", str(cfg["epochs"]), "--bpr_batch", str(cfg["bpr_batch"]),
               "--lr", str(cfg["lr"]), "--seed", str(seed)] + params
        _t0 = time.time()
        res = subprocess.run(cmd, cwd=CODE_DIR, capture_output=True, text=True)
        if res.returncode != 0:
            return None, f"train exit {res.returncode}: {(res.stderr or '').strip().splitlines()[-2:]}"
        gates = parse_gates(res.stdout)
        default_best = os.path.join(CKPT_DIR, f"ppd_v6-{dataset}-{layer}-{recdim}{_TAG}-best.pth.tar")
        # Every arm/seed writes to the SAME default checkpoint name, so "the file exists"
        # is not evidence that THIS run produced it. If training exits 0 without saving
        # (early-stop before the first checkpoint, a silent CUDA fault), the previous
        # arm's weights would be copied under this arm's name and silently evaluated as
        # this arm's result. Require the mtime to have advanced past the run's start.
        if not os.path.exists(default_best):
            return None, "training produced no checkpoint"
        if os.path.getmtime(default_best) < _t0:
            return None, (f"stale checkpoint: {os.path.basename(default_best)} predates "
                          f"this run, so training saved nothing -- refusing to evaluate "
                          f"another arm's weights as this one")
        shutil.copyfile(default_best, seed_ckpt)
        with open(seed_ckpt.replace("-best.pth.tar", "-gates.json"), "w") as f:
            json.dump(gates, f)
    else:
        gpath = seed_ckpt.replace("-best.pth.tar", "-gates.json")
        if os.path.exists(gpath):
            with open(gpath) as f:
                gates = json.load(f)

    dump = os.path.join(PER_USER_DIR, f"{dataset}_ablate-{arm}_seed{seed}.npz")
    cmd = [PYTHON, "run_thesis.py", "--dataset", dataset, "--load", "1",
           "--model", "ppd_v6", "--method", "ppd_v6",
           "--layer", str(layer), "--recdim", str(recdim), "--topks", "[20]",
           "--ckpt", seed_ckpt, "--seed", str(seed), "--fairness",
           "--dump_per_user", dump] + params
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
    metrics.update(gates)
    return metrics, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=[])
    ap.add_argument("--seeds", default="2024,2025,2026")
    ap.add_argument("--out-suffix", default="",
                    help="Write results to <name><suffix>.json. Parallel workers use this so "
                         "no two processes read-modify-write the same file; merge_shards.py "
                         "recombines them.")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    arms = [a for a in args.arms.split(",") if a.strip()]
    unknown_arms = [a for a in arms if a not in ARMS]
    if unknown_arms:
        raise SystemExit(f"unknown arm(s): {', '.join(unknown_arms)}")
    wanted = args.datasets or list(DATASET_CONFIGS)

    os.makedirs(PER_USER_DIR, exist_ok=True)
    failures = []
    for dataset in wanted:
        cfg = DATASET_CONFIGS[dataset]
        print(f"\n===== ABLATION: {dataset} =====", flush=True)
        path = os.path.join(RESULTS_DIR, f"phase2_ablation_{dataset.lower()}{args.out_suffix}.json")
        # Merge, never clobber. Running `--arms f_no_sampler` to add a late arm must extend the
        # existing file, not replace five arms with one.
        out = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    out = json.load(f)
            except (ValueError, OSError):
                out = {}
        for arm in arms:
            seed_metrics = {}
            for seed in seeds:
                print(f"  [{arm}] seed {seed} ...", flush=True)
                m, err = run_arm(dataset, cfg, arm, ARMS[arm], seed)
                if err:
                    print(f"    ERROR: {err}", flush=True)
                    failures.append((dataset, arm, seed, err))
                    continue
                seed_metrics[str(seed)] = m
                print(f"    R@20={m['recall@20']:.6f} AvgPop={m['avgpop@20']:.2f} "
                      f"APLT={m.get('aplt@20', float('nan')):.4f} "
                      f"alpha={m.get('mean_alpha_final', float('nan')):.4f} "
                      f"theta={m.get('mean_theta_final', float('nan')):.4f} "
                      f"lambda={m.get('mean_lambda_final', float('nan')):.4f}", flush=True)
            if seed_metrics:
                mean, std = summarize(seed_metrics)
                out[arm] = {"seeds": seed_metrics, "mean": mean, "std": std,
                            "flags": ARMS[arm]}
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {path} (arms: {', '.join(sorted(out))})", flush=True)

    if failures:
        print("=" * 70)
        print(f"!!!! {len(failures)} RUN(S) FAILED - ABLATION IS INCOMPLETE !!!!")
        for row in failures:
            print("   ", row)
        sys.exit(1)
    print("\nablation complete")


if __name__ == "__main__":
    main()
