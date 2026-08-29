"""
Phase 3: does the gate collapse depend on the data, or on the initialization?

Finding Y observed that PPD v6's spectral strength alpha and topological gate theta retain
21% / 57% of their starting values on Gowalla -- the one dataset where v6 wins -- while
collapsing to numerical zero on Yahoo and Coat. The tempting reading is that the mechanisms
are adaptive: they switch on where popularity is exposure-driven and off where it is earned.

That reading is confounded. Gowalla also *starts* alpha four times higher (astd_alpha_base
0.40 vs 0.10), so "Gowalla keeps more alpha" may be nothing more than "Gowalla was given
more alpha to keep".

This script separates the two by swapping the two knobs across datasets and leaving
everything else at each dataset's tuned configuration:

    arm            alpha_base   topo_gate_init   asks
    native         dataset's own values          reference
    gowalla_init   0.40         0.40             does Yahoo/Coat retain gates when given
                                                 Gowalla's larger start?
    yahoo_init     0.10         0.75             does Gowalla still retain gates when given
                                                 Yahoo's start?

Falsifiable, and both outcomes are publishable:

  * If the collapse tracks the DATASET -- Yahoo still collapses from 0.40, Gowalla still
    retains from 0.10 -- then adaptation is real and finding Y's claim 2 stands.
  * If it tracks the INITIALIZATION -- Yahoo retains when started high -- then finding Y's
    claim 2 is withdrawn, and the honest statement is that alpha/theta simply decay toward
    zero at a rate set by where they start, on every dataset.

Gowalla costs ~3.5 h per run, so it is scheduled last and can be skipped with --skip-gowalla
to get the cheap half of the answer first.

Usage:
    python run_phase3_gate_transfer.py                    # all datasets, all arms
    python run_phase3_gate_transfer.py --skip-gowalla     # Yahoo + Coat only (~2 h)
"""
import argparse
import json
import os
import shutil
import subprocess
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_configs import DATASET_CONFIGS
from run_phase1_baselines import parse_eval_block, summarize, PER_USER_DIR, RESULTS_DIR
from run_phase2_ablation import parse_gates, PYTHON, CODE_DIR, CKPT_DIR, _TAG

# The two knobs under test, and the donor configurations they are taken from.
ARMS = {
    "native":       None,                                        # each dataset's own values
    "gowalla_init": {"astd_alpha_base": "0.4",  "astd_topo_gate_init": "0.4"},
    "yahoo_init":   {"astd_alpha_base": "0.1",  "astd_topo_gate_init": "0.75"},
}


def build_params(cfg, overrides):
    overrides = overrides or {}
    out = []
    for key, val in cfg["ppd_v6_params"].items():
        out += [f"--{key}", str(overrides.get(key, val))]
    # A knob the dataset config never set still has to be applied.
    for key, val in overrides.items():
        if key not in cfg["ppd_v6_params"]:
            out += [f"--{key}", str(val)]
    return out


def alias_of(dataset, cfg, arm):
    """Return the earlier arm this one is byte-for-byte identical to, or None.

    The transfer arms are alpha_base / topo_gate_init overrides, so on a dataset whose native
    configuration already equals an arm's override the two are the *same experiment*. That is
    not hypothetical: Gowalla's native config is 0.40/0.40 and `gowalla_init` sets 0.40/0.40,
    while Yahoo's native is 0.10/0.75 and `yahoo_init` sets 0.10/0.75. Yahoo's run did the
    duplicate work silently and cheaply; on Gowalla it would have cost ~21 GPU-hours to
    retrain a configuration already trained.

    Verified empirically on Yahoo before this shortcut was enabled: `native`, `yahoo_init` and
    the Phase 2 `a_full` arm agree to eight decimal places on all six metrics, while
    `gowalla_init` -- a genuinely different configuration -- differs.

    Aliased arms are recorded with an `alias_of` key so they are never mistaken for independent
    replicates when the results are summarised.
    """
    mine = build_params(cfg, ARMS[arm])
    for other in ARMS:
        if other == arm:
            break  # only alias to an arm that runs earlier in ARMS order
        if build_params(cfg, ARMS[other]) == mine:
            return other
    return None


def reusable_checkpoint(dataset, cfg, arm, seed):
    """A checkpoint from another experiment trained under an identical command line, if any.

    Phase 2's `a_full` arm applies no overrides, so it *is* the dataset's native PPD v6
    configuration -- the same thing this script's `native` arm trains. Where the command lines
    match exactly, the existing weights are reused and only evaluation is re-run. On Gowalla
    that avoids three 3.5-hour training runs.

    Returns the source path or None. The caller copies it and evaluates; it does not assume the
    numbers, it recomputes them from the reused weights.
    """
    from run_phase2_ablation import ARMS as AB_ARMS, build_params as ab_build_params
    layer, recdim = cfg["layer"], cfg["recdim"]
    mine = sorted(build_params(cfg, ARMS[arm]))
    for ab_arm in AB_ARMS:
        if sorted(ab_build_params(cfg, AB_ARMS[ab_arm])) != mine:
            continue
        src = os.path.join(CKPT_DIR,
                           f"ppd_v6-{dataset}-{layer}-{recdim}-ablate-{ab_arm}-seed{seed}-best.pth.tar")
        if os.path.exists(src):
            return src
    return None


def run_arm(dataset, cfg, arm, seed):
    layer, recdim = cfg["layer"], cfg["recdim"]
    tag = f"transfer-{arm}"
    seed_ckpt = os.path.join(CKPT_DIR, f"ppd_v6-{dataset}-{layer}-{recdim}-{tag}-seed{seed}-best.pth.tar")
    params = build_params(cfg, ARMS[arm])
    gates = {}

    if not os.path.exists(seed_ckpt):
        src = reusable_checkpoint(dataset, cfg, arm, seed)
        if src:
            shutil.copyfile(src, seed_ckpt)
            src_gates = src.replace("-best.pth.tar", "-gates.json")
            if os.path.exists(src_gates):
                shutil.copyfile(src_gates, seed_ckpt.replace("-best.pth.tar", "-gates.json"))
            print(f"    reusing weights from {os.path.basename(src)} "
                  f"(identical command line) -- evaluating only", flush=True)

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

    dump = os.path.join(PER_USER_DIR, f"{dataset}_{tag}_seed{seed}.npz")
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
    ap.add_argument("--skip-gowalla", action="store_true",
                    help="Gowalla costs ~3.5 h/run; skip it for the cheap half of the answer.")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    wanted = args.datasets or ["Coat", "Yahoo", "Gowalla"]
    if args.skip_gowalla:
        wanted = [d for d in wanted if d != "Gowalla"]

    os.makedirs(PER_USER_DIR, exist_ok=True)
    failures = []
    for dataset in wanted:
        cfg = DATASET_CONFIGS[dataset]
        native_alpha = cfg["ppd_v6_params"].get("astd_alpha_base")
        native_theta = cfg["ppd_v6_params"].get("astd_topo_gate_init")
        print(f"\n===== GATE TRANSFER: {dataset} "
              f"(native alpha_base={native_alpha}, topo_gate_init={native_theta}) =====", flush=True)
        path = os.path.join(RESULTS_DIR, f"phase3_gate_transfer_{dataset.lower()}{args.out_suffix}.json")
        # Merge, never clobber -- same reasoning as the other drivers.
        out = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    out = json.load(f)
            except (ValueError, OSError):
                out = {}
        for arm in ARMS:
            src = alias_of(dataset, cfg, arm)
            if src and src in out:
                # Same command line as `src` -- reuse its result rather than retraining.
                out[arm] = dict(out[src])
                out[arm]["alias_of"] = src
                out[arm]["overrides"] = ARMS[arm]
                print(f"  [{arm}] identical configuration to [{src}] on {dataset} "
                      f"-- reusing its result, no training", flush=True)
                continue
            seed_metrics = {}
            for seed in seeds:
                print(f"  [{arm}] seed {seed} ...", flush=True)
                m, err = run_arm(dataset, cfg, arm, seed)
                if err:
                    print(f"    ERROR: {err}", flush=True)
                    failures.append((dataset, arm, seed, err))
                    continue
                seed_metrics[str(seed)] = m
                print(f"    R@20={m['recall@20']:.6f} "
                      f"alpha={m.get('mean_alpha_final', float('nan')):.4f} "
                      f"theta={m.get('mean_theta_final', float('nan')):.4f} "
                      f"lambda={m.get('mean_lambda_final', float('nan')):.4f}", flush=True)
            if seed_metrics:
                mean, std = summarize(seed_metrics)
                out[arm] = {"seeds": seed_metrics, "mean": mean, "std": std,
                            "overrides": ARMS[arm]}
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {path} (arms: {', '.join(sorted(out))})", flush=True)

    if failures:
        print("=" * 70)
        print(f"!!!! {len(failures)} RUN(S) FAILED - TRANSFER IS INCOMPLETE !!!!")
        for row in failures:
            print("   ", row)
        sys.exit(1)
    print("\ngate transfer complete")


if __name__ == "__main__":
    main()
