#!/usr/bin/env python
"""
benchmark_yahoo_baselines.py
Step 1: Runs full training and evaluation for LightGCN, TSP, PPD v3, and PPD v4
on the Yahoo dataset across seeds 2024, 2025, and 2026.
Saves final test metrics to results/baselines_yahoo.json.
"""

import os
import sys
import json
import shutil
import subprocess
import re
import numpy as np

code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

import utilities

python_path = sys.executable

SEEDS = [2024, 2025, 2026]
DATASET = "Yahoo"
LAYER = 2
RECDIM = 256
EPOCHS = 100
BPR_BATCH = 2048
LR = 0.001
BETA_YAHOO = 0.07

RESULTS_DIR = os.path.join(code_dir, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CHECKPOINTS_DIR = os.path.join(code_dir, "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)


def parse_metrics_from_output(stdout_text):
    prec_20, rec_20, ndcg_20, pop_20 = 0.0, 0.0, 0.0, 0.0

    precision_match = re.search(r"'precision':\s*array\(\[(.*?)\]\)", stdout_text)
    recall_match = re.search(r"'recall':\s*array\(\[(.*?)\]\)", stdout_text)
    ndcg_match = re.search(r"'ndcg':\s*array\(\[(.*?)\]\)", stdout_text)
    pop_match = re.search(r"(?:Final Test\s+)?Average Popularity@20:\s*([\d\.]+)", stdout_text)

    if precision_match:
        vals = [float(x) for x in re.split(r'\s+|,', precision_match.group(1).replace('\n', ' ').strip()) if x]
        prec_20 = vals[3] if len(vals) > 3 else vals[-1]
    if recall_match:
        vals = [float(x) for x in re.split(r'\s+|,', recall_match.group(1).replace('\n', ' ').strip()) if x]
        rec_20 = vals[3] if len(vals) > 3 else vals[-1]
    if ndcg_match:
        vals = [float(x) for x in re.split(r'\s+|,', ndcg_match.group(1).replace('\n', ' ').strip()) if x]
        ndcg_20 = vals[3] if len(vals) > 3 else vals[-1]
    if pop_match:
        pop_20 = float(pop_match.group(1))

    return {
        "recall@20": float(rec_20),
        "ndcg@20": float(ndcg_20),
        "precision@20": float(prec_20),
        "avgpop@20": float(pop_20)
    }


def train_and_eval_lightgcn(seed):
    seed_best_ckpt = os.path.join(CHECKPOINTS_DIR, f"lgn-{DATASET}-{LAYER}-{RECDIM}-seed{seed}-best.pth.tar")
    default_best_ckpt = os.path.join(CHECKPOINTS_DIR, f"lgn-{DATASET}-{LAYER}-{RECDIM}-best.pth.tar")

    if not os.path.exists(seed_best_ckpt) and os.path.exists(default_best_ckpt):
        shutil.copyfile(default_best_ckpt, seed_best_ckpt)

    if not os.path.exists(seed_best_ckpt):
        print(f"\n==================== [LightGCN] Training Seed {seed} ====================")
        cmd = [
            python_path, "main.py",
            "--method", "lightgcn",
            "--dataset", DATASET,
            "--layer", str(LAYER),
            "--recdim", str(RECDIM),
            "--epochs", str(EPOCHS),
            "--bpr_batch", str(BPR_BATCH),
            "--lr", str(LR),
            "--seed", str(seed)
        ]
        res = subprocess.run(cmd, cwd=code_dir, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print("[STDERR]", res.stderr)

        metrics = parse_metrics_from_output(res.stdout)
        if os.path.exists(default_best_ckpt):
            shutil.copyfile(default_best_ckpt, seed_best_ckpt)
    else:
        print(f"\n[LightGCN] Found existing checkpoint for seed {seed}: {seed_best_ckpt}")
        # Evaluate checkpoint directly
        metrics = eval_run_thesis_baseline("LightGCN", seed, seed_best_ckpt, [])

    print(f"[LightGCN Seed {seed}] Metrics: {metrics}")
    return metrics, seed_best_ckpt


def eval_run_thesis_baseline(method_name, seed, ckpt_path, extra_flags):
    print(f"\n==================== [{method_name}] Evaluating Seed {seed} ====================")
    cmd = [
        python_path, "run_thesis.py",
        "--dataset", DATASET,
        "--load", "1",
        "--model", "lgn",
        "--layer", str(LAYER),
        "--recdim", str(RECDIM),
        "--seed", str(seed),
        "--ckpt", ckpt_path,
        "--fairness"
    ] + extra_flags

    res = subprocess.run(cmd, cwd=code_dir, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("[STDERR]", res.stderr)

    metrics = parse_metrics_from_output(res.stdout)
    print(f"[{method_name} Seed {seed}] Metrics: {metrics}")
    return metrics


def main():
    all_results = {
        "LightGCN": {"seeds": {}, "mean": {}, "std": {}},
        "TSP": {"seeds": {}, "mean": {}, "std": {}},
        "PPD v3": {"seeds": {}, "mean": {}, "std": {}},
        "PPD v4": {"seeds": {}, "mean": {}, "std": {}},
    }

    for seed in SEEDS:
        # 1. Train LightGCN
        lgn_metrics, ckpt_path = train_and_eval_lightgcn(seed)
        all_results["LightGCN"]["seeds"][str(seed)] = lgn_metrics

        # 2. TSP
        tsp_flags = ["--tsp", "--alpha_tsp", "0.1", "--theta_tsp", "0.90", "--top_m_tsp", "15", "--k_tsp", "1"]
        tsp_metrics = eval_run_thesis_baseline("TSP", seed, ckpt_path, tsp_flags)
        all_results["TSP"]["seeds"][str(seed)] = tsp_metrics

        # 3. PPD v3
        ppd3_flags = ["--ppd", "--ppd_variant", "3", "--beta", str(BETA_YAHOO), "--phi", "1.0"]
        ppd3_metrics = eval_run_thesis_baseline("PPD v3", seed, ckpt_path, ppd3_flags)
        all_results["PPD v3"]["seeds"][str(seed)] = ppd3_metrics

        # 4. PPD v4
        ppd4_flags = ["--ppd", "--ppd_variant", "4", "--beta", str(BETA_YAHOO), "--phi", "1.0"]
        ppd4_metrics = eval_run_thesis_baseline("PPD v4", seed, ckpt_path, ppd4_flags)
        all_results["PPD v4"]["seeds"][str(seed)] = ppd4_metrics

    # Compute mean and std across seeds
    metric_keys = ["recall@20", "ndcg@20", "precision@20", "avgpop@20"]
    for method in all_results:
        for k in metric_keys:
            vals = [all_results[method]["seeds"][str(s)][k] for s in SEEDS]
            all_results[method]["mean"][k] = float(np.mean(vals))
            all_results[method]["std"][k] = float(np.std(vals))

    out_file = os.path.join(RESULTS_DIR, "baselines_yahoo.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n==================== BASELINE BENCHMARK SUMMARY (Yahoo) ====================")
    print(f"Saved to: {out_file}\n")
    print(f"{'Method':<12} | {'Recall@20':<18} | {'NDCG@20':<18} | {'Precision@20':<18} | {'AvgPop@20':<18}")
    print("-" * 95)
    for method in ["LightGCN", "TSP", "PPD v3", "PPD v4"]:
        m = all_results[method]["mean"]
        s = all_results[method]["std"]
        print(f"{method:<12} | {m['recall@20']:.6f} +/- {s['recall@20']:.6f} | {m['ndcg@20']:.6f} +/- {s['ndcg@20']:.6f} | {m['precision@20']:.6f} +/- {s['precision@20']:.6f} | {m['avgpop@20']:.2f} +/- {s['avgpop@20']:.2f}")


if __name__ == "__main__":
    main()
