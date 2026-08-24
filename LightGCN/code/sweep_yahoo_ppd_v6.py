#!/usr/bin/env python
"""
sweep_yahoo_ppd_v6.py
Step 2 & 3: Executes hyperparameter sweep for PPD v6 on Yahoo dataset.
Focuses on:
- astd_alpha_base: [0.05, 0.1, 0.15, 0.2]
- astd_topo_gate_init: [0.6, 0.75, 0.9]
- astd_topo_pop_damping: [0.05, 0.1, 0.2]
- astd_neg_hard_ratio: [0.0, 0.05, 0.1]

Tracks epoch-wise diagnostics: alpha, theta, lambda_fuse.
Selects best configuration based on Validation Pareto Hypervolume,
evaluates across seeds (2024, 2025, 2026), and formats Pareto Dominance report.
"""

import os
import sys
import json
import itertools
import subprocess
import re
import numpy as np

code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

import utilities

python_path = sys.executable

DATASET = "Yahoo"
LAYER = 2
RECDIM = 256
EPOCHS = 100
BPR_BATCH = 2048
LR = 0.001
SEEDS = [2024, 2025, 2026]

RESULTS_DIR = os.path.join(code_dir, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_run_output(stdout_text):
    prec_20, rec_20, ndcg_20, pop_20 = 0.0, 0.0, 0.0, 0.0
    val_hv = 0.0

    precision_match = re.search(r"'precision':\s*array\(\[(.*?)\]\)", stdout_text, re.DOTALL)
    recall_match = re.search(r"'recall':\s*array\(\[(.*?)\]\)", stdout_text, re.DOTALL)
    ndcg_match = re.search(r"'ndcg':\s*array\(\[(.*?)\]\)", stdout_text, re.DOTALL)
    pop_match = re.search(r"(?:Final Test\s+)?Average Popularity@20:\s*([\d]+(?:\.[\d]+)?)", stdout_text)
    hv_match = re.findall(r"Hypervolume=([\d]+(?:\.[\d]+)?)", stdout_text)
    best_val_hv_match = re.search(r"New best validation hypervolume\s*=\s*([\d]+(?:\.[\d]+)?)", stdout_text)

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
        pop_20 = float(pop_match.group(1).rstrip('.'))

    if best_val_hv_match:
        val_hv = float(best_val_hv_match.group(1).rstrip('.'))
    elif hv_match:
        val_hv = max([float(x.rstrip('.')) for x in hv_match])

    # Extract diagnostic trajectory
    epoch_alphas = [float(x) for x in re.findall(r"Alpha=([\d\.]+)", stdout_text)]
    epoch_thetas = [float(x) for x in re.findall(r"Theta=([\d\.]+)", stdout_text)]
    epoch_lambdas = [float(x) for x in re.findall(r"MeanLambda=([\d\.]+)", stdout_text)]

    return {
        "metrics": {
            "recall@20": float(rec_20),
            "ndcg@20": float(ndcg_20),
            "precision@20": float(prec_20),
            "avgpop@20": float(pop_20)
        },
        "val_hypervolume": float(val_hv),
        "diagnostics": {
            "mean_alpha_final": epoch_alphas[-1] if epoch_alphas else 0.0,
            "mean_theta_final": epoch_thetas[-1] if epoch_thetas else 0.0,
            "mean_lambda_final": epoch_lambdas[-1] if epoch_lambdas else 0.0,
            "alpha_trajectory": epoch_alphas,
            "theta_trajectory": epoch_thetas,
            "lambda_trajectory": epoch_lambdas
        }
    }


def run_ppd_v6_single(params, seed=2024):
    cmd = [
        python_path, "main.py",
        "--method", "ppd_v6",
        "--dataset", DATASET,
        "--layer", str(LAYER),
        "--recdim", str(RECDIM),
        "--epochs", str(EPOCHS),
        "--bpr_batch", str(BPR_BATCH),
        "--lr", str(LR),
        "--seed", str(seed),
        "--astd_alpha_base", str(params["astd_alpha_base"]),
        "--astd_topo_gate_init", str(params["astd_topo_gate_init"]),
        "--astd_topo_pop_damping", str(params["astd_topo_pop_damping"]),
        "--astd_neg_hard_ratio", str(params["astd_neg_hard_ratio"]),
        "--astd_neg_matched_ratio", str(params.get("astd_neg_matched_ratio", 0.4)),
        "--astd_neg_uniform_ratio", str(round(1.0 - params.get("astd_neg_matched_ratio", 0.4) - params["astd_neg_hard_ratio"], 2)),
        "--astd_highfreq_gamma", "0.1",
        "--astd_fuse_lambda_max", "0.5",
        "--astd_tail_phi", "1.0"
    ]

    res = subprocess.run(cmd, cwd=code_dir, capture_output=True, text=True)
    out = parse_run_output(res.stdout)
    return out


def main():
    print("==================== PPD v6 Yahoo Hyperparameter Sweep ====================")

    alpha_base_list = [0.05, 0.10, 0.15, 0.20]
    topo_gate_init_list = [0.60, 0.75, 0.90]
    topo_pop_damping_list = [0.05, 0.10, 0.20]
    neg_hard_ratio_list = [0.0, 0.05, 0.10]

    grid = list(itertools.product(
        alpha_base_list,
        topo_gate_init_list,
        topo_pop_damping_list,
        neg_hard_ratio_list
    ))

    print(f"Total Sweep Configurations: {len(grid)}")

    sweep_results = []
    best_val_hv = -1.0
    best_params = None

    for idx, (a_base, t_init, t_damp, n_hard) in enumerate(grid):
        params = {
            "astd_alpha_base": a_base,
            "astd_topo_gate_init": t_init,
            "astd_topo_pop_damping": t_damp,
            "astd_neg_hard_ratio": n_hard,
            "astd_neg_matched_ratio": 0.4
        }
        print(f"\n[{idx+1}/{len(grid)}] Testing Config: {params}")
        out = run_ppd_v6_single(params, seed=2024)
        val_hv = out["val_hypervolume"]
        test_m = out["metrics"]
        diag = out["diagnostics"]

        print(f"--> Val Hypervolume: {val_hv:.8f} | Test: Rec={test_m['recall@20']:.6f}, NDCG={test_m['ndcg@20']:.6f}, Prec={test_m['precision@20']:.6f}, Pop={test_m['avgpop@20']:.2f}")
        print(f"--> Diagnostics: Alpha={diag['mean_alpha_final']:.4f}, Theta={diag['mean_theta_final']:.4f}, LambdaFuse={diag['mean_lambda_final']:.4f}")

        record = {
            "params": params,
            "val_hypervolume": val_hv,
            "test_metrics_seed2024": test_m,
            "diagnostics": diag
        }
        sweep_results.append(record)

        if val_hv > best_val_hv:
            best_val_hv = val_hv
            best_params = params
            print(f"*** NEW BEST CONFIG (Val HV: {best_val_hv:.8f}) ***")

    # Save sweep results
    sweep_file = os.path.join(RESULTS_DIR, "ppd_v6_yahoo_sweep.json")
    with open(sweep_file, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nAll sweep results saved to: {sweep_file}")

    print(f"\n==================== Top Winning Config ====================")
    print(json.dumps(best_params, indent=2))

    # Evaluate winning config across 3 seeds
    print(f"\n==================== Evaluating Best Config Across Seeds {SEEDS} ====================")
    seed_metrics = []
    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        out = run_ppd_v6_single(best_params, seed=seed)
        m = out["metrics"]
        seed_metrics.append(m)
        print(f"Seed {seed} Test Metrics:", m)

    mean_metrics = {
        "recall@20": float(np.mean([m["recall@20"] for m in seed_metrics])),
        "ndcg@20": float(np.mean([m["ndcg@20"] for m in seed_metrics])),
        "precision@20": float(np.mean([m["precision@20"] for m in seed_metrics])),
        "avgpop@20": float(np.mean([m["avgpop@20"] for m in seed_metrics]))
    }
    std_metrics = {
        "recall@20": float(np.std([m["recall@20"] for m in seed_metrics])),
        "ndcg@20": float(np.std([m["ndcg@20"] for m in seed_metrics])),
        "precision@20": float(np.std([m["precision@20"] for m in seed_metrics])),
        "avgpop@20": float(np.std([m["avgpop@20"] for m in seed_metrics]))
    }

    final_results = {
        "best_params": best_params,
        "best_val_hypervolume": best_val_hv,
        "seeds": {str(s): m for s, m in zip(SEEDS, seed_metrics)},
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics
    }
    final_file = os.path.join(RESULTS_DIR, "ppd_v6_yahoo_final.json")
    with open(final_file, "w") as f:
        json.dump(final_results, f, indent=2)

    # Load baseline metrics
    baselines_file = os.path.join(RESULTS_DIR, "baselines_yahoo.json")
    if os.path.exists(baselines_file):
        with open(baselines_file, "r") as f:
            baselines_data = json.load(f)
        baseline_means = {k: v["mean"] for k, v in baselines_data.items()}
    else:
        # Fallback to canonical
        baseline_means = {
            "LightGCN": {"recall@20": 0.140645, "ndcg@20": 0.064747, "precision@20": 0.010869, "avgpop@20": 497.359},
            "TSP": {"recall@20": 0.140477, "ndcg@20": 0.064905, "precision@20": 0.010953, "avgpop@20": 485.765},
            "PPD v3": {"recall@20": 0.142534, "ndcg@20": 0.064853, "precision@20": 0.011120, "avgpop@20": 446.400},
            "PPD v4": {"recall@20": 0.139401, "ndcg@20": 0.061455, "precision@20": 0.010911, "avgpop@20": 471.384}
        }

    report = utilities.format_pareto_report(DATASET, "PPD v6 / ASTD-PD (Ours)", mean_metrics, baseline_means)
    print("\n" + report)

    report_file = os.path.join(RESULTS_DIR, "yahoo_pareto_report.md")
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
