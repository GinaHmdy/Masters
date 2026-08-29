#!/usr/bin/env python
"""
run_ppd_v6_experiments.py
Automated evaluation and experiment runner for PPD v6 / ASTD-PD across
Gowalla, KuaiRec, Yahoo, and Coat.
Evaluates Pareto dominance across Recall@20, NDCG@20, Precision@20, and AvgPop@20.
"""

import os
import sys
import argparse
import subprocess
import re
import numpy as np
from pprint import pprint

code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

import utilities

python_path = sys.executable

# Recommended dataset configurations and hyperparameter priors
DATASET_CONFIGS = {
    "Yahoo": {
        "layer": 2,
        "recdim": 256,
        "epochs": 100,
        "bpr_batch": 2048,
        "lr": 0.001,
        "astd_alpha_base": 0.10,
        "astd_topo_gate_init": 0.75,
        "astd_topo_pop_damping": 0.1,
        "astd_highfreq_gamma": 0.1,
        "astd_neg_uniform_ratio": 0.5,
        "astd_neg_matched_ratio": 0.4,
        "astd_neg_hard_ratio": 0.1,
        "astd_fuse_lambda_max": 0.5,
        "astd_tail_phi": 1.0,
        "canonical": {
            "LightGCN": {"recall@20": 0.140645, "ndcg@20": 0.064747, "precision@20": 0.010869, "avgpop@20": 497.359},
            "TSP": {"recall@20": 0.140477, "ndcg@20": 0.064905, "precision@20": 0.010953, "avgpop@20": 485.765},
            "PPD v3": {"recall@20": 0.142534, "ndcg@20": 0.064853, "precision@20": 0.011120, "avgpop@20": 446.400},
            "PPD v4": {"recall@20": 0.139401, "ndcg@20": 0.061455, "precision@20": 0.010911, "avgpop@20": 471.384}
        }
    },
    "Gowalla": {
        "layer": 3,
        "recdim": 64,
        "epochs": 100,
        "bpr_batch": 2048,
        "lr": 0.001,
        "astd_alpha_base": 0.40,
        "astd_topo_gate_init": 0.40,
        "astd_topo_pop_damping": 0.05,
        "astd_highfreq_gamma": 0.1,
        "astd_neg_uniform_ratio": 0.4,
        "astd_neg_matched_ratio": 0.4,
        "astd_neg_hard_ratio": 0.2,
        "astd_fuse_lambda_max": 0.5,
        "astd_tail_phi": 1.0,
        "canonical": {
            "LightGCN": {"recall@20": 0.121115, "ndcg@20": 0.101116, "precision@20": 0.036710, "avgpop@20": 253.862},
            "TSP": {"recall@20": 0.120943, "ndcg@20": 0.101026, "precision@20": 0.036813, "avgpop@20": 250.184},
            "PPD v3": {"recall@20": 0.117965, "ndcg@20": 0.097658, "precision@20": 0.036280, "avgpop@20": 210.666},
            "PPD v4": {"recall@20": 0.119465, "ndcg@20": 0.098808, "precision@20": 0.036466, "avgpop@20": 211.814}
        }
    },
    "KuaiRec": {
        "layer": 2,
        "recdim": 256,
        "epochs": 100,
        "bpr_batch": 2048,
        "lr": 0.001,
        "astd_alpha_base": 0.35,
        "astd_topo_gate_init": 0.50,
        "astd_topo_pop_damping": 0.05,
        "astd_highfreq_gamma": 0.1,
        "astd_neg_uniform_ratio": 0.3,
        "astd_neg_matched_ratio": 0.5,
        "astd_neg_hard_ratio": 0.2,
        "astd_fuse_lambda_max": 0.5,
        "astd_tail_phi": 1.0,
        "canonical": {
            "LightGCN": {"recall@20": 0.005553, "ndcg@20": 0.005015, "precision@20": 0.005709, "avgpop@20": 1058.105},
            "TSP": {"recall@20": 0.004384, "ndcg@20": 0.003556, "precision@20": 0.004220, "avgpop@20": 1038.651},
            "PPD v3": {"recall@20": 0.006518, "ndcg@20": 0.005389, "precision@20": 0.006418, "avgpop@20": 564.687},
            "PPD v4": {"recall@20": 0.027168, "ndcg@20": 0.033133, "precision@20": 0.032199, "avgpop@20": 690.297}
        }
    },
    "Coat": {
        "layer": 2,
        "recdim": 64,
        "epochs": 50,
        "bpr_batch": 512,
        "lr": 0.001,
        "astd_alpha_base": 0.10,
        "astd_topo_gate_init": 0.60,
        "astd_topo_pop_damping": 0.05,
        "astd_highfreq_gamma": 0.1,
        "astd_neg_uniform_ratio": 0.5,
        "astd_neg_matched_ratio": 0.4,
        "astd_neg_hard_ratio": 0.1,
        "astd_fuse_lambda_max": 0.3,
        "astd_tail_phi": 1.0,
        "canonical": {
            "LightGCN": {"recall@20": 0.060000, "ndcg@20": 0.032000, "precision@20": 0.012000, "avgpop@20": 15.000},
            "TSP": {"recall@20": 0.061000, "ndcg@20": 0.032500, "precision@20": 0.012200, "avgpop@20": 14.800},
            "PPD v3": {"recall@20": 0.058000, "ndcg@20": 0.030000, "precision@20": 0.011500, "avgpop@20": 12.500},
            "PPD v4": {"recall@20": 0.059000, "ndcg@20": 0.031000, "precision@20": 0.011800, "avgpop@20": 12.800}
        }
    }
}


def parse_output_metrics(stdout_text):
    """
    Parses recall, ndcg, precision, avgpop from main.py output.
    """
    prec_20, rec_20, ndcg_20, pop_20 = 0.0, 0.0, 0.0, 0.0

    precision_match = re.search(r"'precision':\s*array\(\[(.*?)\]\)", stdout_text, re.S)
    recall_match = re.search(r"'recall':\s*array\(\[(.*?)\]\)", stdout_text, re.S)
    ndcg_match = re.search(r"'ndcg':\s*array\(\[(.*?)\]\)", stdout_text, re.S)
    pop_match = re.search(r"Final Test Average Popularity@20:\s*([\d\.]+)", stdout_text)

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
        "recall@20": rec_20,
        "ndcg@20": ndcg_20,
        "precision@20": prec_20,
        "avgpop@20": pop_20
    }


def run_experiment(dataset, seeds=(2024, 2025, 2026), epochs=None, extra_flags=None):
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}")

    cfg = DATASET_CONFIGS[dataset]
    n_epochs = epochs or cfg["epochs"]
    extra_flags = extra_flags or []

    seed_results = []

    for seed in seeds:
        print(f"\n==================== Running PPD v6 on {dataset} (Seed: {seed}) ====================")
        cmd = [
            python_path, "main.py",
            "--method", "ppd_v6",
            "--dataset", dataset,
            "--layer", str(cfg["layer"]),
            "--recdim", str(cfg["recdim"]),
            "--epochs", str(n_epochs),
            "--bpr_batch", str(cfg["bpr_batch"]),
            "--lr", str(cfg["lr"]),
            "--seed", str(seed),
            "--astd_alpha_base", str(cfg["astd_alpha_base"]),
            "--astd_topo_gate_init", str(cfg["astd_topo_gate_init"]),
            "--astd_topo_pop_damping", str(cfg["astd_topo_pop_damping"]),
            "--astd_highfreq_gamma", str(cfg["astd_highfreq_gamma"]),
            "--astd_neg_uniform_ratio", str(cfg["astd_neg_uniform_ratio"]),
            "--astd_neg_matched_ratio", str(cfg["astd_neg_matched_ratio"]),
            "--astd_neg_hard_ratio", str(cfg["astd_neg_hard_ratio"]),
            "--astd_fuse_lambda_max", str(cfg["astd_fuse_lambda_max"]),
            "--astd_tail_phi", str(cfg["astd_tail_phi"])
        ] + extra_flags

        print("Executing command:", " ".join(cmd))
        res = subprocess.run(cmd, cwd=code_dir, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print("[STDERR]", res.stderr)

        metrics = parse_output_metrics(res.stdout)
        seed_results.append(metrics)
        print(f"Seed {seed} Metrics:", metrics)

    # Compute mean and std
    mean_metrics = {
        "recall@20": float(np.mean([m["recall@20"] for m in seed_results])),
        "ndcg@20": float(np.mean([m["ndcg@20"] for m in seed_results])),
        "precision@20": float(np.mean([m["precision@20"] for m in seed_results])),
        "avgpop@20": float(np.mean([m["avgpop@20"] for m in seed_results])),
    }
    std_metrics = {
        "recall@20": float(np.std([m["recall@20"] for m in seed_results])),
        "ndcg@20": float(np.std([m["ndcg@20"] for m in seed_results])),
        "precision@20": float(np.std([m["precision@20"] for m in seed_results])),
        "avgpop@20": float(np.std([m["avgpop@20"] for m in seed_results])),
    }

    report = utilities.format_pareto_report(
        dataset, "PPD v6 / ASTD-PD", mean_metrics, cfg["canonical"]
    )
    print(report)

    return {
        "dataset": dataset,
        "seed_results": seed_results,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "report": report
    }


def main():
    parser = argparse.ArgumentParser(description="PPD v6 Experiment Runner")
    parser.add_argument("--dataset", type=str, default="Yahoo", choices=["Yahoo", "Gowalla", "KuaiRec", "Coat", "all"])
    parser.add_argument("--seeds", type=str, default="2024,2025,2026")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]

    extra_flags = ["--smoke_test"] if args.smoke_test else []

    all_results = {}
    for d in datasets:
        res = run_experiment(d, seeds=seeds, epochs=args.epochs, extra_flags=extra_flags)
        all_results[d] = res

    print("\n==================== SUMMARY COMPLETED ====================")
    for d, res in all_results.items():
        print(res["report"])


if __name__ == "__main__":
    main()
