#!/usr/bin/env python
"""
run_all_remaining_datasets.py
Automated Baseline Benchmarking and PPD v6 Pareto Evaluation across
Gowalla, KuaiRec, and Coat datasets over seeds 2024, 2025, 2026.
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

RESULTS_DIR = os.path.join(code_dir, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CHECKPOINTS_DIR = os.path.join(code_dir, "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

DATASET_CONFIGS = {
    "Coat": {
        "layer": 2,
        "recdim": 64,
        "epochs": 50,
        "bpr_batch": 512,
        "lr": 0.001,
        "beta_ppd": 0.20,
        "tsp_flags": ["--tsp", "--alpha_tsp", "0.1", "--theta_tsp", "0.60", "--top_m_tsp", "15", "--k_tsp", "1"],
        "ppd_v6_params": {
            "astd_alpha_base": 0.10,
            "astd_topo_gate_init": 0.60,
            "astd_topo_pop_damping": 0.05,
            "astd_neg_hard_ratio": 0.10,
            "astd_neg_matched_ratio": 0.40,
            "astd_neg_uniform_ratio": 0.50,
            "astd_temperature_acc": 0.20,
            "astd_temperature_deb": 0.20,
            "astd_fuse_lambda_max": 0.30,
            "astd_tail_phi": 1.5,
            "astd_lambda_acc": 1.0,
            "astd_lambda_deb": 0.3,
            "astd_lambda_final": 1.0,
            "astd_lambda_ind": 0.05,
            "astd_inference_pop_penalty": 0.0
        }
    },
    "Gowalla": {
        "layer": 3,
        "recdim": 64,
        "epochs": 100,
        "bpr_batch": 2048,
        "lr": 0.001,
        "beta_ppd": 0.20,
        "tsp_flags": ["--tsp", "--alpha_tsp", "0.1", "--theta_tsp", "0.90", "--top_m_tsp", "15", "--k_tsp", "1"],
        "ppd_v6_params": {
            "astd_alpha_base": 0.40,
            "astd_topo_gate_init": 0.40,
            "astd_topo_pop_damping": 0.05,
            "astd_neg_hard_ratio": 0.15,
            "astd_neg_matched_ratio": 0.40,
            "astd_neg_uniform_ratio": 0.45,
            "astd_temperature_acc": 0.20,
            "astd_temperature_deb": 0.20,
            "astd_fuse_lambda_max": 0.35,
            "astd_tail_phi": 1.5,
            "astd_lambda_acc": 1.0,
            "astd_lambda_deb": 0.3,
            "astd_lambda_final": 1.0,
            "astd_lambda_ind": 0.05,
            "astd_inference_pop_penalty": 0.0
        }
    },
    "KuaiRec": {
        "layer": 2,
        "recdim": 256,
        "epochs": 100,
        "bpr_batch": 2048,
        "lr": 0.001,
        "beta_ppd": 2.00,
        "tsp_flags": ["--tsp", "--alpha_tsp", "0.1", "--theta_tsp", "0.90", "--top_m_tsp", "15", "--k_tsp", "1"],
        "ppd_v6_params": {
            "astd_alpha_base": 0.35,
            "astd_topo_gate_init": 0.50,
            "astd_topo_pop_damping": 0.05,
            "astd_neg_hard_ratio": 0.15,
            "astd_neg_matched_ratio": 0.45,
            "astd_neg_uniform_ratio": 0.40,
            "astd_temperature_acc": 0.20,
            "astd_temperature_deb": 0.20,
            "astd_fuse_lambda_max": 0.35,
            "astd_tail_phi": 1.5,
            "astd_lambda_acc": 1.0,
            "astd_lambda_deb": 0.3,
            "astd_lambda_final": 1.0,
            "astd_lambda_ind": 0.05,
            "astd_inference_pop_penalty": 0.0
        }
    }
}


def parse_metrics_from_output(stdout_text):
    prec_20, rec_20, ndcg_20, pop_20 = 0.0, 0.0, 0.0, 0.0

    precision_match = re.search(r"'precision':\s*array\(\[(.*?)\]\)", stdout_text, re.DOTALL)
    recall_match = re.search(r"'recall':\s*array\(\[(.*?)\]\)", stdout_text, re.DOTALL)
    ndcg_match = re.search(r"'ndcg':\s*array\(\[(.*?)\]\)", stdout_text, re.DOTALL)
    pop_match = re.search(r"(?:Final Test\s+)?Average Popularity@20:\s*([\d]+(?:\.[\d]+)?)", stdout_text)

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

    return {
        "recall@20": float(rec_20),
        "ndcg@20": float(ndcg_20),
        "precision@20": float(prec_20),
        "avgpop@20": float(pop_20)
    }


def benchmark_dataset_baselines(dataset):
    cfg = DATASET_CONFIGS[dataset]
    layer = cfg["layer"]
    recdim = cfg["recdim"]
    epochs = cfg["epochs"]
    batch = cfg["bpr_batch"]
    lr = cfg["lr"]
    beta = cfg["beta_ppd"]
    tsp_flags = cfg["tsp_flags"]

    print(f"\n=======================================================")
    print(f"       BENCHMARKING BASELINES: {dataset}")
    print(f"=======================================================")

    results = {
        "LightGCN": {"seeds": {}, "mean": {}, "std": {}},
        "TSP": {"seeds": {}, "mean": {}, "std": {}},
        "PPD v3": {"seeds": {}, "mean": {}, "std": {}},
        "PPD v4": {"seeds": {}, "mean": {}, "std": {}},
    }

    for seed in SEEDS:
        seed_ckpt = os.path.join(CHECKPOINTS_DIR, f"lgn-{dataset}-{layer}-{recdim}-seed{seed}-best.pth.tar")
        default_ckpt = os.path.join(CHECKPOINTS_DIR, f"lgn-{dataset}-{layer}-{recdim}-best.pth.tar")

        # 1. Train LightGCN if checkpoint does not exist
        if not os.path.exists(seed_ckpt):
            print(f"\n--- [LightGCN {dataset}] Training Seed {seed} ---")
            cmd = [
                python_path, "main.py",
                "--method", "lightgcn",
                "--dataset", dataset,
                "--layer", str(layer),
                "--recdim", str(recdim),
                "--epochs", str(epochs),
                "--bpr_batch", str(batch),
                "--lr", str(lr),
                "--seed", str(seed)
            ]
            res = subprocess.run(cmd, cwd=code_dir, capture_output=True, text=True)
            lgn_m = parse_metrics_from_output(res.stdout)
            if os.path.exists(default_ckpt):
                shutil.copyfile(default_ckpt, seed_ckpt)
        else:
            print(f"\n--- [LightGCN {dataset}] Evaluating Existing Checkpoint for Seed {seed} ---")
            cmd = [
                python_path, "run_thesis.py",
                "--dataset", dataset,
                "--load", "1",
                "--model", "lgn",
                "--layer", str(layer),
                "--recdim", str(recdim),
                "--seed", str(seed),
                "--ckpt", seed_ckpt,
                "--fairness"
            ]
            res = subprocess.run(cmd, cwd=code_dir, capture_output=True, text=True)
            lgn_m = parse_metrics_from_output(res.stdout)

        results["LightGCN"]["seeds"][str(seed)] = lgn_m
        print(f"[LightGCN {dataset} Seed {seed}] Metrics: {lgn_m}")

        # 2. TSP
        print(f"\n--- [TSP {dataset}] Evaluating Seed {seed} ---")
        cmd_tsp = [
            python_path, "run_thesis.py",
            "--dataset", dataset,
            "--load", "1",
            "--model", "lgn",
            "--layer", str(layer),
            "--recdim", str(recdim),
            "--seed", str(seed),
            "--ckpt", seed_ckpt,
            "--fairness"
        ] + tsp_flags
        res_tsp = subprocess.run(cmd_tsp, cwd=code_dir, capture_output=True, text=True)
        tsp_m = parse_metrics_from_output(res_tsp.stdout)
        results["TSP"]["seeds"][str(seed)] = tsp_m
        print(f"[TSP {dataset} Seed {seed}] Metrics: {tsp_m}")

        # 3. PPD v3
        print(f"\n--- [PPD v3 {dataset}] Evaluating Seed {seed} ---")
        cmd_ppd3 = [
            python_path, "run_thesis.py",
            "--dataset", dataset,
            "--load", "1",
            "--model", "lgn",
            "--layer", str(layer),
            "--recdim", str(recdim),
            "--seed", str(seed),
            "--ckpt", seed_ckpt,
            "--ppd", "--ppd_variant", "3",
            "--beta", str(beta),
            "--phi", "1.0",
            "--fairness"
        ]
        res_ppd3 = subprocess.run(cmd_ppd3, cwd=code_dir, capture_output=True, text=True)
        ppd3_m = parse_metrics_from_output(res_ppd3.stdout)
        results["PPD v3"]["seeds"][str(seed)] = ppd3_m
        print(f"[PPD v3 {dataset} Seed {seed}] Metrics: {ppd3_m}")

        # 4. PPD v4
        print(f"\n--- [PPD v4 {dataset}] Evaluating Seed {seed} ---")
        cmd_ppd4 = [
            python_path, "run_thesis.py",
            "--dataset", dataset,
            "--load", "1",
            "--model", "lgn",
            "--layer", str(layer),
            "--recdim", str(recdim),
            "--seed", str(seed),
            "--ckpt", seed_ckpt,
            "--ppd", "--ppd_variant", "4",
            "--beta", str(beta),
            "--phi", "1.0",
            "--fairness"
        ]
        res_ppd4 = subprocess.run(cmd_ppd4, cwd=code_dir, capture_output=True, text=True)
        ppd4_m = parse_metrics_from_output(res_ppd4.stdout)
        results["PPD v4"]["seeds"][str(seed)] = ppd4_m
        print(f"[PPD v4 {dataset} Seed {seed}] Metrics: {ppd4_m}")

    metric_keys = ["recall@20", "ndcg@20", "precision@20", "avgpop@20"]
    for method in results:
        for k in metric_keys:
            vals = [results[method]["seeds"][str(s)][k] for s in SEEDS]
            results[method]["mean"][k] = float(np.mean(vals))
            results[method]["std"][k] = float(np.std(vals))

    baseline_file = os.path.join(RESULTS_DIR, f"baselines_{dataset.lower()}.json")
    with open(baseline_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_ppd_v6_dataset(dataset):
    cfg = DATASET_CONFIGS[dataset]
    layer = cfg["layer"]
    recdim = cfg["recdim"]
    epochs = cfg["epochs"]
    batch = cfg["bpr_batch"]
    lr = cfg["lr"]
    params = cfg["ppd_v6_params"]

    print(f"\n=======================================================")
    print(f"       RUNNING PPD v6 / ASTD-PD: {dataset}")
    print(f"=======================================================")

    seed_results = []

    for seed in SEEDS:
        print(f"\n--- [PPD v6 {dataset}] Training Seed {seed} ---")
        cmd = [
            python_path, "main.py",
            "--method", "ppd_v6",
            "--dataset", dataset,
            "--layer", str(layer),
            "--recdim", str(recdim),
            "--epochs", str(epochs),
            "--bpr_batch", str(batch),
            "--lr", str(lr),
            "--seed", str(seed),
            "--astd_alpha_base", str(params["astd_alpha_base"]),
            "--astd_topo_gate_init", str(params["astd_topo_gate_init"]),
            "--astd_topo_pop_damping", str(params["astd_topo_pop_damping"]),
            "--astd_neg_hard_ratio", str(params["astd_neg_hard_ratio"]),
            "--astd_neg_matched_ratio", str(params["astd_neg_matched_ratio"]),
            "--astd_neg_uniform_ratio", str(params["astd_neg_uniform_ratio"]),
            "--astd_temperature_acc", str(params["astd_temperature_acc"]),
            "--astd_temperature_deb", str(params["astd_temperature_deb"]),
            "--astd_fuse_lambda_max", str(params["astd_fuse_lambda_max"]),
            "--astd_tail_phi", str(params["astd_tail_phi"]),
            "--astd_lambda_acc", str(params["astd_lambda_acc"]),
            "--astd_lambda_deb", str(params["astd_lambda_deb"]),
            "--astd_lambda_final", str(params["astd_lambda_final"]),
            "--astd_lambda_ind", str(params["astd_lambda_ind"]),
            "--astd_inference_pop_penalty", str(params["astd_inference_pop_penalty"])
        ]
        res = subprocess.run(cmd, cwd=code_dir, capture_output=True, text=True)
        m = parse_metrics_from_output(res.stdout)
        seed_results.append(m)
        print(f"[PPD v6 {dataset} Seed {seed}] Metrics: {m}")

    mean_metrics = {
        "recall@20": float(np.mean([m["recall@20"] for m in seed_results])),
        "ndcg@20": float(np.mean([m["ndcg@20"] for m in seed_results])),
        "precision@20": float(np.mean([m["precision@20"] for m in seed_results])),
        "avgpop@20": float(np.mean([m["avgpop@20"] for m in seed_results]))
    }
    std_metrics = {
        "recall@20": float(np.std([m["recall@20"] for m in seed_results])),
        "ndcg@20": float(np.std([m["ndcg@20"] for m in seed_results])),
        "precision@20": float(np.std([m["precision@20"] for m in seed_results])),
        "avgpop@20": float(np.std([m["avgpop@20"] for m in seed_results]))
    }

    final_results = {
        "dataset": dataset,
        "params": params,
        "seeds": {str(s): m for s, m in zip(SEEDS, seed_results)},
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics
    }

    v6_file = os.path.join(RESULTS_DIR, f"ppd_v6_{dataset.lower()}_final.json")
    with open(v6_file, "w") as f:
        json.dump(final_results, f, indent=2)

    return final_results


def evaluate_and_report(dataset, baseline_results, ppd_v6_results):
    baseline_means = {k: v["mean"] for k, v in baseline_results.items()}
    report = utilities.format_pareto_report(
        dataset, "PPD v6 / ASTD-PD (Ours)", ppd_v6_results["mean_metrics"], baseline_means
    )
    print("\n" + report)

    report_file = os.path.join(RESULTS_DIR, f"{dataset.lower()}_pareto_report.md")
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Report saved to: {report_file}")
    return report


def main():
    datasets = ["Coat", "KuaiRec", "Gowalla"]

    all_reports = {}
    summary_data = {}

    for d in datasets:
        b_res = benchmark_dataset_baselines(d)
        v6_res = run_ppd_v6_dataset(d)
        rep = evaluate_and_report(d, b_res, v6_res)
        all_reports[d] = rep
        summary_data[d] = {
            "baselines": b_res,
            "ppd_v6": v6_res
        }

    summary_file = os.path.join(RESULTS_DIR, "all_datasets_pareto_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n=======================================================")
    print("       ALL DATASETS EVALUATION COMPLETED")
    print("=======================================================")
    for d, rep in all_reports.items():
        print(rep)


if __name__ == "__main__":
    main()
