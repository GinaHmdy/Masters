# run_thesis_experiments.py
import subprocess
import os
import re
import shutil
import time

python_path = "/home/mohabmostafa/miniconda3/envs/gMasters/bin/python"
code_dir = "/home/mohabmostafa/98-Downloads/Masters/LightGCN/code"
checkpoints_dir = os.path.join(code_dir, "checkpoints")

datasets_config = {
    "Gowalla": {"layer": 3, "recdim": 64},
    "KuaiRec": {"layer": 2, "recdim": 256},
    "Yahoo": {"layer": 2, "recdim": 256}
}

# 1. Wait for currently running training tasks to complete
print("[1/3] Waiting for current standard SGL training runs to finish...")
while True:
    kuairec_done = os.path.exists(os.path.join(checkpoints_dir, "dap-KuaiRec-2-256.pth.tar"))
    yahoo_done = os.path.exists(os.path.join(checkpoints_dir, "dap-Yahoo-2-256.pth.tar"))
    if kuairec_done and yahoo_done:
        print("[OK] Standard SGL training for KuaiRec and Yahoo is done!")
        break
    time.sleep(5)

# Copy standard checkpoints to -sgl.pth.tar
shutil.copy(os.path.join(checkpoints_dir, "dap-Gowalla-3-64.pth.tar"), os.path.join(checkpoints_dir, "dap-Gowalla-3-64-sgl.pth.tar"))
shutil.copy(os.path.join(checkpoints_dir, "dap-KuaiRec-2-256.pth.tar"), os.path.join(checkpoints_dir, "dap-KuaiRec-2-256-sgl.pth.tar"))
shutil.copy(os.path.join(checkpoints_dir, "dap-Yahoo-2-256.pth.tar"), os.path.join(checkpoints_dir, "dap-Yahoo-2-256-sgl.pth.tar"))
print("[OK] Standard SGL checkpoints copied.")

# 2. Train dynamic ego attention checkpoints
print("[2/3] Training dynamic ego attention models...")
for dataset, cfg in datasets_config.items():
    layer = str(cfg["layer"])
    recdim = str(cfg["recdim"])
    target_ckpt = os.path.join(checkpoints_dir, f"dap-{dataset}-{layer}-{recdim}-sgl-dynamic.pth.tar")
    
    if os.path.exists(target_ckpt):
        print(f"[Skip] {dataset} dynamic checkpoint already exists.")
        continue
        
    print(f"\n==================== Training Dynamic Ego model for {dataset} ====================")
    # Remove old temporary file if exists
    temp_ckpt = os.path.join(checkpoints_dir, f"dap-{dataset}-{layer}-{recdim}.pth.tar")
    if os.path.exists(temp_ckpt):
        os.remove(temp_ckpt)
        
    train_cmd = [
        python_path, "main.py",
        "--dataset", dataset,
        "--model", "dap",
        "--layer", layer,
        "--recdim", recdim,
        "--epochs", "15",
        "--sgl",
        "--sgl_weight", "0.1",
        "--dynamic_ego"
    ]
    print("Running:", " ".join(train_cmd))
    subprocess.run(train_cmd, cwd=code_dir, check=True)
    
    # Copy checkpoint
    shutil.copy(temp_ckpt, target_ckpt)
    print(f"[OK] Saved {target_ckpt}")

# 3. Grid Search evaluation
print("\n[3/3] Starting grid search evaluation...")

grid = {
    "Gowalla": {
        "layers": 3,
        "recdim": 64,
        "betas": [0.20, 0.50],
        "ego_weights": [0.6, -1.0],
        "dynamic_ego": True,
        "ref": {"recall": 0.076893, "ndcg": 0.054366, "precision": 0.023447, "popularity": 258.450}
    },
    "KuaiRec": {
        "layers": 2,
        "recdim": 256,
        "betas": [0.5, 1.0, 2.0],
        "ego_weights": [-1.0],
        "dynamic_ego": True,
        "ref": {"recall": 0.233261, "ndcg": 0.222384, "precision": 0.106575, "popularity": 390.871}
    },
    "Yahoo": {
        "layers": 2,
        "recdim": 256,
        "betas": [0.02, 0.05, 0.07, 0.15],
        "ego_weights": [-1.0],
        "dynamic_ego": True,
        "ref": {"recall": 0.088663, "ndcg": 0.073549, "precision": 0.027056, "popularity": 280.957}
    }
}

def classify_run(rec, ndcg, prec, pop, ref):
    # Accuracies must be higher or equal to reference
    is_dominant = (rec >= ref["recall"] and ndcg >= ref["ndcg"] and prec >= ref["precision"] and pop < ref["popularity"])
    
    # Accuracy Win: accuracy metrics > 5% higher with popularity within 1% tolerance
    # Let's check percent deltas for accuracy
    rec_d = (rec - ref["recall"]) / ref["recall"]
    ndcg_d = (ndcg - ref["ndcg"]) / ref["ndcg"]
    prec_d = (prec - ref["precision"]) / ref["precision"]
    pop_d = (pop - ref["popularity"]) / ref["popularity"]
    
    is_accuracy_win = (rec_d >= 0.05 and ndcg_d >= 0.05 and prec_d >= 0.05 and pop_d <= 0.01)
    
    # Fairness Win: popularity >20% lower with accuracy metrics within 1% tolerance
    is_fairness_win = (pop_d <= -0.20 and rec_d >= -0.01 and ndcg_d >= -0.01 and prec_d >= -0.01)
    
    if is_dominant:
        return "A. DOMINANT"
    elif is_accuracy_win:
        return "B. ACCURACY-WIN"
    elif is_fairness_win:
        return "C. FAIRNESS-WIN"
    else:
        return "D. TRADE-OFF/LOSS"

grid_results = []

for dataset, info in grid.items():
    layer = str(info["layers"])
    recdim = str(info["recdim"])
    betas = info["betas"]
    ego_weights = info["ego_weights"]
    ref = info["ref"]
    
    # Prepare list of configurations to evaluate
    configs = []
    # Fixed or uniform weights
    for ew in ego_weights:
        configs.append({"ego_weight": ew, "dynamic": False})
    # Dynamic weight
    if info["dynamic_ego"]:
        configs.append({"ego_weight": -1.0, "dynamic": True})
        
    for beta in betas:
        for cfg_ew in configs:
            ew = cfg_ew["ego_weight"]
            is_dyn = cfg_ew["dynamic"]
            
            ew_str = "dynamic" if is_dyn else f"{ew:.1f}"
            ckpt_name = f"dap-{dataset}-{layer}-{recdim}-sgl-dynamic.pth.tar" if is_dyn else f"dap-{dataset}-{layer}-{recdim}-sgl.pth.tar"
            ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
            
            print(f"\nEvaluating: Dataset={dataset} | Beta={beta} | Ego={ew_str}")
            
            eval_cmd = [
                python_path, "run_thesis.py",
                "--dataset", dataset,
                "--load", "1",
                "--model", "dap",
                "--layer", layer,
                "--recdim", recdim,
                "--ckpt", ckpt_path,
                "--dap",
                "--tsp_hodge",
                "--ppd",
                "--ppd_variant", "4",
                "--beta", str(beta),
                "--phi", "1.0",
                "--theta_tsp", "0.90",
                "--top_m_tsp", "15",
                "--k_tsp", "1",
                "--alpha_tsp", "0.1",
                "--fairness"
            ]
            if is_dyn:
                eval_cmd.append("--dynamic_ego")
            else:
                eval_cmd += ["--ego_weight", str(ew)]
                
            print("Command:", " ".join(eval_cmd))
            res = subprocess.run(eval_cmd, cwd=code_dir, capture_output=True, text=True)
            output = res.stdout
            
            # Print stderr if there were warnings/errors
            if res.stderr:
                print(res.stderr)
                
            # Parse metrics
            precision_match = re.search(r"'precision':\s*array\(\[(.*?)\]\)", output)
            recall_match = re.search(r"'recall':\s*array\(\[(.*?)\]\)", output)
            ndcg_match = re.search(r"'ndcg':\s*array\(\[(.*?)\]\)", output)
            popularity_match = re.search(r"Average Popularity@20:\s*([\d\.]+)", output)
            
            prec_20, rec_20, ndcg_20, pop_20 = 0.0, 0.0, 0.0, 0.0
            
            if precision_match:
                clean_arr = precision_match.group(1).replace('\n', ' ').strip()
                vals = [float(x) for x in re.split(r'\s+|,', clean_arr) if x]
                prec_20 = vals[3]
            if recall_match:
                clean_arr = recall_match.group(1).replace('\n', ' ').strip()
                vals = [float(x) for x in re.split(r'\s+|,', clean_arr) if x]
                rec_20 = vals[3]
            if ndcg_match:
                clean_arr = ndcg_match.group(1).replace('\n', ' ').strip()
                vals = [float(x) for x in re.split(r'\s+|,', clean_arr) if x]
                ndcg_20 = vals[3]
            if popularity_match:
                pop_20 = float(popularity_match.group(1))
                
            # Compute deltas
            rec_delta = (rec_20 - ref["recall"]) / ref["recall"] * 100
            ndcg_delta = (ndcg_20 - ref["ndcg"]) / ref["ndcg"] * 100
            prec_delta = (prec_20 - ref["precision"]) / ref["precision"] * 100
            pop_delta = (pop_20 - ref["popularity"]) / ref["popularity"] * 100
            
            classification = classify_run(rec_20, ndcg_20, prec_20, pop_20, ref)
            
            run_result = {
                "dataset": dataset,
                "beta": beta,
                "ego": ew_str,
                "recall": rec_20,
                "ndcg": ndcg_20,
                "precision": prec_20,
                "popularity": pop_20,
                "rec_delta": rec_delta,
                "ndcg_delta": ndcg_delta,
                "prec_delta": prec_delta,
                "pop_delta": pop_delta,
                "classification": classification
            }
            print(f"--> Recall@20: {rec_20:.6f} ({rec_delta:+.2f}%)")
            print(f"--> NDCG@20: {ndcg_20:.6f} ({ndcg_delta:+.2f}%)")
            print(f"--> Precision@20: {prec_20:.6f} ({prec_delta:+.2f}%)")
            print(f"--> Popularity@20: {pop_20:.3f} ({pop_delta:+.2f}%)")
            print(f"--> Classification: {classification}")
            
            grid_results.append(run_result)

# Write results to /home/mohabmostafa/98-Downloads/Masters/checkpoints/phase3_results.txt in append mode
results_tracker = "/home/mohabmostafa/98-Downloads/Masters/checkpoints/phase3_results.txt"
with open(results_tracker, "a") as f:
    f.write("\n\n--- Combined Pipeline Grid Search (DAP + SGL + Hodge-Laplacian TSP + PPD Variant 4) ---\n")
    for r in grid_results:
        f.write(
            f"dataset={r['dataset']} | beta={r['beta']} | ego={r['ego']} | "
            f"Recall@20={r['recall']:.6f} ({r['rec_delta']:+.2f}%) | "
            f"NDCG@20={r['ndcg']:.6f} ({r['ndcg_delta']:+.2f}%) | "
            f"Precision@20={r['precision']:.6f} ({r['prec_delta']:+.2f}%) | "
            f"Popularity@20={r['popularity']:.3f} ({r['pop_delta']:+.2f}%) | "
            f"class={r['classification']}\n"
        )

print("\n[OK] Grid search evaluation completed and saved to phase3_results.txt.")
