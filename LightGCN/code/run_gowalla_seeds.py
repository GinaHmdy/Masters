# run_gowalla_seeds.py
import subprocess
import os
import re
import numpy as np

python_path = "/home/mohabmostafa/miniconda3/envs/gMasters/bin/python"
code_dir = "/home/mohabmostafa/98-Downloads/Masters/LightGCN/code"
checkpoints_dir = os.path.join(code_dir, "checkpoints")

seeds = [2021, 2022, 2023]
configs = [
    {"name": "ego_uniform", "ego_weight": "-1.0"},
    {"name": "ego_0.6", "ego_weight": "0.6"}
]

# Track results
# results_dict = {config_name: {metric: [vals]}}
results_dict = {
    "ego_uniform": {"recall": [], "ndcg": [], "precision": [], "popularity": []},
    "ego_0.6": {"recall": [], "ndcg": [], "precision": [], "popularity": []}
}

for seed in seeds:
    print(f"\n==================== Seed {seed}: Training DAP + SGL model ====================")
    # Temporary checkpoint path
    temp_ckpt = os.path.join(checkpoints_dir, f"dap-Gowalla-3-64.pth.tar")
    if os.path.exists(temp_ckpt):
        os.remove(temp_ckpt)
        
    train_cmd = [
        python_path, "main.py",
        "--dataset", "Gowalla",
        "--model", "dap",
        "--layer", "3",
        "--recdim", "64",
        "--epochs", "15",
        "--sgl",
        "--sgl_weight", "0.1",
        "--seed", str(seed)
    ]
    print("Training command:", " ".join(train_cmd))
    subprocess.run(train_cmd, cwd=code_dir, check=True)
    
    # Copy to a seed-specific path
    seed_ckpt = os.path.join(checkpoints_dir, f"dap-Gowalla-3-64-seed{seed}.pth.tar")
    import shutil
    shutil.copy(temp_ckpt, seed_ckpt)
    print(f"[OK] Saved seed checkpoint: {seed_ckpt}")
    
    # Evaluate the two configs on this checkpoint
    for cfg in configs:
        print(f"\nEvaluating Seed {seed} | Config {cfg['name']}")
        eval_cmd = [
            python_path, "run_thesis.py",
            "--dataset", "Gowalla",
            "--load", "1",
            "--model", "dap",
            "--layer", "3",
            "--recdim", "64",
            "--ckpt", seed_ckpt,
            "--dap",
            "--tsp_hodge",
            "--ppd",
            "--ppd_variant", "4",
            "--beta", "0.2",
            "--phi", "1.0",
            "--theta_tsp", "0.90",
            "--top_m_tsp", "15",
            "--k_tsp", "1",
            "--alpha_tsp", "0.1",
            "--ego_weight", cfg["ego_weight"],
            "--fairness"
        ]
        print("Command:", " ".join(eval_cmd))
        res = subprocess.run(eval_cmd, cwd=code_dir, capture_output=True, text=True)
        output = res.stdout
        
        # Parse metrics
        precision_match = re.search(r"'precision':\s*array\(\[(.*?)\]\)", output, re.S)
        recall_match = re.search(r"'recall':\s*array\(\[(.*?)\]\)", output, re.S)
        ndcg_match = re.search(r"'ndcg':\s*array\(\[(.*?)\]\)", output, re.S)
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
            
        results_dict[cfg["name"]]["recall"].append(rec_20)
        results_dict[cfg["name"]]["ndcg"].append(ndcg_20)
        results_dict[cfg["name"]]["precision"].append(prec_20)
        results_dict[cfg["name"]]["popularity"].append(pop_20)

print("\n\n==================== GOWALLA SEED VARIANCE RESULTS ====================")
for cfg_name, metrics in results_dict.items():
    print(f"\nConfig: {cfg_name}")
    for metric_name, vals in metrics.items():
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        print(f"  {metric_name.upper()}@20: {vals} -> Mean: {mean_val:.6f} | Std: {std_val:.6f}")
