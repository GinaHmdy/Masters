# evaluate_canonical.py
import subprocess
import os
import re

python_path = "/home/mohabmostafa/miniconda3/envs/gMasters/bin/python"
code_dir = "/home/mohabmostafa/98-Downloads/Masters/LightGCN/code"
canonical_dir = os.path.join(code_dir, "checkpoints/canonical")

datasets_config = {
    "Gowalla": {
        "layer": 3,
        "recdim": 64,
        "beta": 0.20,
        "ckpt": os.path.join(canonical_dir, "lgn-Gowalla-3-64.pth.tar")
    },
    "KuaiRec": {
        "layer": 2,
        "recdim": 256,
        "beta": 2.00,
        "ckpt": os.path.join(canonical_dir, "lgn-KuaiRec-2-256.pth.tar")
    },
    "Yahoo": {
        "layer": 2,
        "recdim": 256,
        "beta": 0.07,
        "ckpt": os.path.join(canonical_dir, "lgn-Yahoo-2-256.pth.tar")
    }
}

configs = {
    "Baseline": [],
    "TSP alone": ["--tsp", "--alpha_tsp", "0.1", "--theta_tsp", "0.90", "--top_m_tsp", "15", "--k_tsp", "1"],
    "PPD v3 alone": ["--ppd", "--ppd_variant", "3"],
    "PPD v4 alone": ["--ppd", "--ppd_variant", "4"]
}

results = []

for dataset, info in datasets_config.items():
    for name, flags in configs.items():
        print(f"\n==================== Evaluating {dataset} - {name} ====================")
        
        # Build command
        cmd = [
            python_path, "run_thesis.py",
            "--dataset", dataset,
            "--load", "1",
            "--model", "lgn",
            "--layer", str(info["layer"]),
            "--recdim", str(info["recdim"]),
            "--ckpt", info["ckpt"],
            "--fairness"
        ] + flags
        
        # Add beta if ppd is used
        if "--ppd" in flags:
            cmd += ["--beta", str(info["beta"]), "--phi", "1.0"]
            
        print("Command:", " ".join(cmd))
        
        res = subprocess.run(cmd, cwd=code_dir, capture_output=True, text=True)
        output = res.stdout
        
        # Print output to console for logging/verification
        print(output)
        if res.stderr:
            print("[STDERR]", res.stderr)
            
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
            
        results.append({
            "dataset": dataset,
            "method": name,
            "recall": rec_20,
            "ndcg": ndcg_20,
            "precision": prec_20,
            "popularity": pop_20
        })

print("\n\n==================== CANONICAL REFERENCE TABLE ====================")
print("| Dataset | Method | Recall@20 | NDCG@20 | Precision@20 | Average Popularity@20 |")
print("| :--- | :--- | :--- | :--- | :--- | :--- |")
for r in results:
    print(f"| {r['dataset']} | {r['method']} | {r['recall']:.6f} | {r['ndcg']:.6f} | {r['precision']:.6f} | {r['popularity']:.3f} |")

# Also write to file
with open("canonical_reference.txt", "w") as f:
    f.write("Canonical Reference Table:\n")
    f.write("| Dataset | Method | Recall@20 | NDCG@20 | Precision@20 | Average Popularity@20 |\n")
    f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
    for r in results:
        f.write(f"| {r['dataset']} | {r['method']} | {r['recall']:.6f} | {r['ndcg']:.6f} | {r['precision']:.6f} | {r['popularity']:.3f} |\n")
