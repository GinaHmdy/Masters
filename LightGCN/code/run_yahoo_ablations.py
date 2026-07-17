# run_yahoo_ablations.py
import subprocess
import os
import re

python_path = "/home/mohabmostafa/miniconda3/envs/gMasters/bin/python"
code_dir = "/home/mohabmostafa/98-Downloads/Masters/LightGCN/code"
checkpoints_dir = os.path.join(code_dir, "checkpoints")

# Canonical PPD v3 alone reference for Yahoo
ref = {
    "recall": 0.139781,
    "ndcg": 0.063436,
    "precision": 0.010932,
    "popularity": 616.667
}

def classify_run(rec, ndcg, prec, pop, ref):
    is_dominant = (rec >= ref["recall"] and ndcg >= ref["ndcg"] and prec >= ref["precision"] and pop < ref["popularity"])
    
    rec_d = (rec - ref["recall"]) / ref["recall"]
    ndcg_d = (ndcg - ref["ndcg"]) / ref["ndcg"]
    prec_d = (prec - ref["precision"]) / ref["precision"]
    pop_d = (pop - ref["popularity"]) / ref["popularity"]
    
    is_accuracy_win = (rec_d >= 0.05 and ndcg_d >= 0.05 and prec_d >= 0.05 and pop_d <= 0.01)
    is_fairness_win = (pop_d <= -0.20 and rec_d >= -0.01 and ndcg_d >= -0.01 and prec_d >= -0.01)
    
    if is_dominant:
        return "A. DOMINANT"
    elif is_accuracy_win:
        return "B. ACCURACY-WIN"
    elif is_fairness_win:
        return "C. FAIRNESS-WIN"
    else:
        return "D. TRADE-OFF/LOSS"

ablations = [
    {
        "id": 1,
        "name": "DAP + SGL + standard --tsp + PPD v4",
        "ckpt": os.path.join(checkpoints_dir, "dap-Yahoo-2-256-sgl.pth.tar"),
        "flags": ["--dap", "--tsp", "--ppd", "--ppd_variant", "4", "--beta", "0.07", "--phi", "1.0"]
    },
    {
        "id": 2,
        "name": "DAP + SGL + --tsp_hodge + PPD v4 (last layer only)",
        "ckpt": os.path.join(checkpoints_dir, "dap-Yahoo-2-256-sgl.pth.tar"),
        "flags": ["--dap", "--tsp_hodge", "--ppd", "--ppd_variant", "4", "--ppd_v4_layers", "last", "--beta", "0.07", "--phi", "1.0"]
    },
    {
        "id": 3,
        "name": "DAP + --tsp_hodge + PPD v4 (SGL disabled)",
        "ckpt": os.path.join(checkpoints_dir, "dap-Yahoo-2-256-nosgl.pth.tar"),
        "flags": ["--dap", "--tsp_hodge", "--ppd", "--ppd_variant", "4", "--beta", "0.07", "--phi", "1.0"]
    }
]

ablation_results = []

for ab in ablations:
    print(f"\n==================== Running Ablation {ab['id']}: {ab['name']} ====================")
    cmd = [
        python_path, "run_thesis.py",
        "--dataset", "Yahoo",
        "--load", "1",
        "--model", "dap",
        "--layer", "2",
        "--recdim", "256",
        "--ckpt", ab["ckpt"],
        "--fairness"
    ] + ab["flags"]
    
    print("Command:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=code_dir, capture_output=True, text=True)
    output = res.stdout
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
        
    rec_delta = (rec_20 - ref["recall"]) / ref["recall"] * 100
    ndcg_delta = (ndcg_20 - ref["ndcg"]) / ref["ndcg"] * 100
    prec_delta = (prec_20 - ref["precision"]) / ref["precision"] * 100
    pop_delta = (pop_20 - ref["popularity"]) / ref["popularity"] * 100
    
    classification = classify_run(rec_20, ndcg_20, prec_20, pop_20, ref)
    
    ablation_results.append({
        "id": ab["id"],
        "name": ab["name"],
        "recall": rec_20,
        "ndcg": ndcg_20,
        "precision": prec_20,
        "popularity": pop_20,
        "rec_delta": rec_delta,
        "ndcg_delta": ndcg_delta,
        "prec_delta": prec_delta,
        "pop_delta": pop_delta,
        "classification": classification
    })

print("\n\n==================== YAHOO ABLATION RESULTS ====================")
print("| Ablation | Recall@20 (Delta) | NDCG@20 (Delta) | Precision@20 (Delta) | Average Popularity@20 (Delta) | Classification |")
print("| :--- | :--- | :--- | :--- | :--- | :--- |")
for r in ablation_results:
    print(f"| Ablation {r['id']}: {r['name']} | {r['recall']:.6f} ({r['rec_delta']:+.2f}%) | {r['ndcg']:.6f} ({r['ndcg_delta']:+.2f}%) | {r['precision']:.6f} ({r['prec_delta']:+.2f}%) | {r['popularity']:.3f} ({r['pop_delta']:+.2f}%) | {r['classification']} |")
