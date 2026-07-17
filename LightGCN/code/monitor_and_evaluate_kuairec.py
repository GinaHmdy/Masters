# monitor_and_evaluate_kuairec.py
import os
import time
import subprocess
import re
import signal

python_path = "/home/mohabmostafa/miniconda3/envs/gMasters/bin/python"
code_dir = "/home/mohabmostafa/98-Downloads/Masters/LightGCN/code"
log_path = "/home/mohabmostafa/.gemini/antigravity-cli/brain/f54b0eed-26fb-491f-87f0-d28ae3d7e722/.system_generated/tasks/task-1957.log"
checkpoints_dir = os.path.join(code_dir, "checkpoints")
readme_path = "/home/mohabmostafa/98-Downloads/Masters/LightGCN/README.md"
pid = 79919
target_epoch = 35

# Canonical PPD v3 alone reference for KuaiRec
ref = {
    "recall": 0.092749,
    "ndcg": 0.212923,
    "precision": 0.169681,
    "popularity": 1216.544
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

print(f"Monitoring PID {pid} until epoch {target_epoch}...")

while True:
    # Check if process is still running
    try:
        os.kill(pid, 0)
    except OSError:
        print(f"Process PID {pid} is dead. Proceeding to evaluation...")
        break
        
    # Read the log file to check the epoch count
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                # Search for epoch lines like "EPOCH[35/1000]" or "[Valid] Epoch 35:"
                match = re.search(r"EPOCH\[(\d+)/1000\]", line)
                if match:
                    current_epoch = int(match.group(1))
                    if current_epoch >= target_epoch:
                        print(f"Reached target epoch {current_epoch} (>= {target_epoch}). Stopping PID {pid}...")
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except OSError:
                            pass
                        break
    time.sleep(10)

# Allow checkpoint file to finish writing
time.sleep(5)

# Verify checkpoint file exists
best_ckpt = os.path.join(checkpoints_dir, "dap-KuaiRec-2-256-best.pth.tar")
if not os.path.exists(best_ckpt):
    print(f"ERROR: Checkpoint {best_ckpt} does not exist!")
    exit(1)

print("Starting KuaiRec converged evaluations...")

evals = [
    {"beta": 1.0, "pop_limit_multiplier": 1.0},
    {"beta": 2.0, "pop_limit_multiplier": 1.0}
]

eval_results = []

for ev in evals:
    print(f"\n==================== Running KuaiRec Evaluation beta={ev['beta']} ====================")
    cmd = [
        python_path, "run_thesis.py",
        "--dataset", "KuaiRec",
        "--load", "1",
        "--model", "dap",
        "--layer", "2",
        "--recdim", "256",
        "--ckpt", best_ckpt,
        "--dap",
        "--tsp_hodge",
        "--ppd",
        "--ppd_variant", "4",
        "--beta", str(ev["beta"]),
        "--phi", "1.0",
        "--theta_tsp", "0.90",
        "--top_m_tsp", "15",
        "--k_tsp", "1",
        "--alpha_tsp", "0.1",
        "--fairness"
    ]
    
    print("Command:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=code_dir, capture_output=True, text=True)
    output = res.stdout
    print(output)
    
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
    
    eval_results.append({
        "beta": ev["beta"],
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

print("\n\n==================== KUAIREC CONVERGED MODEL RESULTS ====================")
print("| Beta | Recall@20 (Delta) | NDCG@20 (Delta) | Precision@20 (Delta) | Average Popularity@20 (Delta) | Classification |")
print("| :--- | :--- | :--- | :--- | :--- | :--- |")
for r in eval_results:
    print(f"| beta = {r['beta']} | {r['recall']:.6f} ({r['rec_delta']:+.2f}%) | {r['ndcg']:.6f} ({r['ndcg_delta']:+.2f}%) | {r['precision']:.6f} ({r['prec_delta']:+.2f}%) | {r['popularity']:.3f} ({r['pop_delta']:+.2f}%) | {r['classification']} |")

# Write results to README.md
with open(readme_path, "a") as readme:
    readme.write("\n\n## Master Thesis Experiments - Phase V (Canonical & Consolidated Pipeline)\n")
    readme.write("\n### Gowalla Seed-Variance Robustness (3 Seeds)\n")
    readme.write("| Config | Metric | Seed 2021 | Seed 2022 | Seed 2023 | Mean ± Std | Reference Baseline |\n")
    readme.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
    readme.write("| ego_uniform | Recall@20 | 0.149334 | 0.145398 | 0.147154 | 0.147295 ± 0.001610 | 0.121115 (+21.62%) |\n")
    readme.write("|             | NDCG@20 | 0.122958 | 0.120391 | 0.120680 | 0.121343 ± 0.001148 | 0.101116 (+20.00%) |\n")
    readme.write("|             | AvgPop@20 | 119.888 | 142.433 | 141.107 | 134.476 ± 10.330 | 253.862 (-47.03%) |\n")
    readme.write("| ego_0.6 | Recall@20 | 0.126205 | 0.133444 | 0.135720 | 0.131790 ± 0.004057 | 0.121115 (+8.81%) |\n")
    readme.write("|         | NDCG@20 | 0.100057 | 0.107050 | 0.108037 | 0.105048 ± 0.003552 | 0.101116 (+3.89%) |\n")
    readme.write("|         | AvgPop@20 | 87.601 | 101.850 | 101.036 | 96.829 ± 6.533 | 253.862 (-61.86%) |\n")
    
    readme.write("\n### Yahoo Ablation Results (Classification against Canonical PPD v3 alone)\n")
    readme.write("| Ablation | Recall@20 (Delta) | NDCG@20 (Delta) | Precision@20 (Delta) | Average Popularity@20 (Delta) | Classification |\n")
    readme.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
    readme.write("| Ablation 1: DAP + SGL + standard --tsp + PPD v4 | 0.049082 (-64.89%) | 0.024030 (-62.12%) | 0.004133 (-62.19%) | 114.510 (-81.43%) | D. TRADE-OFF/LOSS |\n")
    readme.write("| Ablation 2: DAP + SGL + --tsp_hodge + PPD v4 (last layer only) | 0.049208 (-64.80%) | 0.023940 (-62.26%) | 0.004133 (-62.19%) | 114.302 (-81.46%) | D. TRADE-OFF/LOSS |\n")
    readme.write("| Ablation 3: DAP + --tsp_hodge + PPD v4 (SGL disabled) | 0.085687 (-38.70%) | 0.038977 (-38.56%) | 0.007533 (-31.10%) | 121.678 (-80.27%) | D. TRADE-OFF/LOSS |\n")
    
    readme.write("\n### KuaiRec Converged Model Results (Classification against Canonical PPD v3 alone)\n")
    readme.write("| Beta | Recall@20 (Delta) | NDCG@20 (Delta) | Precision@20 (Delta) | Average Popularity@20 (Delta) | Classification |\n")
    readme.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
    for r in eval_results:
        readme.write(f"| beta = {r['beta']} | {r['recall']:.6f} ({r['rec_delta']:+.2f}%) | {r['ndcg']:.6f} ({r['ndcg_delta']:+.2f}%) | {r['precision']:.6f} ({r['prec_delta']:+.2f}%) | {r['popularity']:.3f} ({r['pop_delta']:+.2f}%) | {r['classification']} |\n")

print("README.md successfully updated with all findings!")
