# run_component_search.py
import subprocess
import os
import json
import re
import sys

python_path = "/home/mohabmostafa/miniconda3/envs/gMasters/bin/python"
code_dir = "/home/mohabmostafa/98-Downloads/Masters/LightGCN/code"
checkpoints_dir = os.path.join(code_dir, "checkpoints")

commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=code_dir).decode("utf-8").strip()

# ---------------------------------------------------------------------------
# Baselines are read from results/*.json (3-seed means, current protocol).
# They are NOT hardcoded: the superseded values (Gowalla 0.121115 etc.) came
# from checkpoints/canonical/, which is a weaker LightGCN than the one the
# current protocol trains. Comparing against those inflated every margin.
# ---------------------------------------------------------------------------
def _load_baseline(dataset):
    path = os.path.join(code_dir, "results", "baselines_%s.json" % dataset.lower())
    with open(path) as f:
        blob = json.load(f)
    lgn = blob["LightGCN"]["mean"]
    return {
        "baseline_recall": float(lgn["recall@20"]),
        "baseline_ndcg": float(lgn["ndcg@20"]),
        "baseline_prec": float(lgn["precision@20"]),
        "baseline_pop": float(lgn["avgpop@20"]),
    }

datasets_config = {
    "Gowalla": {
        "layer": "3",
        "recdim": "64",
        "beta": "0.2",
        "checkpoints": {
            "lgn": os.path.join(checkpoints_dir, "lgn-Gowalla-3-64-best.pth.tar"),
            "dap": os.path.join(checkpoints_dir, "dap-Gowalla-3-64.pth.tar"),
            "dap-sgl": os.path.join(checkpoints_dir, "dap-Gowalla-3-64-sgl.pth.tar")
        }
    },
    "KuaiRec": {
        "layer": "2",
        "recdim": "256",
        "beta": "2.0",
        "checkpoints": {
            "lgn": os.path.join(checkpoints_dir, "lgn-KuaiRec-2-256-best.pth.tar"),
            "dap": os.path.join(checkpoints_dir, "dap-KuaiRec-2-256.pth.tar"),
            "dap-sgl": os.path.join(checkpoints_dir, "dap-KuaiRec-2-256-sgl.pth.tar")
        }
    },
    "Yahoo": {
        "layer": "2",
        "recdim": "256",
        "beta": "0.07",
        "checkpoints": {
            "lgn": os.path.join(checkpoints_dir, "lgn-Yahoo-2-256-best.pth.tar"),
            "dap": os.path.join(checkpoints_dir, "dap-Yahoo-2-256.pth.tar"),
            "dap-sgl": os.path.join(checkpoints_dir, "dap-Yahoo-2-256-sgl.pth.tar")
        }
    }
}
for _ds in datasets_config:
    datasets_config[_ds].update(_load_baseline(_ds))

def classify_run(rec, ndcg, prec, pop, ref):
    # Objective 1: Dominant (Accuracy >= baseline, Popularity < baseline)
    is_dominant = (rec >= ref["baseline_recall"] and ndcg >= ref["baseline_ndcg"] and prec >= ref["baseline_prec"] and pop < ref["baseline_pop"])
    
    rec_d = (rec - ref["baseline_recall"]) / ref["baseline_recall"]
    ndcg_d = (ndcg - ref["baseline_ndcg"]) / ref["baseline_ndcg"]
    prec_d = (prec - ref["baseline_prec"]) / ref["baseline_prec"]
    pop_d = (pop - ref["baseline_pop"]) / ref["baseline_pop"]
    
    # Objective 2 Options:
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

results_by_dataset = {}

for dataset_name, d_cfg in datasets_config.items():
    print(f"\n==========================================")
    print(f"Starting Component Inclusion Search for {dataset_name}")
    print(f"==========================================")
    
    results_by_dataset[dataset_name] = []
    
    for model_key, ckpt_path in d_cfg["checkpoints"].items():
        if not os.path.exists(ckpt_path):
            print(f"WARNING: Checkpoint {ckpt_path} does not exist. Skipping model {model_key}...")
            continue
            
        dap_on = "on" if "dap" in model_key else "off"
        sgl_on = "on" if "sgl" in model_key else "off"
        
        for tsp in ["off", "standard", "hodge"]:
            for ppd in ["off", "v3", "v4"]:
                # Build command flags
                cmd_flags = []
                
                # DAP flag
                if dap_on == "on":
                    cmd_flags.append("--dap")
                    
                # TSP flags
                if tsp == "standard":
                    cmd_flags += ["--tsp", "--alpha_tsp", "0.1", "--theta_tsp", "0.90", "--top_m_tsp", "15", "--k_tsp", "1"]
                elif tsp == "hodge":
                    cmd_flags += ["--tsp_hodge", "--alpha_tsp", "0.1", "--theta_tsp", "0.90", "--top_m_tsp", "15", "--k_tsp", "1"]
                    
                # PPD flags
                if ppd == "v3":
                    cmd_flags += ["--ppd", "--ppd_variant", "3", "--beta", d_cfg["beta"], "--phi", "1.0"]
                elif ppd == "v4":
                    cmd_flags += ["--ppd", "--ppd_variant", "4", "--beta", d_cfg["beta"], "--phi", "1.0"]
                    
                cmd = [
                    python_path, "run_thesis.py",
                    "--dataset", dataset_name,
                    "--load", "1",
                    "--model", "dap" if dap_on == "on" else "lgn",
                    "--layer", d_cfg["layer"],
                    "--recdim", d_cfg["recdim"],
                    "--ckpt", ckpt_path,
                    "--fairness"
                ] + cmd_flags
                
                print(f"Evaluating: Model={model_key}, TSP={tsp}, PPD={ppd} ...")
                res = subprocess.run(cmd, cwd=code_dir, capture_output=True, text=True)
                output = res.stdout
                
                precision_match = re.search(r"'precision':\s*array\(\[(.*?)\]\)", output)
                recall_match = re.search(r"'recall':\s*array\(\[(.*?)\]\)", output)
                ndcg_match = re.search(r"'ndcg':\s*array\(\[(.*?)\]\)", output)
                popularity_match = re.search(r"Average Popularity@20:\s*([\d\.]+)", output)
                
                if not (precision_match and recall_match and ndcg_match and popularity_match):
                    print(f"ERROR: Failed to parse outputs for command: {' '.join(cmd)}")
                    if res.stderr:
                        print("[STDERR]", res.stderr)
                    continue
                    
                prec_20 = [float(x) for x in re.split(r'\s+|,', precision_match.group(1).replace('\n', ' ').strip()) if x][3]
                rec_20 = [float(x) for x in re.split(r'\s+|,', recall_match.group(1).replace('\n', ' ').strip()) if x][3]
                ndcg_20 = [float(x) for x in re.split(r'\s+|,', ndcg_match.group(1).replace('\n', ' ').strip()) if x][3]
                pop_20 = float(popularity_match.group(1))
                
                classification = classify_run(rec_20, ndcg_20, prec_20, pop_20, d_cfg)
                
                results_by_dataset[dataset_name].append({
                    "model": model_key,
                    "dap": dap_on,
                    "sgl": sgl_on,
                    "tsp": tsp,
                    "ppd": ppd,
                    "recall": rec_20,
                    "ndcg": ndcg_20,
                    "precision": prec_20,
                    "popularity": pop_20,
                    "classification": classification,
                    "commit": commit_hash
                })

print("\n\n==================== COMPONENT INCLUSION SEARCH RESULTS ====================")
for dataset_name, results in results_by_dataset.items():
    print(f"\n### {dataset_name} Search Results")
    print("| Model | DAP | SGL | TSP | PPD | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Classification | Commit |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        print(f"| {r['model']} | {r['dap']} | {r['sgl']} | {r['tsp']} | {r['ppd']} | {r['recall']:.6f} | {r['ndcg']:.6f} | {r['precision']:.6f} | {r['popularity']:.3f} | {r['classification']} | {r['commit']} |")
