# run_kuairec_sweeps.py
import subprocess
import os
import re
import sys

python_path = "/home/mohabmostafa/miniconda3/envs/gMasters/bin/python"
code_dir = "/home/mohabmostafa/98-Downloads/Masters/LightGCN/code"
ckpt_path = os.path.join(code_dir, "checkpoints/canonical/lgn-KuaiRec-2-256.pth.tar")

commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=code_dir).decode("utf-8").strip()

# References (from verified Phase VII canonical baselines)
ref_v3 = {
    "name": "PPD v3 alone (Ref)",
    "recall": 0.006518,
    "ndcg": 0.005389,
    "precision": 0.006418,
    "popularity": 564.687
}

ref_v4 = {
    "name": "PPD v4 alone (Ref)",
    "recall": 0.027168,
    "ndcg": 0.033133,
    "precision": 0.032199,
    "popularity": 690.297
}

def classify_run(rec, ndcg, prec, pop):
    # DOMINANT: beats both v3 and v4 on every accuracy metric AND has lower popularity than both
    beats_accuracy = (rec >= ref_v3["recall"] and rec >= ref_v4["recall"] and
                      ndcg >= ref_v3["ndcg"] and ndcg >= ref_v4["ndcg"] and
                      prec >= ref_v3["precision"] and prec >= ref_v4["precision"])
    lower_popularity = (pop < ref_v3["popularity"] and pop < ref_v4["popularity"])
    
    if beats_accuracy and lower_popularity:
        return "A. DOMINANT"
        
    # PARTIAL: beats one of v3/v4 fully but not the other
    # To "fully beat" a reference configuration, it must have accuracy >= reference AND popularity < reference
    beats_v3 = (rec >= ref_v3["recall"] and ndcg >= ref_v3["ndcg"] and prec >= ref_v3["precision"] and pop < ref_v3["popularity"])
    beats_v4 = (rec >= ref_v4["recall"] and ndcg >= ref_v4["ndcg"] and prec >= ref_v4["precision"] and pop < ref_v4["popularity"])
    
    if beats_v3 or beats_v4:
        return "B. PARTIAL"
        
    return "C. NO IMPROVEMENT"

sweeps = [
    # Variant 1: Partial-layer PPD v4 (apply only at last layer)
    {
        "variant_id": 1,
        "name": "Partial-layer v4 (last layer only, beta=1.0)",
        "flags": ["--ppd_variant", "4", "--ppd_v4_layers", "last", "--beta", "1.0", "--phi", "1.0"]
    },
    {
        "variant_id": 1,
        "name": "Partial-layer v4 (last layer only, beta=2.0)",
        "flags": ["--ppd_variant", "4", "--ppd_v4_layers", "last", "--beta", "2.0", "--phi", "1.0"]
    },
    # Variant 2: Beta-scheduled PPD v4
    {
        "variant_id": 2,
        "name": "Beta-scheduled v4 (all layers, ramp 0.2 -> 1.0)",
        "flags": ["--ppd_variant", "4", "--ppd_v4_layers", "all", "--ppd_beta_schedule", "0.2,1.0", "--beta", "1.0", "--phi", "1.0"]
    },
    {
        "variant_id": 2,
        "name": "Beta-scheduled v4 (all layers, ramp 0.5 -> 2.0)",
        "flags": ["--ppd_variant", "4", "--ppd_v4_layers", "all", "--ppd_beta_schedule", "0.5,2.0", "--beta", "2.0", "--phi", "1.0"]
    },
    {
        "variant_id": 2,
        "name": "Beta-scheduled v4 (all layers, ramp 1.0 -> 2.0)",
        "flags": ["--ppd_variant", "4", "--ppd_v4_layers", "all", "--ppd_beta_schedule", "1.0,2.0", "--beta", "2.0", "--phi", "1.0"]
    },
    # Variant 3: Sequential v3-then-v4
    {
        "variant_id": 3,
        "name": "Sequential v3-then-v4 (v3 beta=1.0, v4 beta=0.5 last layer)",
        "flags": ["--ppd_variant", "4", "--ppd_seq_v3_v4", "--ppd_seq_beta_v3", "1.0", "--ppd_v4_layers", "last", "--beta", "0.5", "--phi", "1.0"]
    },
    {
        "variant_id": 3,
        "name": "Sequential v3-then-v4 (v3 beta=2.0, v4 beta=1.0 last layer)",
        "flags": ["--ppd_variant", "4", "--ppd_seq_v3_v4", "--ppd_seq_beta_v3", "2.0", "--ppd_v4_layers", "last", "--beta", "1.0", "--phi", "1.0"]
    },
    {
        "variant_id": 3,
        "name": "Sequential v3-then-v4 (v3 beta=1.0, v4 beta=1.0 last layer)",
        "flags": ["--ppd_variant", "4", "--ppd_seq_v3_v4", "--ppd_seq_beta_v3", "1.0", "--ppd_v4_layers", "last", "--beta", "1.0", "--phi", "1.0"]
    }
]

sweep_results = []

for idx, sw in enumerate(sweeps):
    print(f"\n==================== Running Sweep {idx+1}/{len(sweeps)}: {sw['name']} ====================")
    cmd = [
        python_path, "run_thesis.py",
        "--dataset", "KuaiRec",
        "--load", "1",
        "--model", "lgn",
        "--layer", "2",
        "--recdim", "256",
        "--ckpt", ckpt_path,
        "--fairness",
        "--ppd"
    ] + sw["flags"]
    
    print("Command:", " ".join(cmd))
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
    
    classification = classify_run(rec_20, ndcg_20, prec_20, pop_20)
    
    sweep_results.append({
        "variant_id": sw["variant_id"],
        "name": sw["name"],
        "recall": rec_20,
        "ndcg": ndcg_20,
        "precision": prec_20,
        "popularity": pop_20,
        "classification": classification,
        "commit": commit_hash
    })

print("\n\n==================== KUAIREC VARIANT SWEEP RESULTS ====================")
print("| Configuration | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 | Classification | Commit |")
print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
# References first
print(f"| {ref_v3['name']} | {ref_v3['recall']:.6f} | {ref_v3['ndcg']:.6f} | {ref_v3['precision']:.6f} | {ref_v3['popularity']:.3f} | N/A | - |")
print(f"| {ref_v4['name']} | {ref_v4['recall']:.6f} | {ref_v4['ndcg']:.6f} | {ref_v4['precision']:.6f} | {ref_v4['popularity']:.3f} | N/A | - |")
for r in sweep_results:
    print(f"| {r['name']} | {r['recall']:.6f} | {r['ndcg']:.6f} | {r['precision']:.6f} | {r['popularity']:.3f} | {r['classification']} | {r['commit']} |")
