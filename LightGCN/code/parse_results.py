import os
import re

log_dir = "logs"

pattern = re.compile(
    r"\{'precision': array\(\[(.*?)\]\), 'recall': array\(\[(.*?)\]\), 'ndcg': array\(\[(.*?)\]\), 'hr': array\(\[(.*?)\]\)\}"
)

results = []

for file in os.listdir(log_dir):
    path = os.path.join(log_dir, file)

    if not os.path.isfile(path):
        continue

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    matches = pattern.findall(text)

    if not matches:
        continue

    # take LAST evaluation block
    precision, recall, ndcg, hr = matches[-1]

    alpha = re.search(r"_a([0-9.]+)", file)
    theta = re.search(r"_theta([0-9.]+)", file)
    k = re.search(r"_k([0-9]+)", file)
    m = re.search(r"_m([0-9]+)", file)
    beta = re.search(r"_b([0-9.]+)", file)
    phi = re.search(r"_phi([0-9.]+)", file)

    results.append({
        "file": file,
        "precision": float(precision),
        "recall": float(recall),
        "ndcg": float(ndcg),
        "hr": float(hr),
        "alpha": alpha.group(1) if alpha else None,
        "theta": theta.group(1) if theta else None,
        "k": k.group(1) if k else None,
        "m": m.group(1) if m else None,
        "beta": beta.group(1) if beta else None,
        "phi": phi.group(1) if phi else None,
    })

# sort by recall descending
results.sort(key=lambda x: x["recall"], reverse=True)

print("\nTop results by Recall@20\n")

for r in results[:20]:
    print(
        f"{r['file']} | "
        f"P={r['precision']:.6f} "
        f"R={r['recall']:.6f} "
        f"NDCG={r['ndcg']:.6f} "
        f"HR={r['hr']:.6f} "
        f"alpha={r['alpha']} "
        f"theta={r['theta']} "
        f"k={r['k']} "
        f"m={r['m']} "
        f"beta={r['beta']} "
        f"phi={r['phi']}"
    )