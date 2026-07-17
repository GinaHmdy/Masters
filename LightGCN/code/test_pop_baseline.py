# test_pop_baseline.py
import sys
sys.argv += ['--dataset', 'KuaiRec']
import world
import register
from register import dataset
import numpy as np

item_counts = np.array(dataset.UserItemNet.sum(axis=0)).squeeze()
# Sort items by popularity in descending order
popular_sort = np.argsort(item_counts)[::-1]

recall_sum = 0
all_pops = []

for u, pos in dataset.testDict.items():
    train_pos = set(dataset.getUserPosItems([u])[0])
    # Get top 20 items not in training positive set
    recs = []
    for item in popular_sort:
        if item not in train_pos:
            recs.append(item)
            if len(recs) == 20:
                break
    hits = len(set(pos).intersection(recs))
    recall_sum += hits / len(pos)
    all_pops.extend(item_counts[recs])

print(f"MostPopular Recall@20: {recall_sum / len(dataset.testDict):.6f}")
print(f"MostPopular Average Popularity@20: {np.mean(all_pops):.4f}")
