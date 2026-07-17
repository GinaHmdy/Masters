# diff_test_eval.py
import sys
import os
import torch
import numpy as np
import scipy.sparse as sp

# Set paths
sys.path.append("code")
import world
# Configure world arguments before importing other modules
sys.argv += [
    "--dataset", "KuaiRec",
    "--load", "1",
    "--model", "lgn",
    "--layer", "2",
    "--recdim", "256",
    "--ckpt", "code/checkpoints/canonical/lgn-KuaiRec-2-256.pth.tar",
    "--ppd",
    "--ppd_variant", "3",
    "--beta", "2.00",
    "--phi", "1.0",
    "--fairness"
]

import utils
import register
from register import dataset
import Procedure
from utilities import compute_pop_bias_scores

print("\n--- STANDALONE CORRECTNESS AUDIT SCRIPT ---")
print("Loading model and checkpoint...")
model = register.MODELS['lgn'](world.config, dataset).to(world.device)
ckpt_path = "code/checkpoints/canonical/lgn-KuaiRec-2-256.pth.tar"
state_dict = torch.load(ckpt_path, map_location=world.device)
model.load_state_dict(state_dict, strict=True)
print("[OK] Model and checkpoint loaded strictly.")

# 1. Inspect number of candidates and test data
test_dict = dataset.testDict
print(f"Number of test users: {len(test_dict)}")
user_ids = list(test_dict.keys())
print(f"Sample user 0 test items: {test_dict[user_ids[0]]}")
train_items_0 = dataset.getUserPosItems([user_ids[0]])[0]
print(f"Sample user 0 train items (to be excluded): {train_items_0}")
print(f"Intersection (leakage check): {set(test_dict[user_ids[0]]).intersection(set(train_items_0))}")

# 2. Run post-hoc PPD bias calculation and zero-layer projection
print("\nRunning PPD Variant 3 bias estimation...")
bias_scores, all_updated_embeddings = compute_pop_bias_scores(
    "data/KuaiRec",
    dataset._allPos,
    model,
    dataset,
    best_epoch=0,
    variant=3,
    beta=2.00,
    phi=1.0
)

# 3. Inspect items' popularity scores in UserItemNet
item_counts = np.array(dataset.UserItemNet.sum(axis=0)).squeeze()
print(f"\nItem popularity statistics:")
print(f"Max train popularity: {item_counts.max()}")
print(f"Min train popularity: {item_counts.min()}")
print(f"Mean train popularity: {item_counts.mean():.4f}")

# 4. Evaluate through Procedure.Test (post-Phase-V path)
print("\nEvaluating through Procedure.Test...")
results = Procedure.Test(
    dataset, model, epoch=0, w=None,
    multicore=0,
    posthoc=0,
    all_updated_embeddings=None
)
print("Procedure.Test results:")
print(results)

# 5. Evaluate through calculate_average_popularity
from run_thesis import calculate_average_popularity
print("\nEvaluating through calculate_average_popularity...")
avg_pop = calculate_average_popularity(
    model, dataset, topk=20, posthoc=0, all_updated_embeddings=None
)
print(f"Average Popularity@20: {avg_pop:.4f}")
