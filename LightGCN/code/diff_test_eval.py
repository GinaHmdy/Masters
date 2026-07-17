# diff_test_eval.py
import sys
import os
import torch
import numpy as np
import scipy.sparse as sp

# Configure world arguments before importing other modules
sys.argv += [
    "--dataset", "KuaiRec",
    "--load", "1",
    "--model", "lgn",
    "--layer", "2",
    "--recdim", "256",
    "--ckpt", "checkpoints/canonical/lgn-KuaiRec-2-256.pth.tar",
    "--fairness"
]

import world
import utils
import register
from register import dataset
import Procedure
from utilities import compute_pop_bias_scores

print("\n--- STANDALONE CORRECTNESS AUDIT SCRIPT ---")
print("Loading model and checkpoint...")
model = register.MODELS['lgn'](world.config, dataset).to(world.device)
ckpt_path = "checkpoints/canonical/lgn-KuaiRec-2-256.pth.tar"
state_dict = torch.load(ckpt_path, map_location=world.device)
model.load_state_dict(state_dict, strict=True)
print("[OK] Model and checkpoint loaded strictly.")

# Evaluate baseline through Procedure.Test
print("\n[Baseline] Evaluating through Procedure.Test...")
results_base = Procedure.Test(
    dataset, model, epoch=0, w=None,
    multicore=0,
    posthoc=0,
    all_updated_embeddings=None
)
print("Baseline results:")
print(results_base)

# Evaluate baseline average popularity
from run_thesis import calculate_average_popularity
print("\n[Baseline] Evaluating through calculate_average_popularity...")
avg_pop_base = calculate_average_popularity(
    model, dataset, topk=20, posthoc=0, all_updated_embeddings=None
)
print(f"Baseline Average Popularity@20: {avg_pop_base:.4f}")

# Now reload checkpoint to run PPD
state_dict = torch.load(ckpt_path, map_location=world.device)
model.load_state_dict(state_dict, strict=True)

# Run post-hoc PPD bias calculation and zero-layer projection
print("\n[PPD v3] Running PPD Variant 3 bias estimation...")
bias_scores, all_updated_embeddings = compute_pop_bias_scores(
    "../../data/KuaiRec",
    dataset._allPos,
    model,
    dataset,
    best_epoch=0,
    variant=3,
    beta=2.00,
    phi=1.0
)

# Evaluate PPD v3 through Procedure.Test
print("\n[PPD v3] Evaluating through Procedure.Test...")
results_ppd = Procedure.Test(
    dataset, model, epoch=0, w=None,
    multicore=0,
    posthoc=0,
    all_updated_embeddings=None
)
print("PPD v3 results:")
print(results_ppd)

# Evaluate PPD v3 average popularity
print("\n[PPD v3] Evaluating through calculate_average_popularity...")
avg_pop_ppd = calculate_average_popularity(
    model, dataset, topk=20, posthoc=0, all_updated_embeddings=None
)
print(f"PPD v3 Average Popularity@20: {avg_pop_ppd:.4f}")
