# inspect_recs.py
import sys
import os
import torch
import numpy as np

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

model = register.MODELS['lgn'](world.config, dataset).to(world.device)
ckpt_path = "checkpoints/canonical/lgn-KuaiRec-2-256.pth.tar"
state_dict = torch.load(ckpt_path, map_location=world.device)
model.load_state_dict(state_dict, strict=True)

item_counts = np.array(dataset.UserItemNet.sum(axis=0)).squeeze()

# 1. Baseline recommendations for user 0
model.eval()
with torch.no_grad():
    rating_base = model.getUsersRating(torch.tensor([0], device=world.device))
    # Exclude training items
    train_items_0 = dataset.getUserPosItems([0])[0]
    rating_base[0, train_items_0] = -1e9
    _, top_items_base = torch.topk(rating_base, 20, dim=1)
    top_items_base = top_items_base.cpu().squeeze().numpy()

print("Baseline Top 20 Recommendations for User 0:")
for item in top_items_base:
    print(f"Item ID: {item}, Train Popularity: {item_counts[item]}")

# 2. Run PPD Variant 3
compute_pop_bias_scores(
    "../../data/KuaiRec",
    dataset._allPos,
    model,
    dataset,
    best_epoch=0,
    variant=3,
    beta=2.00,
    phi=1.0
)

# 3. PPD recommendations for user 0
with torch.no_grad():
    rating_ppd = model.getUsersRating(torch.tensor([0], device=world.device))
    rating_ppd[0, train_items_0] = -1e9
    _, top_items_ppd = torch.topk(rating_ppd, 20, dim=1)
    top_items_ppd = top_items_ppd.cpu().squeeze().numpy()

print("\nPPD v3 Top 20 Recommendations for User 0:")
for item in top_items_ppd:
    print(f"Item ID: {item}, Train Popularity: {item_counts[item]}")
