# test_beta_sweep.py
import sys
import os
import torch
import numpy as np

sys.argv += [
    "--dataset", "KuaiRec",
    "--load", "1",
    "--model", "lgn",
    "--layer", "2",
    "--recdim", "256",
    "--ckpt", "checkpoints/canonical/lgn-KuaiRec-2-256.pth.tar",
    "--ppd",
    "--ppd_variant", "3",
    "--beta", "1.0",
    "--phi", "1.0"
]

import world
import utils
import register
from register import dataset
import Procedure
from utilities import compute_pop_bias_scores, project_and_remove, compute_d_pop_all_normalized
from run_thesis import calculate_average_popularity

model = register.MODELS['lgn'](world.config, dataset).to(world.device)
ckpt_path = "checkpoints/canonical/lgn-KuaiRec-2-256.pth.tar"

# We compute bias_scores, d_pop_u, d_pop_i once from the unmodified baseline model
state_dict = torch.load(ckpt_path, map_location=world.device)
model.load_state_dict(state_dict, strict=True)

user_emb_weight = torch.nn.functional.normalize(model.embedding_user.weight)
item_emb_weight = torch.nn.functional.normalize(model.embedding_item.weight)

bias_scores, _ = compute_pop_bias_scores(
    "../../data/KuaiRec",
    dataset._allPos,
    model,
    dataset,
    best_epoch=0,
    variant=3,
    beta=1.0,
    phi=1.0
)

d_pop_u, d_pop_i = compute_d_pop_all_normalized(
    user_emb_weight, item_emb_weight,
    len(user_emb_weight), len(item_emb_weight),
    bias_scores,
    dataset._allPos,
    phi=1.0
)

print("\n--- SWEEPING BETA IN PROJECT_AND_REMOVE ---")
for beta in [0.0, 0.1, 0.5, 1.0, 1.5, 2.0]:
    # Reload original weights
    model.load_state_dict(state_dict, strict=True)
    
    # Project and remove with specific beta
    user_emb_new = project_and_remove(user_emb_weight, d_pop_u)
    item_emb_new = project_and_remove(item_emb_weight, d_pop_i)
    
    with torch.no_grad():
        model.embedding_user.weight.copy_(user_emb_new)
        model.embedding_item.weight.copy_(item_emb_new)
        
    results = Procedure.Test(
        dataset, model, epoch=0, w=None,
        multicore=0, posthoc=0, all_updated_embeddings=None
    )
    avg_pop = calculate_average_popularity(model, dataset, topk=20, posthoc=0)
    print(f"Beta: {beta:.2f} | Recall@20: {results['recall'][3]:.6f} | AvgPop@20: {avg_pop:.4f}")
