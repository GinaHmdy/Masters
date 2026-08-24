import world
import utils
from world import cprint

import torch
from tensorboardX import SummaryWriter
import time
import Procedure
import copy
from os.path import join
import numpy as np

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")

if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=world.device), strict=False)
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

Neg_k = 1

# init tensorboard
BUDGET = 20  # early stopping patience epochs

if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

def compute_val_bpr_loss(dataset, Recmodel):
    Recmodel.eval()
    with torch.no_grad():
        val_users = dataset.validUser
        val_pos = dataset.validItem
        val_neg = []
        for u in val_users:
            pos_for_u = dataset.allPos[u]
            if len(pos_for_u) == 0:
                val_neg.append(0)
                continue
            while True:
                neg = np.random.randint(0, dataset.m_items)
                if neg not in pos_for_u:
                    break
            val_neg.append(neg)
        val_users_t = torch.tensor(val_users, dtype=torch.long, device=world.device)
        val_pos_t = torch.tensor(val_pos, dtype=torch.long, device=world.device)
        val_neg_t = torch.tensor(val_neg, dtype=torch.long, device=world.device)
        
        total_loss = 0.0
        batch_size = 2048
        # Get propagated embeddings once for the validation batch
        all_users, all_items = Recmodel.computer()
        
        for i in range(0, len(val_users), batch_size):
            u_b = val_users_t[i:i+batch_size]
            p_b = val_pos_t[i:i+batch_size]
            n_b = val_neg_t[i:i+batch_size]
            
            users_emb = all_users[u_b]
            pos_emb = all_items[p_b]
            neg_emb = all_items[n_b]
            
            # Normalize embeddings to unit length for scale-invariant validation loss
            users_emb = torch.nn.functional.normalize(users_emb, p=2, dim=1)
            pos_emb = torch.nn.functional.normalize(pos_emb, p=2, dim=1)
            neg_emb = torch.nn.functional.normalize(neg_emb, p=2, dim=1)
            
            pos_scores = torch.sum(users_emb * pos_emb, dim=1)
            neg_scores = torch.sum(users_emb * neg_emb, dim=1)
            loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
            total_loss += loss.item() * len(u_b)
        return total_loss / len(val_users)

is_ppd_v6 = (world.model_name == 'ppd_v6') or (world.config.get('method', 'lightgcn') == 'ppd_v6')

try:
    patience = BUDGET
    best_state_dict = None
    best_val_loss = float('inf')
    best_val_hv = -1.0
    best_results_valid = {}
    best_results_test = {}
    best_epoch = -1

    if is_ppd_v6:
        print("[main.py] Dispatching to PPD v6 / ASTD-PD training procedure with Pareto Hypervolume validation.")
        import utilities
        from dataloader import StratifiedPopularityNegativeSampler

        optimizer = torch.optim.Adam(Recmodel.parameters(), lr=world.config['lr'])
        neg_sampler = StratifiedPopularityNegativeSampler(
            Recmodel.pop_store.item_popularity_rank,
            Recmodel.pop_store.item_popularity_decile,
            dataset.allPos,
            dataset.m_items,
            uniform_ratio=world.config.get('astd_neg_uniform_ratio', 0.4),
            matched_ratio=world.config.get('astd_neg_matched_ratio', 0.4),
            hard_ratio=world.config.get('astd_neg_hard_ratio', 0.2),
        )

        warmup_epochs = min(10, world.TRAIN_epochs // 2) if not world.config.get('smoke_test', False) else 1

        for epoch in range(world.TRAIN_epochs):
            output_info = Procedure.train_ppd_v6(
                dataset, Recmodel, optimizer, epoch, neg_sampler=neg_sampler, w=w
            )
            print(f"EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_info}")

            # Validation evaluation
            val_results = Procedure.Valid(dataset, Recmodel, epoch, w, multicore=world.config["multicore"])
            val_pop = Procedure.calculate_average_popularity(Recmodel, dataset, topk=20)
            val_rec = val_results['recall'][3] if len(val_results['recall']) > 3 else val_results['recall'][-1]
            val_ndcg = val_results['ndcg'][3] if len(val_results['ndcg']) > 3 else val_results['ndcg'][-1]
            val_prec = val_results['precision'][3] if len(val_results['precision']) > 3 else val_results['precision'][-1]

            val_metrics = {
                'recall@20': val_rec,
                'ndcg@20': val_ndcg,
                'precision@20': val_prec,
                'avgpop@20': val_pop
            }
            val_hv = utilities.compute_pareto_hypervolume(val_metrics)
            print(f"[Valid PPDv6] Epoch {epoch}: Recall@20={val_rec:.6f}, NDCG@20={val_ndcg:.6f}, Prec@20={val_prec:.6f}, AvgPop@20={val_pop:.2f} -> Hypervolume={val_hv:.8f}")

            if epoch < warmup_epochs:
                best_val_hv = val_hv
                best_epoch = epoch
                best_results_valid = val_metrics
                best_state_dict = copy.deepcopy(Recmodel.state_dict())
                torch.save(Recmodel.state_dict(), weight_file)
                best_tag_file = weight_file.replace(".pth.tar", "-best.pth.tar")
                torch.save(Recmodel.state_dict(), best_tag_file)
            else:
                if val_hv > best_val_hv:
                    best_val_hv = val_hv
                    best_epoch = epoch
                    best_results_valid = val_metrics
                    best_state_dict = copy.deepcopy(Recmodel.state_dict())
                    patience = BUDGET
                    torch.save(Recmodel.state_dict(), weight_file)
                    best_tag_file = weight_file.replace(".pth.tar", "-best.pth.tar")
                    torch.save(Recmodel.state_dict(), best_tag_file)
                    print(f"--> Improved! New best validation hypervolume = {best_val_hv:.8f}. Resetting patience to {patience}")
                else:
                    patience -= 1
                    print(f"--> No improvement. Patience left: {patience}")
                    if patience == 0:
                        print("Early stopping triggered due to validation Pareto hypervolume plateauing.")
                        break

            if world.config.get('smoke_test', False) and epoch >= 0:
                print("[Smoke Test] Completed 1 epoch successfully.")
                break

    else:
        # Legacy training procedure
        warmup_epochs = min(100, world.TRAIN_epochs // 2)
        print(f"Early stopping warmup active: first {warmup_epochs} epochs will not trigger early stopping.")

        for epoch in range(world.TRAIN_epochs):
            # Compute validation BPR loss every epoch
            val_loss = compute_val_bpr_loss(dataset, Recmodel)
            print(f"[Valid] Epoch {epoch}: Val BPR Loss = {val_loss:.6f}")
            
            # During warmup, we always treat the latest epoch as the best and save it to disk
            if epoch < warmup_epochs:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(Recmodel.state_dict())
                torch.save(Recmodel.state_dict(), weight_file)
                best_tag_file = weight_file.replace(".pth.tar", "-best.pth.tar")
                torch.save(Recmodel.state_dict(), best_tag_file)
                print(f"--> Warmup active (Epoch {epoch}/{warmup_epochs}). Saved current checkpoints to disk.")
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_state_dict = copy.deepcopy(Recmodel.state_dict())
                    patience = BUDGET
                    torch.save(Recmodel.state_dict(), weight_file)
                    best_tag_file = weight_file.replace(".pth.tar", "-best.pth.tar")
                    torch.save(Recmodel.state_dict(), best_tag_file)
                    print(f"--> Improved! New best validation loss = {best_val_loss:.6f}. Saved checkpoints to disk. Resetting patience to {patience}")
                else:
                    patience -= 1
                    print(f"--> No improvement. Patience left: {patience}")
                    if patience == 0:
                        print("Early stopping triggered due to validation BPR loss plateauing.")
                        break

            output_information = Procedure.BPR_train_original(
                dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w
            )
            print(f"EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}")

            if world.config.get('smoke_test', False):
                print("[Smoke Test] Completed 1 epoch successfully.")
                break

    print("------ Best Valid Results -----")
    print(best_results_valid)

    # Load best weights before testing
    if best_state_dict is not None:
        Recmodel.load_state_dict(best_state_dict)
        print(f"Evaluating test set at the best epoch ({best_epoch})...")
        best_results_test = Procedure.Test(
            dataset,
            Recmodel,
            best_epoch,
            w,
            world.config["multicore"],
            dataset_name=world.dataset,
        )
        test_pop = Procedure.calculate_average_popularity(Recmodel, dataset, topk=20)
        print(f"Final Test Average Popularity@20: {test_pop:.6f}")
    else:
        print(
            "WARNING: best_state_dict is None (validation may not have run enough). "
            "Saving current model weights."
        )

    print("------ Best Test Results (at best valid epoch) -----")
    print(best_results_test)

    # Save baseline checkpoint to the usual checkpoint path
    torch.save(Recmodel.state_dict(), weight_file)
    print(f"Saved BEST checkpoint to: {weight_file}")

    # Also save a clean 'best' tag file
    best_tag_file = weight_file.replace(".pth.tar", "-best.pth.tar")
    torch.save(Recmodel.state_dict(), best_tag_file)
    print(f"Saved BEST checkpoint (tagged) to: {best_tag_file}")

    # Save results next to checkpoints
    results_file = best_tag_file.replace(".pth.tar", ".results.txt")
    with open(results_file, "w") as f:
        f.write(f"dataset={world.dataset}\n")
        f.write(f"model={world.model_name}\n")
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_valid={best_results_valid}\n")
        f.write(f"best_test={best_results_test}\n")
        if is_ppd_v6:
            f.write(f"best_val_hv={best_val_hv}\n")
            f.write(f"test_avgpop_20={test_pop}\n")
    print(f"Saved BEST results to: {results_file}")

finally:
    if world.tensorboard:
        w.close()