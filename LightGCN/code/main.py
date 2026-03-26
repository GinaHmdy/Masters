import world
import utils
from world import cprint

import torch
from tensorboardX import SummaryWriter
import time
import Procedure
import copy
from os.path import join

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
        Recmodel.load_state_dict(torch.load(weight_file, map_location=world.device))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

Neg_k = 1

# init tensorboard
BUDGET = 10

if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    budget = BUDGET

    best_state_dict = None
    best_result_valid_upto_r20 = -1.0
    best_results_valid = {}
    best_results_test = {}
    best_epoch = -1

    for epoch in range(world.TRAIN_epochs):
        if epoch % 10 == 0:
            cprint("[Valid]")
            results_valid = Procedure.Valid(dataset, Recmodel, epoch, w, world.config["multicore"])
            print("------ Current Valid Results -----")
            print(results_valid)

            # Only start early-stopping decisions after warmup,
            # but DO NOT skip training for these epochs.
            if epoch >= BUDGET * 10:
                # using recall@20 (based on your current indexing)
                recall_upto20_valid = results_valid["recall"][3]

                if recall_upto20_valid > best_result_valid_upto_r20:
                    best_result_valid_upto_r20 = recall_upto20_valid
                    best_results_valid = copy.deepcopy(results_valid)
                    best_epoch = epoch

                    # store best weights
                    best_state_dict = copy.deepcopy(Recmodel.state_dict())

                    budget = BUDGET
                    print("------ Updating Best Results -----")
                    print(best_results_valid)

                    # store test results at the same best point
                    best_results_test = Procedure.Test(
                        dataset,
                        Recmodel,
                        epoch,
                        w,
                        world.config["multicore"],
                        dataset_name=world.dataset,
                    )
                    print("------ Current Test Results (Best so far) -----")
                    print(best_results_test)
                else:
                    budget -= 1
                    if budget == 0:
                        print("No more training budget")
                        break

        output_information = Procedure.BPR_train_original(
            dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w
        )
        print(f"EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}")

    print("------ Best Valid Results -----")
    print(best_results_valid)

    print("------ Best Test Results (at best valid epoch) -----")
    print(best_results_test)

    # Load best weights before saving
    if best_state_dict is not None:
        Recmodel.load_state_dict(best_state_dict)
    else:
        print(
            "WARNING: best_state_dict is None (validation may not have run enough). "
            "Saving current model weights."
        )

    # Save baseline checkpoint to the usual checkpoint path
    torch.save(Recmodel.state_dict(), weight_file)
    print(f"Saved BEST baseline checkpoint to: {weight_file}")

    # Also save a clean 'best' tag file
    best_tag_file = weight_file.replace(".pth.tar", "-best.pth.tar")
    torch.save(Recmodel.state_dict(), best_tag_file)
    print(f"Saved BEST baseline checkpoint (tagged) to: {best_tag_file}")

    # Save results next to checkpoints
    results_file = best_tag_file.replace(".pth.tar", ".results.txt")
    with open(results_file, "w") as f:
        f.write(f"dataset={world.dataset}\n")
        f.write(f"model={world.model_name}\n")
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_valid={best_results_valid}\n")
        f.write(f"best_test={best_results_test}\n")
    print(f"Saved BEST baseline results to: {results_file}")

finally:
    if world.tensorboard:
        w.close()