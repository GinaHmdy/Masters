import world
import utils
from constants import VARIANT_VECTOR_BIAS_ZERO_LAYER_UPDATE, VARIANT_EMBEDDING_UPDATE, \
    VARIANT_VECTOR_BIAS_ALL_LAYER_UPDATE, VARIANT_TRACK_AND_PRUNE
from utilities import compute_pop_bias_scores

from world import cprint
import torch
import numpy as np
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
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
BUDGET = 10

if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    budget = BUDGET
    best_result_valid_upto_r20 = 0.0
    best_results_valid = {}
    results_test = {}
    best_epoch = 0
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            cprint("[Valid]")
            results_valid = Procedure.Valid(dataset, Recmodel, epoch, w, world.config['multicore'])
            print("------ Current Valid Results -----")
            print(results_valid)
            if epoch < BUDGET * 10:
                continue
            # recall_upto20_valid = (results_valid['recall'][5] + results_valid['recall'][4] +
                                   # results_valid['recall'][3] + results_valid['recall'][2])
            recall_upto20_valid = results_valid['recall'][3]
            if recall_upto20_valid > best_result_valid_upto_r20:
                best_result_valid_upto_r20 = recall_upto20_valid
                best_results_valid = copy.deepcopy(results_valid)
                budget = BUDGET
                print("------ Updating Best Results -----")
                print(best_results_valid)

                results_test = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'],
                                              dataset_name=world.dataset)
                print("------ Current Test Results -----")
                print(results_test)

                best_epoch = epoch
            else:
                budget = budget - 1
                if budget == 0:
                    print("No more training budget")
                    # exit(0)
                    break

        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')

    print("------ Best Valid Results -----")
    print(best_results_valid)
    # print("------ Final Test Results -----")
    # print(results_test)

    # Added by Aminul -- next two blocks
    # if we use pretrained MF for propensity weight, then saving the model
    if world.model_name == 'lgn':
        torch.save(Recmodel.state_dict(), "../../data/" + world.dataset + "/model_state_dict_lgn.pth")
    else:
        torch.save(Recmodel.state_dict(), "../../data/" + world.dataset + "/model_state_dict.pth")

    # saving the best result
    if world.model_name == 'lgn':
        with open("../../data/" + world.dataset + "/results_lgn_" + world.config["variant"] + ".txt", "w") as file:
            file.write(str(results_test))
    elif world.model_name == 'mf':
        with open("../../data/" + world.dataset + "/results_mf.txt", "w") as file:
                file.write(str(results_test))
    elif world.model_name == 'dap':
        with open("../../data/" + world.dataset + "/results_dap_" + world.config["variant"] + ".txt", "w") as file:
            file.write(str(results_test))
    else:
        with open("../../data/" + world.dataset + "/results_unknown.txt", "w") as file:
                file.write(str(results_test))

    if world.model_name == 'dap':
        # start = time.time()
        # weights_1 = np.arange(0.0, 2.0 + 0.1, 0.1).round(1).tolist()
        # weights_2 = np.arange(0.0, 2.0 + 0.1, 0.1).round(1).tolist()
        # num_clusters = [1]
        weights_1 = [0.1]
        weights_2 = [0.0]
        num_clusters = [1, 5]
        best_weights_1 = weights_1[0]
        best_weights_2 = weights_2[0]
        best_cluster = num_clusters[0]
        best_result_valid_dap_upto_r20 = 0.0
        results_test_dap = {}
        for num_cluster in num_clusters:
            for weight_1 in weights_1:
                for weight_2 in weights_2:
                    results_valid_dap = Procedure.Valid(dataset, Recmodel, 1000, w, world.config['multicore'],
                                      posthoc=2, weight_1=weight_1, weight_2=weight_2, num_cluster=num_cluster)
                    print("------ DAP Valid Results -----")
                    print(results_valid_dap)
                    recall_upto20_valid_dap = results_valid_dap['recall'][3]
                    if recall_upto20_valid_dap > best_result_valid_dap_upto_r20:
                        best_result_valid_dap_upto_r20 = recall_upto20_valid_dap
                        results_test_dap = Procedure.Test(dataset, Recmodel, 1000, w, world.config['multicore'],
                                      posthoc=2, weight_1=weight_1, weight_2=weight_2, num_cluster=num_cluster)
                        print("------ DAP Test Results -----")
                        print(results_test_dap)
                        best_weights_1 = weight_1
                        best_weights_2 = weight_2
                        best_cluster = num_cluster

        print("------ DAP Best Results -----")
        print("best_weights_1 : " + str(best_weights_1) + "\n")
        print("best_weights_2 : " + str(best_weights_2) + "\n")
        print("best_cluster : " + str(best_cluster) + "\n")
        print(results_test_dap)
        # end = time.time()
        # print(f"Running time for DAP: {end - start:.4f} seconds")
    else:
        # start = time.time()
        compute_pop_bias_scores("../../data/" + world.dataset,
                            Recmodel.dataset._allPos,
                            Recmodel, Recmodel.dataset, best_epoch,
                            VARIANT_VECTOR_BIAS_ZERO_LAYER_UPDATE,
                            world.config['beta'], world.config['phi'])
        # end = time.time()
        # print(f"Running time for PPD: {end - start:.4f} seconds")
finally:
    if world.tensorboard:
        w.close()