# profile_kuairec.py
import sys
sys.argv += ['--dataset', 'KuaiRec', '--model', 'dap', '--layer', '2', '--recdim', '256', '--sgl', '--sgl_weight', '0.1']
import torch
import time
import cProfile
import pstats
import world
import register
from register import dataset
import Procedure
import utils

Recmodel = register.MODELS['dap'](world.config, dataset).to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)
w = None
Recmodel.train()

print("Profiling one epoch of training...")
profiler = cProfile.Profile()
profiler.enable()

t0 = time.time()
output = Procedure.BPR_train_original(dataset, Recmodel, bpr, 0, neg_k=1, w=w)
t1 = time.time()

profiler.disable()
print(f"Total epoch training time: {t1 - t0:.2f} seconds")
print(output)

stats = pstats.Stats(profiler).sort_stats('tottime')
stats.print_stats(30)
