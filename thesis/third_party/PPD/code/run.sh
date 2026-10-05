#!/bin/bash

#./run.sh 0.6 False Epinions mf 4 1000 1.0
#./run.sh 0.6 False Epinions layer 4 1000 1.0

#./run.sh 0.2 True gowalla mf 4 1000 1.5
#./run.sh 0.2 True gowalls layer 4 1000 1.5

# hyperparamters -
# residual_coff {0.1, ..., 1.0}
# layer {1, ..., 5}

residual_coff=$1
exp_on=$2
dataset=$3
layer=$4
epoch=$5

#start_time=$(date +%s)
python main.py --residual_coff $residual_coff --exp_on $exp_on --dataset $dataset --layer $layer --epochs $epoch
#end_time=$(date +%s)
#runtime=$((end_time - start_time))

echo "Main model running done for layer variant"
#echo "Total running time: ${runtime} seconds"