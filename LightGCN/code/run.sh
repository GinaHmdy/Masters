#!/bin/bash

#./run.sh KuaiRec lgn navip 4 1000
#./run.sh KuaiRec lgn lgn 4 1000

dataset=$1
model=$2
variant=$3
layer=$4
epoch=$5
beta=$6
phi=$7

#start_time=$(date +%s)
python main.py --dataset $dataset --model $model --variant $variant --layer $layer --epochs $epoch --beta $beta --phi $phi
#end_time=$(date +%s)
#runtime=$((end_time - start_time))

echo "Running done for $variant$ variant"
#echo "Total running time: ${runtime} seconds"