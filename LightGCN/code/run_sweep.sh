#!/bin/bash
mkdir -p logs

for alpha in 0.05 0.1 0.2
do
  for theta in 0.92 0.90 0.88
  do
    for k in 1 2
    do
      for topm in 10 15
      do
        echo "Running Gowalla TSP: alpha=$alpha theta=$theta k=$k top_m=$topm"

        python run_thesis.py \
          --dataset Gowalla \
          --load 1 \
          --layer 3 \
          --recdim 64 \
          --topks "[20]" \
          --tsp \
          --theta_tsp $theta \
          --top_m_tsp $topm \
          --k_tsp $k \
          --beta_tsp 0.01 \
          --alpha_tsp $alpha \
          --layers_tsp 1 \
          --semantic_batch 128 \
          --max_edges_tsp 50000 \
          | tee logs/gowalla_tsp_a${alpha}_theta${theta}_k${k}_m${topm}.log

      done
    done
  done
done