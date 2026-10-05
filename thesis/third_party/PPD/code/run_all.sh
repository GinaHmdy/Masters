#!/bin/bash

dataset=$1
variant=$2
layer=$3
epoch=$4

residual_coffs=(0.0 0.2 0.4 0.6 0.8 1.0)
bias_severities=(1.0 2.0 3.0 4.0)
#residual_coffs=(0.2 0.4 0.6)
#bias_severities=(1.0 2.0)

procs=0
last_index_rc=$((${#residual_coffs[@]} - 1))
residual_coffs_last="${residual_coffs[$last_index_rc]}"
last_index_bs=$((${#bias_severities[@]} - 1))
bias_severities_last="${bias_severities[$last_index_bs]}"

if [ "$variant" == "layer" ]; then
    echo "Started running all the tunable parameters for layer variant."
    for residual_coff in "${residual_coffs[@]}"; do
      for bias_severity in "${bias_severities[@]}"; do
        sh run.sh "$residual_coff" False "$dataset" "$variant" "$layer" "$epoch" "$bias_severity" &
        procs=$(( $procs + 1 ))
        if [ "$residual_coff" == "$residual_coffs_last" ] && [ "$bias_severity" == "$bias_severities_last" ]; then
            echo "This is the last tunable parameters."
	      elif [ $procs -eq 5 ]; then
		      wait
		      procs=0
	      fi
      done
    done
else
    echo "Started running all the tunable parameters for mf variant."
    for residual_coff in "${residual_coffs[@]}"; do
      sh run.sh "$residual_coff" False "$dataset" "$variant" "$layer" "$epoch" 1.0 &
      procs=$(( $procs + 1 ))
	      if [ $procs -eq 1 ]; then
		      wait
		      procs=0
	      fi
    done
fi
wait