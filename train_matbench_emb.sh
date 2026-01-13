#!/bin/bash

PYFILE="train_crabnet_matbench.py"
OUTDIR="matbench_test"

mkdir -p ${OUTDIR}

emb_list=(
    "mat2vec"
    "classical_mds_32d"
    "classical_mds_64d"
    "mds_32d"
    "mds_64d"
)

subset="matbench_jdft2d"

for emb in "${emb_list[@]}"; do
    for fold in {0..4}; do

        log_file="${OUTDIR}/crabnet_${subset}_${fold}_${emb}.log"

        echo "Running: emb_method=${emb}, fold=${fold}, subset=${subset}"
        echo "Log -> ${log_file}"

        nohup python ${PYFILE} \
            --emb_method "${emb}" \
            --fold ${fold} \
            --subset "${subset}" \
            > "${log_file}" 2>&1


        echo "Finished: ${log_file}"
        echo "--------------------------------------"
    done
done

echo "All jobs finished."
