#!/bin/bash

PYFILE="train_crabnet_matbench.py"
OUTDIR="matbench_test"

mkdir -p ${OUTDIR}

emb_list=(
    # "mat2vec"
    "CMDS_32_cos_zscore"
    "CMDS_32_euc_zscore"
    "CMDS_64_cos_zscore"
    "CMDS_64_euc_zscore"
    "MDS_32_cos_zscore"
    "MDS_32_euc_zscore"
    "MDS_64_cos_zscore"
    "MDS_64_euc_zscore"
#    "all_6_classical_mds_32d_zscore"
#    "all_6_classical_mds_64d_zscore"
#    "all_6_mds_32d_zscore"
#    "all_6_mds_64d_zscore"
#
#    "all6_CMDS_32d_cos_l2_zscore"
#    "all6_CMDS_64d_cos_l2_zscore"
#    "all6_MDS_32d_cos_l2_zscore"
#    "all6_MDS_64d_cos_l2_zscore"

)

subset_list=(
    # "matbench_jdft2d"
    # "matbench_phonons"
#    "matbench_dielectric"
#    "matbench_log_gvrh"
#    "matbench_log_kvrh"
#    "matbench_perovskites"
      "matbench_steels"
      "matbench_expt_gap"
      "matbench_expt_is_metal"
      "matbench_glass"
)

for subset in "${subset_list[@]}"; do
    for emb in "${emb_list[@]}"; do
        for fold in {0..4}; do

            log_file="${OUTDIR}/crabnet_${subset}_${fold}_${emb}.log"

            echo "Running: subset=${subset}, emb_method=${emb}, fold=${fold}"
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
done

echo "All jobs finished."
