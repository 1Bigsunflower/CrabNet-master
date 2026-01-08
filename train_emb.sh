#!/bin/bash

MAX_JOBS=1   # 同时运行的最大数

mkdir -p logs

METHODS=(
mat2vec
FCC BCC SC DMD
X2O X2O3 X2O5 XO XO2 XO3
FCC_scale_max FCC_log_abs FCC_zscore
BCC_scale_max BCC_log_abs BCC_zscore
SC_scale_max SC_log_abs SC_zscore
DMD_scale_max DMD_log_abs DMD_zscore
X2O_scale_max X2O_log_abs X2O_zscore
X2O3_scale_max X2O3_log_abs X2O3_zscore
X2O5_scale_max X2O5_log_abs X2O5_zscore
XO_scale_max XO_log_abs XO_zscore
XO2_scale_max XO2_log_abs XO2_zscore
XO3_scale_max XO3_log_abs XO3_zscore
)

for m in "${METHODS[@]}"
do
    echo "Running $m ..."
    nohup python train_crabnet.py --emb_method "$m" > "logs/train_${m}.log" 2>&1 &

    # 控制最多只跑 MAX_JOBS 个
    while (( $(jobs -r | wc -l) >= MAX_JOBS ))
    do
        wait -n   # 等待任意一个后台任务结束
    done
done

wait   # 等所有任务结束
echo "All jobs finished."
