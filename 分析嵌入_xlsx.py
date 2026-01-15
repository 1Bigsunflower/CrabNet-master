import re
import os
import numpy as np
import pandas as pd

LOG_DIR = "matbench_test"

subset_list = [
    # "matbench_jdft2d",
    # "matbench_phonons",
    # "matbench_dielectric",
    # "matbench_log_gvrh",
    # "matbench_log_kvrh",
    # "matbench_perovskites",
    "matbench_steels",
    "matbench_expt_gap",
    "matbench_expt_is_metal",
    "matbench_glass",
]
TASK_TYPE = {
    # "matbench_jdft2d": "regression",
    # "matbench_phonons": "regression",
    # "matbench_dielectric": "regression",
    # "matbench_log_gvrh": "regression",
    # "matbench_log_kvrh": "regression",
    # "matbench_perovskites": "regression",

    "matbench_steels": "regression",
    "matbench_expt_gap": "regression",

    "matbench_expt_is_metal": "classification",
    "matbench_glass": "classification",
}
emb_list = [
    "mat2vec",

    "all_6_classical_mds_32d_zscore",
    "all_6_classical_mds_64d_zscore",
    "all_6_mds_32d_zscore",
    "all_6_mds_64d_zscore",

    "all6_CMDS_32d_cos_l2_zscore",
    "all6_CMDS_64d_cos_l2_zscore",
    "all6_MDS_32d_cos_l2_zscore",
    "all6_MDS_64d_cos_l2_zscore",
]

folds = range(0, 5)

# 匹配 MAE = 52.07631894683838
mae_pattern = re.compile(r"MAE\s*=\s*([0-9.+-Ee]+)")
auc_pattern = re.compile(r"(ROC[-_ ]?AUC|AUC)\s*=\s*([0-9.+-Ee]+)")

rows = []

for subset in subset_list:
    task_type = TASK_TYPE.get(subset)
    if task_type is None:
        raise ValueError(f"未知的 subset 类型: {subset}")

    is_classification = task_type == "classification"

    for emb in emb_list:
        fold_scores = []

        for fold in folds:
            log_name = f"crabnet_{subset}_{fold}_{emb}.log"
            log_path = os.path.join(LOG_DIR, log_name)

            if not os.path.exists(log_path):
                print(f"[WARN] 缺失日志: {log_path}")
                fold_scores.append(np.nan)
                continue

            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            if is_classification:
                match = auc_pattern.search(text)
                if match:
                    score = round(float(match.group(2)), 6)
                else:
                    print(f"[WARN] 未找到 AUC: {log_path}")
                    score = np.nan
            else:
                match = mae_pattern.search(text)
                if match:
                    score = round(float(match.group(1)), 6)
                else:
                    print(f"[WARN] 未找到 MAE: {log_path}")
                    score = np.nan

            fold_scores.append(score)

        avg = np.nanmean(fold_scores)
        if not np.isnan(avg):
            avg = round(avg, 6)

        metric_name = "auc" if is_classification else "mae"
        rows.append([subset, emb] + fold_scores + [avg])

    rows.append([np.nan] * (2 + len(folds) + 1))

columns = ["subset", "emb_method"] + [f"fold_{f}" for f in folds] + ["avg_score"]

df = pd.DataFrame(rows, columns=columns)

out_file = "matbench_all_subset_mae_summary.xlsx"
df.to_excel(out_file, index=False)

print("Done. 保存到:", out_file)
print(df)
