import re
import os
import numpy as np
import pandas as pd

LOG_DIR = "matbench_test"
subset = "matbench_jdft2d"

emb_list = [
    "mat2vec",
    "classical_mds_32d",
    "classical_mds_64d",
    "mds_32d",
    "mds_64d",
]

folds = range(0, 5)

# 正则：匹配 MAE = 52.07631894683838
mae_pattern = re.compile(r"MAE\s*=\s*([0-9.+-Ee]+)")

results = []

for emb in emb_list:
    row = []
    for fold in folds:
        log_name = f"crabnet_{subset}_{fold}_{emb}.log"
        log_path = os.path.join(LOG_DIR, log_name)

        if not os.path.exists(log_path):
            print(f"[WARN] 缺失日志: {log_path}")
            row.append(np.nan)
            continue

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        match = mae_pattern.search(text)
        if match:
            mae = float(match.group(1))
            mae = round(mae, 6)  # 保留6位
        else:
            print(f"[WARN] 未找到 MAE: {log_path}")
            mae = np.nan

        row.append(mae)

    # 计算平均（忽略 nan）
    avg = np.nanmean(row)
    if not np.isnan(avg):
        avg = round(avg, 6)

    row.append(avg)

    results.append([emb] + row)

# 构建 DataFrame
columns = ["emb_method"] + [f"fold_{f}" for f in folds] + ["avg_mae"]
df = pd.DataFrame(results, columns=columns)

# 保存到 excel
out_file = "matbench_jdft2d_mae_summary.xlsx"
df.to_excel(out_file, index=False)

print("Done. 保存到:", out_file)
print(df)
