# coding: utf-8
import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device
parser = argparse.ArgumentParser(description='')
parser.add_argument('--emb_method', default='mat2vec', type=str,
                    # choices=['mat2vec', 'FCC', 'BCC', 'SC',' DMD', 'X2O','X2O3','X2O5','XO','XO2','XO3'],
                    help='embedding methods to use')
args = parser.parse_args()
torch.set_printoptions(threshold=torch.inf)
torch.set_printoptions(sci_mode=False)
warnings.filterwarnings("ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False")

# ===============================
# 全局设置
# ===============================
compute_device = get_compute_device(prefer_last=True)
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)


def get_model(data_dir, mat_prop, classification=False, batch_size=None,
              transfer=None, verbose=True, embedding_dir='mat2vec'):

    model = Model(
        CrabNet(
            compute_device=compute_device,
            embedding_dir=embedding_dir
        ).to(compute_device,),
        model_name=f'{mat_prop}',
        verbose=verbose,
    )

    if transfer is not None:
        model.load_network(f'{transfer}.pth')
        model.model_name = f'{mat_prop}'

    if classification:
        model.classification = True

    train_data = f'{data_dir}/{mat_prop}/train.csv'

    # ====== batch size 计算（原样保留）======
    data_size = pd.read_csv(train_data).shape[0]
    batch_size = 2 ** round(np.log2(data_size) - 4)
    batch_size = min(max(batch_size, 2**7), 2**12)

    # df = pd.read_csv(train_data)
    # print(df.head(2))

    model.load_data(train_data, batch_size=batch_size, train=True)
    print(f'training with batchsize {model.batch_size}')

    model.fit(epochs=500, losscurve=False)

    model.save_network()
    return model


# ===============================
# 预测与保存
# ===============================
def to_csv(output, save_name):
    act, pred, formulae, uncertainty = output
    df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    df.columns = ['composition', 'target', 'pred-0', 'uncertainty']
    os.makedirs('model_predictions', exist_ok=True)
    df.to_csv(f'model_predictions/{save_name}', index_label='Index')


def load_model(data_dir, mat_prop, classification, file_name, verbose=True):
    model = Model(
        CrabNet(compute_device=compute_device,
                embedding_dir=args.emb_method
                ).to(compute_device),
        # model_name=f'{mat_prop}',
        verbose=verbose
    )
    import os
    # print("当前工作目录:", os.getcwd())
    # ⭐ 构造模型路径
    model_path = os.path.join('models', 'trained_models', f"{mat_prop}.pth")
    # print("模型查找路径:", model_path)

    # ⭐ 检查是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    model.load_network(f"{mat_prop}.pth")

    if classification:
        model.classification = True

    data = f'{data_dir}/{mat_prop}/{file_name}'
    model.load_data(data, batch_size=2**9, train=False)
    return model


def save_results(data_dir, mat_prop, classification, file_name, verbose=True):
    model = load_model(data_dir, mat_prop, classification, file_name, verbose)
    output = model.predict(model.data_loader)

    y_true, y_pred = output[0], output[1]

    if model.classification:
        auc = roc_auc_score(y_true, y_pred)
        print(f'{mat_prop} ROC AUC ({file_name}): {auc:.4f}')
        return auc

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # ✅ 打印所有指标，包括 MSE
    print(f'{file_name} metrics:')
    print(f'  MAE  : {mae:.4f}')
    print(f'  MSE  : {mse:.4f}')
    print(f'  RMSE : {rmse:.4f}')
    print(f'  R²   : {r2:.4f}')

    fname = f'{mat_prop}_{file_name.replace(".csv", "")}_output.csv'
    to_csv(output, fname)

    os.makedirs("model_metrics", exist_ok=True)
    metrics_path = f"model_metrics/{mat_prop}_metrics.csv"
    pd.DataFrame([{
        "split": file_name.replace(".csv", ""),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }]).to_csv(
        metrics_path,
        mode='a',
        header=not os.path.exists(metrics_path),
        index=False
    )

    return mae, mse, rmse, r2


# ===============================
# 化学式分数转小数
# ===============================
def frac_to_decimal_in_formula(formula, ndigits=2):
    if not isinstance(formula, str):
        return formula

    pattern = r'([A-Z][a-z]?)(\d+)\s*/\s*(\d+)'

    def repl(m):
        return f"{m.group(1)}{round(float(m.group(2)) / float(m.group(3)), ndigits)}"

    return re.sub(pattern, repl, formula)


if __name__ == '__main__':

    df = pd.read_excel("Unfiltered.xlsx")
    df = df[["formula", "target"]]

    df["formula"] = (
        df["formula"]
        .astype(str)
        .str.replace(r'\s+|\u200b', '', regex=True)
        .apply(frac_to_decimal_in_formula)
    )

    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42
    )

    data_dir = 'data/m_data'
    classification = False
    train = True

    mat_prop = f'property_{args.emb_method}'
    model_file = f'models/trained_models/{mat_prop}.pth'
    base_dir = Path(f"data/m_data/{mat_prop}")
    base_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(base_dir / "train.csv", index=False)
    test_df.to_csv(base_dir / "test.csv", index=False)
    if os.path.exists(model_file):
        print(f"删除旧的模型文件: {model_file}")
        os.remove(model_file)
    if train:
        print(f'Property "{mat_prop}" selected for training')
        get_model(data_dir, mat_prop, classification, embedding_dir=f'{args.emb_method}')
    print('=' * 50)
    save_results(data_dir, mat_prop, classification, 'train.csv', verbose=False)
    save_results(data_dir, mat_prop, classification, 'test.csv', verbose=False)
    print('=' * 50)


# coding: utf-8
# mat2vec 0.4414
# 0.5227