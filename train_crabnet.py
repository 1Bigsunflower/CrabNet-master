import os
import numpy as np
import pandas as pd
import torch
import re
from pathlib import Path
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score


from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device
from sklearn.model_selection import train_test_split

compute_device = get_compute_device(prefer_last=True)
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)



# %%

def get_model(data_dir, mat_prop, classification=False, batch_size=None,
              transfer=None, verbose=True):
    # Get the TorchedCrabNet architecture loaded
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{mat_prop}', verbose=verbose)

    # Train network starting at pretrained weights
    if transfer is not None:
        model.load_network(f'{transfer}.pth')
        model.model_name = f'{mat_prop}'

    # Apply BCEWithLogitsLoss to model output if binary classification is True
    if classification:
        model.classification = True

    # Get the datafiles you will learn from
    train_data = f'{data_dir}/{mat_prop}/train.csv'
    try:
        val_data = f'{data_dir}/{mat_prop}/val.csv'
    except:
        print('Please ensure you have train (train.csv) and validation data',
               f'(val.csv) in folder "data/materials_data/{mat_prop}"')

    # Load the train and validation data before fitting the network
    data_size = pd.read_csv(train_data).shape[0]
    batch_size = 2**round(np.log2(data_size)-4)
    if batch_size < 2**7:
        batch_size = 2**7
    if batch_size > 2**12:
        batch_size = 2**12
    model.load_data(train_data, batch_size=batch_size, train=True)
    print(f'training with batchsize {model.batch_size} '
          f'(2**{np.log2(model.batch_size):0.3f})')
    model.load_data(val_data, batch_size=batch_size)

    # Set the number of epochs, decide if you want a loss curve to be plotted
    model.fit(epochs=500, losscurve=False)

    # Save the network (saved as f"{model_name}.pth")
    model.save_network()
    return model


def to_csv(output, save_name):
    # parse output and save to csv
    act, pred, formulae, uncertainty = output
    df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    df.columns = ['composition', 'target', 'pred-0', 'uncertainty']
    save_path = 'model_predictions'
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f'{save_path}/{save_name}', index_label='Index')


def load_model(data_dir, mat_prop, classification, file_name, verbose=True):
    # Load up a saved network.
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{mat_prop}', verbose=verbose)
    model.load_network(f'{mat_prop}.pth')

    # Check if classifcation task
    if classification:
        model.classification = True

    # Load the data you want to predict with
    data = f'{data_dir}/{mat_prop}/{file_name}'
    # data is reloaded to model.data_loader
    model.load_data(data, batch_size=2**9, train=False)
    return model


def get_results(model):
    output = model.predict(model.data_loader)  # predict the data saved here
    return model, output


def save_results(data_dir, mat_prop, classification, file_name, verbose=True):
    model = load_model(data_dir, mat_prop, classification, file_name, verbose=verbose)
    model, output = get_results(model)

    y_true = output[0]
    y_pred = output[1]

    # Classification
    if model.classification:
        auc = roc_auc_score(y_true, y_pred)
        print(f'{mat_prop} ROC AUC: {auc:0.3f}')
        return model, auc

    # Regression
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f'{mat_prop} metrics:')
    print(f'  MAE  : {mae:.4f}')
    print(f'  MSE  : {mse:.4f}')
    print(f'  RMSE : {rmse:.4f}')
    print(f'  R²   : {r2:.4f}')

    # save predictions to csv
    fname = f'{mat_prop}_{file_name.replace(".csv", "")}_output.csv'
    to_csv(output, fname)

    # （可选）保存指标到 CSV
    metrics_df = pd.DataFrame([{
        "split": file_name.replace(".csv", ""),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }])

    os.makedirs("model_metrics", exist_ok=True)
    metrics_path = f"model_metrics/{mat_prop}_metrics.csv"
    metrics_df.to_csv(metrics_path,
                      mode='a',
                      header=not os.path.exists(metrics_path),
                      index=False)

    return model, mae, mse, rmse, r2


def frac_to_decimal_in_formula(formula, ndigits=2):
    """
    将化学式中形如 Fe1/3 的分数转为 Fe0.33
    """
    if not isinstance(formula, str):
        return formula

    # 匹配 元素符号 + 分数
    pattern = r'([A-Z][a-z]?)(\d+)\s*/\s*(\d+)'

    def repl(m):
        elem = m.group(1)
        num = float(m.group(2))
        den = float(m.group(3))
        val = round(num / den, ndigits)
        return f"{elem}{val}"

    return re.sub(pattern, repl, formula)

# %%
if __name__ == '__main__':
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    # 读取数据
    df = pd.read_excel("../北科大/data/分子式.xls")
    df = df[["formula", "target"]]

    # 清理空格和零宽字符
    df["formula"] = df["formula"].astype(str).str.replace(r'\s+|\u200b', '', regex=True)

    # 转分数为小数
    df["formula"] = df["formula"].apply(frac_to_decimal_in_formula)

    # 创建存储目录
    base_dir = Path("data/m_data/property")
    base_dir.mkdir(parents=True, exist_ok=True)

    # 数据划分，和第一段保持一致
    TrainData, TestData = train_test_split(df, test_size=0.3, random_state=42)

    # 保存 CSV
    TrainData.to_csv(base_dir / "train.csv", index=False)
    TestData.to_csv(base_dir / "test.csv", index=False)

    from utils.composition import CompositionError

    bad_rows = []

    for i, f in enumerate(df["formula"]):
        try:
            from utils.composition import _element_composition

            _element_composition(str(f))
        except Exception:
            bad_rows.append((i, f))

    print("非法 formula 行数:", len(bad_rows))
    print("前 10 个非法值:", bad_rows[:10])

    # Choose the directory where your data is storedF
    data_dir = 'data/m_data'
    #     # Choose the folder with your materials properties
    mat_prop = 'property'
    # Choose if you data is a regression or binary classification
    classification = False
    # train = False
    train = True

    # Train your model using the "get_model" function
    if train:
        print(f'Property "{mat_prop}" selected for training')
        model = get_model(data_dir, mat_prop, classification, verbose=True)

    cutter = '====================================================='
    first = " "*((len(cutter)-len(mat_prop))//2) + " "*int((len(mat_prop)+1)%2)
    last = " "*((len(cutter)-len(mat_prop))//2)
    print('=====================================================')
    print(f'{first}{mat_prop}{last}')
    print('=====================================================')
    print('calculating train mae')
    model_train, mae_train, mse_train, rmse_train, r2_train = save_results(
        data_dir, mat_prop, classification, 'train.csv', verbose=False)

    model_val, mae_val, mse_val, rmse_val, r2_val = save_results(
        data_dir, mat_prop, classification, 'val.csv', verbose=False)

    model_test, mae_test, mse_test, rmse_test, r2_test = save_results(
        data_dir, mat_prop, classification, 'test.csv', verbose=False)

    print('=====================================================')