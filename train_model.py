import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
from PIL import Image

# ---------- 新增：Optuna ----------
import optuna

from model import LSTM_CNN_Model

# 自動判斷設備
if torch.cuda.is_available():
    device = torch.device("cuda")  # 如果有 CUDA，使用 GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # 如果是 Apple Silicon，使用 MPS
else:
    device = torch.device("cpu")  # 默認使用 CPU

print(f"Using device: {device}")

# 固定的超參數 (或將其中部分改為 Optuna 搜尋範圍)
LSTM_INPUT_DIM = 10  # LSTM 特徵數 (通常固定，因為對應你的技術指標數)
IMG_SIZE = (64, 64)
WINDOW_SIZE = 10

# 以下作為「最終訓練」的預設 epoch
FINAL_EPOCHS = 50

# 讀取股票數據
with open("stock_df.pkl", "rb") as f:
    stock_data = pickle.load(f)  # stock_data 是 dict，key 是股票代號，value 是 dataframe

# 圖片轉換 (為了簡化，先保留最基礎的 transform，如果需要更多增強可自行調整)
transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ]
)


def prepare_data_for_one_stock(df, window_size=30):
    """
    傳入單一股票的 df，回傳 X_lstm, X_cnn, y。
    假設 df 按日期排序 (由舊到新)，以免未來資料提前參與訓練。
    """
    df = df.dropna().copy()  # 移除空值

    # 如果資料量不足以切到 window_size，就直接回傳空
    if len(df) <= window_size:
        return None, None, None

    # MinMaxScaler：縮放 Close、各技術指標
    scaler = MinMaxScaler()
    # 注意：這樣會每檔股票各自fit！若想整體共用或各檔獨立，請自行調整
    scaled_features = scaler.fit_transform(
        df[
            [
                'Close',
                'MA_5',
                'MA_10',
                'MA_20',
                'slowk',
                'slowd',
                'RSI_14',
                'MACD',
                'MACD_signal',
                'MACD_hist',
            ]
        ].values
    )
    # 這裡範例是每次都寫死 "scaler.pkl"，若多檔股票會被覆蓋；可考慮用 f"scaler_{stock_id}.pkl" 分別存
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    labels = df['Close'].values
    img_paths = df['image_path'].values

    X_lstm, X_cnn, y = [], [], []

    # 逐筆建立樣本 (時間序列)
    for i in range(window_size, len(df) - 1):
        # LSTM 的部分：取前 window_size 筆
        X_lstm.append(scaled_features[i - window_size : i])

        # CNN 的部分：讀取對應的圖 (當天)
        img = Image.open(img_paths[i]).convert("RGB")
        img = transform(img)
        X_cnn.append(img)

        # label 取「當天之後一天」的 close
        y.append(labels[i + 1])

    X_lstm = np.array(X_lstm, dtype=np.float32)
    X_cnn = torch.stack(X_cnn)  # [N, 3, H, W] tensor
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    return X_lstm, X_cnn, y


def objective(trial, X_lstm_train, X_lstm_val, X_cnn_train, X_cnn_val, y_train, y_val):
    """
    交給 Optuna 的目標函式:
    接收資料，依照 trial 中提議的超參數建構模型，
    簡單訓練幾個 epoch 後，回傳 validation loss 供 Optuna 參考。
    """

    # 1. 從 trial 中取超參數
    lstm_hidden_dim = trial.suggest_categorical("lstm_hidden_dim", [16, 32, 64])
    lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 2)
    cnn_output_channels = trial.suggest_categorical("cnn_output_channels", [8, 16, 32])
    cnn_kernel_size = trial.suggest_categorical("cnn_kernel_size", [3, 5])
    cnn_fc_size = trial.suggest_categorical("cnn_fc_size", [64, 128])
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "AdamW"])
    criterion_name = trial.suggest_categorical("criterion", ["L1Loss", "MSELoss", "HuberLoss"])

    # 2. 建立模型
    model = LSTM_CNN_Model(
        lstm_input_dim=LSTM_INPUT_DIM,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_num_layers=lstm_num_layers,
        cnn_input_channels=3,
        cnn_output_channels=cnn_output_channels,
        cnn_kernel_size=cnn_kernel_size,
        cnn_fc_size=cnn_fc_size,
    ).to(device)

    # 損失函式
    if criterion_name == "L1Loss":
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    # 優化器
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # 3. 簡單的超參數搜尋時，別把 epoch 設太大，否則時間會很久
    search_epochs = 20

    # 開始訓練 (只做簡化版本)
    for epoch in range(search_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_lstm_train, X_cnn_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # 訓練完後，用 validation set 計算 loss 當作回傳
    model.eval()
    with torch.no_grad():
        val_preds = model(X_lstm_val, X_cnn_val)
        val_loss = criterion(val_preds, y_val).item()

    return val_loss


def tune_hyperparameters(X_lstm_all, X_cnn_all, y_all):
    """
    使用 Optuna 對超參數進行搜尋 (針對單一股票的資料)。
    回傳最佳參數字典 best_params。
    """

    # 先做簡單的「時間序列切分」前 70% 做 train，後 30% 做 validation
    N = len(X_lstm_all)
    split_idx = int(N * 0.7)

    X_lstm_train = X_lstm_all[:split_idx]
    X_lstm_val = X_lstm_all[split_idx:]
    X_cnn_train = X_cnn_all[:split_idx]
    X_cnn_val = X_cnn_all[split_idx:]
    y_train = y_all[:split_idx]
    y_val = y_all[split_idx:]

    # 搬到 device
    X_lstm_train = torch.tensor(X_lstm_train, dtype=torch.float32).to(device)
    X_lstm_val = torch.tensor(X_lstm_val, dtype=torch.float32).to(device)
    X_cnn_train = X_cnn_train.to(device)
    X_cnn_val = X_cnn_val.to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    def wrapped_objective(trial):
        return objective(trial, X_lstm_train, X_lstm_val, X_cnn_train, X_cnn_val, y_train, y_val)

    # 建立 study
    study = optuna.create_study(direction="minimize")
    study.optimize(wrapped_objective, n_trials=50, timeout=None)  # 可自行調整 n_trials

    print("Study best trial:", study.best_trial.value)
    print("Best hyperparameters:", study.best_params)

    return study.best_params


def train_and_evaluate_one_stock(stock_id, df, final_epochs=20):
    """
    對單一股票:
      1) 先用 Optuna 搜尋超參數
      2) 再用最佳超參數完整訓練 (前 80% train, 後 20% test)
      3) 繪製 loss 和預測結果
      4) 儲存權重 & 回傳 best_params
    """
    # 準備資料
    result = prepare_data_for_one_stock(df, window_size=WINDOW_SIZE)
    if result[0] is None:
        print(f"股票 {stock_id} 資料不足，跳過訓練。")
        return None

    X_lstm_all, X_cnn_all, y_all = result

    # -------------------
    # 1) 使用 Optuna 尋找超參數
    # -------------------
    best_params = tune_hyperparameters(X_lstm_all, X_cnn_all, y_all)

    # -------------------
    # 2) 根據最佳超參數進行完整訓練 (前 80% train, 後 20% test)
    # -------------------
    N = len(X_lstm_all)
    split_idx = int(N * 0.8)

    X_lstm_train = X_lstm_all[:split_idx]
    X_lstm_test = X_lstm_all[split_idx:]
    X_cnn_train = X_cnn_all[:split_idx]
    X_cnn_test = X_cnn_all[split_idx:]
    y_train = y_all[:split_idx]
    y_test = y_all[split_idx:]

    # 取出測試集的日期範圍
    test_dates = df.index[split_idx:]
    with open("test_dates.pkl", "wb") as f:
        pickle.dump(test_dates, f)

    X_lstm_train = torch.tensor(X_lstm_train, dtype=torch.float32).to(device)
    X_lstm_test = torch.tensor(X_lstm_test, dtype=torch.float32).to(device)
    X_cnn_train = X_cnn_train.to(device)
    X_cnn_test = X_cnn_test.to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # 從 best_params 取出對應值
    lstm_hidden_dim = best_params["lstm_hidden_dim"]
    lstm_num_layers = best_params["lstm_num_layers"]
    cnn_output_channels = best_params["cnn_output_channels"]
    cnn_kernel_size = best_params["cnn_kernel_size"]
    cnn_fc_size = best_params["cnn_fc_size"]
    lr = best_params["lr"]

    # 建立最終模型
    model = LSTM_CNN_Model(
        lstm_input_dim=LSTM_INPUT_DIM,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_num_layers=lstm_num_layers,
        cnn_input_channels=3,
        cnn_output_channels=cnn_output_channels,
        cnn_kernel_size=cnn_kernel_size,
        cnn_fc_size=cnn_fc_size,
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n=== 股票 {stock_id} 最終訓練，使用最佳參數 {best_params} ===")
    train_losses = []

    for epoch in range(final_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_lstm_train, X_cnn_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        print(f"[{stock_id}] Epoch [{epoch+1}/{final_epochs}]  Loss: {loss.item():.4f}")

    # -------------------
    # 3) 評估模型 & 繪製圖表
    # -------------------
    # 繪製訓練 Loss 圖
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, final_epochs + 1), train_losses, label='Train Loss')
    plt.title(f'Train Loss - {stock_id}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    loss_plot_name = f'{stock_id}_loss.png'
    plt.savefig(loss_plot_name)
    plt.close()
    print(f"Loss 曲線圖已儲存為: {loss_plot_name}")

    # 評估模型
    model.eval()
    with torch.no_grad():
        predictions = model(X_lstm_test, X_cnn_test)
        test_loss = criterion(predictions, y_test).item()

    # 將 tensor 移回 CPU
    predictions = predictions.cpu().numpy().flatten()
    y_test = y_test.cpu().numpy().flatten()

    # 繪製「真實值 vs 預測值」圖
    plt.figure(figsize=(8, 5))
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title(f'Actual vs Predicted - {stock_id}')
    plt.xlabel('Index')
    plt.ylabel('Close Price')
    plt.legend()
    plt.tight_layout()
    pred_plot_name = f'{stock_id}_prediction.png'
    plt.savefig(pred_plot_name)
    plt.close()
    print(f"真實值 vs 預測值 圖已儲存為: {pred_plot_name}")

    # 指標
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    print(f"[{stock_id}] Test Loss: {test_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

    # 保存模型 (權重)
    model_save_name = f"{stock_id}_best_lstm_cnn_model.pth"
    torch.save(model.state_dict(), model_save_name)
    print(f"模型權重已儲存為: {model_save_name}")
    print(f"=== 股票 {stock_id} 訓練完成 ===\n")

    # 檢查變數長度
    print(f"Length of test_dates: {len(test_dates)}")
    print(f"Length of y_test: {len(y_test)}")
    print(f"Length of predictions: {len(predictions)}")

    # 確保長度一致
    min_length = min(len(test_dates), len(y_test), len(predictions))
    test_dates = test_dates[:min_length]
    y_test = y_test[:min_length]
    predictions = predictions[:min_length]

    # 保存真實值和預測值到 DataFrame
    results_df = pd.DataFrame({'Date': test_dates, 'Actual': y_test, 'Predicted': predictions})
    results_df.to_pickle(f"{stock_id}_results.pkl")
    print(f"真實值和預測值已儲存為: {stock_id}_results.pkl")

    return best_params  # 回傳該檔股票的最佳參數


if __name__ == "__main__":

    # 用來保存「所有股票對應的最佳參數」
    best_params_all_stocks = {}

    # 針對 stock_data 中每一檔股票分別「搜尋超參數 + 評估」
    for stock_id, df in stock_data.items():
        # 若 df 沒確定是時間排序，可在此先做: df.sort_index(inplace=True)
        best_params = train_and_evaluate_one_stock(stock_id, df, final_epochs=FINAL_EPOCHS)
        if best_params is not None:
            # 收集到字典
            best_params_all_stocks[stock_id] = best_params

    # 將最佳參數字典統一儲存到 pkl
    with open("best_params_all.pkl", "wb") as f:
        pickle.dump(best_params_all_stocks, f)

    print("所有股票最佳參數已儲存到 best_params_all.pkl !")
