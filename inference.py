import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd

from model import LSTM_CNN_Model  # 你的模型定義檔

# 如果 GPU 可用就用 GPU，否則用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###########################
# 一些固定超參數 (需與訓練對應)
###########################
LSTM_INPUT_DIM = 10  # 你用到的技術指標數量(例如 Close, MA_5, MA_10, MA_20, slowk, slowd, RSI_14, MACD, MACD_signal, MACD_hist)

# CNN 影像處理流程，需與訓練時一致
IMG_SIZE = (64, 64)
transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ]
)


def load_scaler(scaler_path):
    """
    讀取訓練時的 MinMaxScaler，確保推論時的縮放方式一致。
    """
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def load_model(
    model_path,
    lstm_hidden_dim=32,
    lstm_num_layers=1,
    cnn_output_channels=16,
    cnn_kernel_size=3,
    cnn_fc_size=64,
):
    """
    依照你訓練時的最佳參數，建立模型後載入 .pth 權重。
    如果各檔股票的最佳參數不同，就要因檔而異；這裡以簡化範例示意。
    """
    model = LSTM_CNN_Model(
        lstm_input_dim=LSTM_INPUT_DIM,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_num_layers=lstm_num_layers,
        cnn_input_channels=3,  # RGB
        cnn_output_channels=cnn_output_channels,
        cnn_kernel_size=cnn_kernel_size,
        cnn_fc_size=cnn_fc_size,
    ).to(device)

    # 載入權重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # 設定為推論模式
    return model


def prepare_single_inference(df_10, scaler, image_path):
    """
    給定「連續 10 天」的原始資料 df_10 (含 Close、MA_5、MA_10、...、MACD_hist)，
    以及當天對應的圖 (image_path)，返回 LSTM 與 CNN 的 tensor。

    df_10: 形狀 (10, 多欄)，至少包含下列欄位:
        [
            'Close', 'MA_5', 'MA_10', 'MA_20',
            'slowk', 'slowd', 'RSI_14',
            'MACD', 'MACD_signal', 'MACD_hist'
        ]
    scaler: MinMaxScaler (與訓練時相同)
    image_path: 該天對應的圖檔路徑

    回傳:
    X_lstm: shape [1, 10, 10]
    X_cnn: shape [1, 3, 64, 64]
    """
    # 1) 選擇要用的 10 個欄位
    needed_cols = [
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
    # 取出這 10 天(10x10)的數值
    raw_features = df_10[needed_cols].values  # shape (10, 10)

    # 2) 用同一個 scaler transform
    scaled_features = scaler.transform(raw_features)  # shape (10,10)

    # 3) 變成 numpy 再變 torch tensor，並加 batch 維度
    X_lstm = np.array(scaled_features, dtype=np.float32)  # [10,10]
    X_lstm = X_lstm[np.newaxis, :, :]  # [1,10,10]
    X_lstm = torch.tensor(X_lstm, device=device)  # 搬到 device

    # 4) CNN 圖片處理
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img)  # shape [3,64,64]
    X_cnn = img_tensor.unsqueeze(0).to(device)  # [1,3,64,64]

    return X_lstm, X_cnn


def main_inference_example():
    """
    供你直接執行的主程式範例:
    1) 讀取 stock_df.pkl
    2) 讀取 scaler
    3) 讀取某個股票的最佳模型權重
    4) 在該 df 裡取最後 10 筆，當作推論樣本
    5) 進行預測
    """
    # 1) 讀取 pkl
    with open("stock_df.pkl", "rb") as f:
        stock_data = pickle.load(f)  # dict, key=股票代號, value=df

    # 2) 讀取 scaler
    #   (假設你在訓練完後有把 MinMaxScaler 存成 "my_scaler.pkl")
    scaler = load_scaler("scaler.pkl")

    # 3) 指定要測試的股票 & 權重檔
    stock_id = "1773"
    model_path = f"{stock_id}_best_lstm_cnn_model.pth"
    # 若每檔股票的最佳參數都不一樣，這裡也要對應改成對應該檔的超參數

    # 以固定超參數為例
    model = load_model(
        model_path=model_path,
        lstm_hidden_dim=32,
        lstm_num_layers=1,
        cnn_output_channels=16,
        cnn_kernel_size=3,
        cnn_fc_size=64,
    )

    # 4) 取出該股票的 df，取最後 10 天 + 最後一天的 image_path
    df = stock_data[stock_id]
    # 假設 df 已經 sort_index()，且至少有 10 天可用
    # 最後一筆資料會是 df.iloc[-1]，那天對應的圖路徑是 df.iloc[-1]['image_path']
    # 你希望用「前 9 天 + 當天」一共 10 筆特徵 → 預測「隔天」(外推)?
    # 這邊先示範「用最後 10 天」來看當天預測 (同程式結構)
    # 例: df_10 = df.iloc[-10:]  # shape (10, ...)
    # image_path 用最後一天的

    df_10 = df.iloc[-10:]  # 取最後 10 天
    image_path = df_10.iloc[-1]['image_path']  # 最後一天的圖檔路徑

    # 5) 準備 LSTM/CNN 輸入
    X_lstm, X_cnn = prepare_single_inference(df_10, scaler, image_path)

    # 6) 推論
    model.eval()
    with torch.no_grad():
        pred_tensor = model(X_lstm, X_cnn)  # shape [1,1]
        pred_val = pred_tensor.item()

    # 7) 如你訓練時是一併縮放了 Close, MA_5 等 10 個欄位
    #    則 pred_val 此時是 0~1 區間的預測值(對應Close那個欄位)。
    #    想反轉回原始股價，要在同一個 scaler 做 inverse_transform。
    #    不過因為 scaler 同時 scale 10 個特徵，我們要把預測值放在 "Close" 那個位置
    #    其他欄位先隨便填 0 或 0.5 皆可，只要位置對齊即可。
    #    例如:
    dummy = np.zeros((1, LSTM_INPUT_DIM), dtype=np.float32)  # shape [1,10]
    #  假設第 0 個欄位對應 Close (要看你 fit_transform 的時候順序!)
    #  如果 "Close" 是 scaled_features 裡的第0欄，就放在 dummy[0,0]
    #  如果 "Close" 是第 0 個欄位，那 pred_val 就放 dummy[0,0]。
    dummy[0, 0] = pred_val  # index=0 for Close

    real_val = scaler.inverse_transform(dummy)[0, 0]
    # 上面這行會把 dummy 從 [0,1] 映射回原始尺度，取 [0,0] 就是對應Close

    print(f"【{stock_id}】模型預測 (縮放後): {pred_val:.4f}")
    print(f"【{stock_id}】模型預測 (反轉回原始): {real_val:.2f}")


if __name__ == "__main__":
    main_inference_example()
