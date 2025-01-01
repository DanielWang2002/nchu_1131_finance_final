import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
from PIL import Image
from model import LSTM_CNN_Model

# 自動判斷設備
if torch.cuda.is_available():
    device = torch.device("cuda")  # 如果有 CUDA，使用 GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # 如果是 Apple Silicon，使用 MPS
else:
    device = torch.device("cpu")  # 默認使用 CPU

print(f"Using device: {device}")

# 超參數
LSTM_INPUT_DIM = 10  # LSTM 特徵數
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
CNN_INPUT_CHANNELS = 3  # 圖片通道數（RGB）
CNN_OUTPUT_CHANNELS = 32
CNN_KERNEL_SIZE = 3
CNN_FC_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
IMG_SIZE = (64, 64)

# 讀取模型
model = LSTM_CNN_Model(
    lstm_input_dim=LSTM_INPUT_DIM,
    lstm_hidden_dim=LSTM_HIDDEN_DIM,
    lstm_num_layers=LSTM_NUM_LAYERS,
    cnn_input_channels=CNN_INPUT_CHANNELS,
    cnn_output_channels=CNN_OUTPUT_CHANNELS,
    cnn_kernel_size=CNN_KERNEL_SIZE,
    cnn_fc_size=CNN_FC_SIZE,
).to(device)

# 損失函數與優化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 讀取股票數據
with open("stock_df.pkl", "rb") as f:
    stock_data = pickle.load(f)  # stock_data 是 dict，key 是股票代號，value 是 dataframe

# print(stock_data["1773"].columns)


# 圖片轉換
transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ]
)


# 數據準備
def prepare_data(stock_data, window_size=30):
    X_lstm, X_cnn, y = [], [], []
    scaler = MinMaxScaler()
    for stock_id, df in stock_data.items():
        df = df.dropna()
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
        labels = df['Close'].values
        img_paths = df['image_path'].values

        for i in range(window_size, len(df) - 1):
            X_lstm.append(scaled_features[i - window_size : i])
            img = Image.open(img_paths[i]).convert("RGB")
            img = transform(img)
            X_cnn.append(img)
            y.append(labels[i + 1])

    X_lstm = np.array(X_lstm)
    X_cnn = torch.stack(X_cnn)  # 圖片轉為 tensor
    y = np.array(y)

    return X_lstm, X_cnn, y


# 構建數據
X_lstm, X_cnn, y = prepare_data(stock_data)

# 分割訓練集與測試集
X_lstm_train, X_lstm_test, X_cnn_train, X_cnn_test, y_train, y_test = train_test_split(
    X_lstm, X_cnn, y, test_size=0.2, random_state=42
)

# 將數據轉為 tensor
X_lstm_train = torch.tensor(X_lstm_train, dtype=torch.float32).to(device)
X_lstm_test = torch.tensor(X_lstm_test, dtype=torch.float32).to(device)
X_cnn_train = X_cnn_train.to(device)
X_cnn_test = X_cnn_test.to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

# 訓練模型
print("Training model...")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    # 前向傳播
    outputs = model(X_lstm_train, X_cnn_train)
    loss = criterion(outputs, y_train)

    # 反向傳播
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f}")

# 評估模型
print("Evaluating model...")
model.eval()
with torch.no_grad():
    predictions = model(X_lstm_test, X_cnn_test)
    test_loss = criterion(predictions, y_test).item()
    y_test = y_test.cpu().numpy()
    predictions = predictions.cpu().numpy()

# 績效指標
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f"Test Loss: {test_loss:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# 保存模型
torch.save(model.state_dict(), "lstm_cnn_model.pth")
print("Model saved as lstm_cnn_model.pth")
