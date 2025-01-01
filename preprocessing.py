import os
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import pickle
import talib
from tqdm import tqdm
from FinMind.data import DataLoader

# 股票代號列表
stock_list = ["1773", "2449", "3529", "5292", "6679"]

# 初始化 DataLoader
api = DataLoader()

# 下載股票數據
stock_df = {}
for stock in stock_list:
    stock_df[stock] = api.taiwan_stock_daily(stock_id=stock, start_date="2020-01-01")

# 建立儲存圖片的資料夾
if not os.path.exists('./k_line'):
    os.makedirs('./k_line')

# 設定參數
window_size = 11  # K棒總數
center_position = window_size // 2  # 中心位置

# 處理每個股票
for stock_id, df in stock_df.items():
    # 資料預處理
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # 重新命名欄位以符合 mplfinance 要求
    df = df.rename(
        columns={
            'open': 'Open',
            'max': 'High',
            'min': 'Low',
            'close': 'Close',
            'Trading_Volume': 'Volume',
        }
    )

    # 新增圖片檔名欄位
    df['image_path'] = None

    # 對每一天生成圖片
    for i in tqdm(range(center_position, len(df) - center_position)):
        # 取得滑動視窗的資料
        window_data = df.iloc[i - center_position : i + center_position + 1].copy()

        # 設定圖表樣式
        fig, ax = plt.subplots(figsize=(10, 6))
        mpf.plot(window_data, type='candle', style='charles', volume=False, ax=ax)

        # 移除座標軸和標籤
        ax.axis('off')

        # 設定檔名
        filename = f"{stock_id}_{i}.png"
        filepath = f"./k_line/{filename}"

        # 儲存圖片
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

        # 更新 DataFrame 的圖片路徑
        df.loc[df.index[i], 'image_path'] = filepath

    # 更新原始 DataFrame
    stock_df[stock_id] = df


# 增加技術指標的函數
def add_technical_indicators(df):
    # 移動平均線 (MA)
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()

    # KD 指標
    df['slowk'], df['slowd'] = talib.STOCH(
        df['High'],  # 高價
        df['Low'],  # 低價
        df['Close'],  # 收盤價
        fastk_period=9,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )

    # 相對強弱指數 (RSI)
    df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)

    # 移動平均匯聚背離指標 (MACD)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
        df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    return df


# 處理每個股票
for stock_id, df in stock_df.items():
    # 資料預處理
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # 重新命名欄位以符合 mplfinance 要求
    df = df.rename(
        columns={
            'open': 'Open',
            'max': 'High',  # 修正: 'max' 改為 'High'
            'min': 'Low',  # 修正: 'min' 改為 'Low'
            'close': 'Close',
            'Trading_Volume': 'Volume',
        }
    )

    # 新增圖片檔名欄位
    df['image_path'] = None

    # 對每一天生成圖片
    for i in tqdm(range(center_position, len(df) - center_position)):
        # 取得滑動視窗的資料
        window_data = df.iloc[i - center_position : i + center_position + 1].copy()

        # 設定圖表樣式
        fig, ax = plt.subplots(figsize=(10, 6))
        mpf.plot(window_data, type='candle', style='charles', volume=False, ax=ax)

        # 移除座標軸和標籤
        ax.axis('off')

        # 設定檔名
        filename = f"{stock_id}_{i}.png"
        filepath = f"./k_line/{filename}"

        # 儲存圖片
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

        # 更新 DataFrame 的圖片路徑
        df.loc[df.index[i], 'image_path'] = filepath

    # 新增技術指標
    df = add_technical_indicators(df)

    # 更新原始 DataFrame
    stock_df[stock_id] = df

print(stock_df['1773'].head())

# 保存 DataFrame
with open('stock_df.pkl', 'wb') as f:
    pickle.dump(stock_df, f)
