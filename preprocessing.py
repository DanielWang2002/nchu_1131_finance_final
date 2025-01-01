import os
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import pickle
import talib
from tqdm import tqdm
from FinMind.data import DataLoader
import gc

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


def add_technical_indicators(df):
    """增加技術指標到 DataFrame"""
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()

    df['slowk'], df['slowd'] = talib.STOCH(
        df['High'],
        df['Low'],
        df['Close'],
        fastk_period=9,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)

    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
        df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    return df


# 處理每個股票
for stock_id, df in stock_df.items():
    # 資料預處理
    df = df.copy()
    df.index = pd.to_datetime(df.index)
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
        plt.close(fig)  # 關閉圖表資源

        # 更新 DataFrame 的圖片路徑
        df.loc[df.index[i], 'image_path'] = filepath

        # 強制垃圾回收
        gc.collect()

    # 新增技術指標
    df = add_technical_indicators(df)

    # 更新原始 DataFrame
    stock_df[stock_id] = df

    # 再次進行垃圾回收
    gc.collect()

print(stock_df['1773'].head())

# 保存 DataFrame
with open('stock_df.pkl', 'wb') as f:
    pickle.dump(stock_df, f)
