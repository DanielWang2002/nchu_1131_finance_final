import os
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import pickle
import talib
from tqdm import tqdm
from FinMind.data import DataLoader
from concurrent.futures import ProcessPoolExecutor
import gc

# 股票代號列表
stock_list = ["1773", "2449", "3529", "5292", "6679"]

# 初始化 DataLoader
api = DataLoader()

# 下載股票數據
stock_df = {}
for stock in stock_list:
    df = api.taiwan_stock_daily(stock_id=stock, start_date="2023-01-01")

    # 轉為適用 mplfinance 的欄位格式
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.rename(
        columns={
            'open': 'Open',
            'max': 'High',
            'min': 'Low',
            'close': 'Close',
            'Trading_Volume': 'Volume',
        },
        inplace=True,
    )
    stock_df[stock] = df

# 建立儲存圖片的資料夾
if not os.path.exists('./k_line'):
    os.makedirs('./k_line')

# look_back = 10，代表「前 10 天」的 K 棒
look_back = 10


def add_technical_indicators(df):
    """
    對 df 計算一些常用技術指標。
    """
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


def process_stock(args):
    stock_id, df = args
    df = df.copy()  # 不要直接改到原始
    df = add_technical_indicators(df)  # 先加技術指標，也可以放後面

    image_paths = []

    # 設定繪圖區
    fig, ax = plt.subplots(figsize=(10, 6))

    # 用 look_back (10) 來控制迴圈
    # i 表示 "當天 (要預測的那天)"，
    # 圖片只畫 "前 10 天" => df.iloc[i - 10 : i]
    for i in tqdm(range(look_back, len(df)), desc=f"Processing {stock_id}"):
        # === 改動處 ===
        # 原本: window_data = df.iloc[i - look_back : i + 1]
        # 現在: 只拿到 i - 10 ~ i-1，共 10 根 K 棒，不含當天
        window_data = df.iloc[i - look_back : i]

        ax.clear()
        mpf.plot(window_data, type='candle', style='charles', volume=False, ax=ax)
        ax.axis('off')

        filename = f"{stock_id}_{i}.png"
        filepath = f"./k_line/{filename}"
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=100)
        image_paths.append(filepath)

        gc.collect()

    plt.close(fig)

    # 現在 i 從 look_back 到 len(df)-1
    # 因此 image_paths 長度為 len(df) - look_back
    # 要把這些路徑，對應到 df.index[i] (第 i 天)
    image_path_series = pd.Series(
        data=image_paths, index=df.index[look_back:]  # 與 image_paths 個數相同
    )

    # 將這個欄位放回 df，代表「第 i 天所對應的圖路徑」畫的是 (i-10 ~ i-1)
    df.loc[image_path_series.index, 'image_path'] = image_path_series

    gc.collect()
    return stock_id, df


with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_stock, stock_df.items()))

for stock_id, processed_df in results:
    stock_df[stock_id] = processed_df

# 最後針對每支股票把有 NaN 的行刪除 (看需求是否一定要這樣)
for stock_id, df in stock_df.items():
    stock_df[stock_id] = df.dropna()

# 簡單檢查其中一檔的 image_path
print(stock_df['1773']['image_path'].head())

# 將最終結果存到 pkl 檔中
with open('stock_df.pkl', 'wb') as f:
    pickle.dump(stock_df, f)
