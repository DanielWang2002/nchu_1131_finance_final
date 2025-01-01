import pickle
from datetime import datetime, timedelta

import pandas as pd

import config
from modules.Fundamental import FundamentalData
from modules.Chip import ChiplData
from modules.LLM import GeminiAPI

fdata = FundamentalData(config.FINMIND_API_TOKEN)
cdata = ChiplData(config.FINMIND_API_TOKEN)
gapi = GeminiAPI(config.GEMINI_API_TOKEN)

# load stock list
with open('stock_df.pkl', 'rb') as f:
    stock_df = pickle.load(f)

# 根據stock_df(dict)的key取得股票代碼，讀取模型的日期、真實、預測結果
stock_ids = list(stock_df.keys())
stock_results = {}
for stock_id in stock_ids:
    with open(f'{stock_id}_results.pkl', 'rb') as f:
        stock_results[stock_id] = pickle.load(f)  # df: Date, Actual, Predicted

# 根據 stock_resultes 的 Date 取得前一個月內的基本面、籌碼面資料
for stock_id in stock_ids:
    stock_result = stock_results[stock_id]
    stock_result['Date'] = pd.to_datetime(stock_result['Date'])
    stock_result = stock_result.set_index('Date')

    # 遍歷每一天，每10天執行一次
    for i, date in enumerate(stock_result.index):
        if i % 10 == 0:
            # 取得前一個月的日期
            last_month = date - timedelta(days=90)
            # 取得前一個月的基本面、籌碼面資料
            fundamental_data = fdata.get_fundamental_df(stock_id, last_month.strftime('%Y-%m-%d'))
            print(fundamental_data)
            chip_data = cdata.get_chip_df(stock_id, last_month.strftime('%Y-%m-%d'))

            # 將基本面、籌碼面資料加入 stock_result
            stock_result.loc[date, 'fundamental_data'] = fundamental_data
            stock_result.loc[date, 'chip_data'] = chip_data
            print(stock_result.loc[date])
    break
