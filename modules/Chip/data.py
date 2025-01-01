import datetime
from time import sleep
from typing import Optional

import pandas as pd
from FinMind.data import DataLoader

import config

sleep_time = 2


class ChiplData:
    def __init__(self, api_token):
        """
        初始化 ChiplData，並完成登入

        :param api_token: API 金鑰
        """
        self.api = DataLoader()
        self.api.login_by_token(api_token=api_token)

    def get_chip_df(
        self, stock_id: str, start_date: str, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        抓取指定股票的籌碼面資料

        :param stock_id: 股票代碼
        :param start_date: 起始日期 (YYYY-MM-DD)
        :param end_date: 結束日期 (YYYY-MM-DD)，預設為 start_date 的一個月後
        :return: 包含所有籌碼面資料的 DataFrame
        """

        # 如果沒有給定結束日期，預設為start_date的一個月後
        if end_date is None:
            date_obj = datetime.datetime.strptime(start_date, "%Y-%m-%d") + datetime.timedelta(
                days=30
            )
            end_date = date_obj.strftime("%Y-%m-%d")

        # 個股融資融券表
        mp = self.api.taiwan_stock_margin_purchase_short_sale(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date,
        )

        sleep(sleep_time)

        # 法人買賣表
        ii = self.api.taiwan_stock_institutional_investors(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date,
        )

        sleep(sleep_time)

        # 外資持股表
        sh = self.api.taiwan_stock_shareholding(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date,
        )

        sleep(sleep_time)

        # 合併所有表格
        df = mp.merge(ii, on="date", how="outer")
        df = df.merge(sh, on="date", how="outer")

        return df

    def summarize_chip_data(self, df: pd.DataFrame, stock_id: str) -> str:
        """
        將籌碼面 DataFrame 轉換為自然語言格式，並最小化重複資料輸出。

        根據實際抓到的欄位名稱進行判斷 (不更動原始 df 欄位)：

        1. 融資融券 (來自 taiwan_stock_margin_purchase_short_sale)，常見欄位：
           - MarginPurchaseTodayBalance    (融資餘額)
           - ShortSaleTodayBalance         (融券餘額)
           - MarginPurchaseBuy             (當日融資買進)
           - MarginPurchaseCashRepayment   (當日融資償還)
           - MarginPurchaseSell            (有些人也會拿這當作賣出/償還欄位，視需求而定)
           - ShortSaleSell                 (當日融券賣出)
           - ShortSaleBuy                  (當日融券買進/回補)
           - ShortSaleCashRepayment        (另一種融券回補記法)

        2. 三大法人買賣 (來自 taiwan_stock_institutional_investors)，FinMind 常用結構：
           - name (外資 / 投信 / Dealer_self / ... )
           - buy, sell
           （若要對應「外資買、外資賣」等，須判斷 row["name"] == "Foreign_Investor" 再去看 buy / sell）

        3. 外資持股 (來自 taiwan_stock_shareholding)，常見欄位：
           - ForeignInvestmentShares       (外資持股張數)
           - ForeignInvestmentSharesRatio  (外資持股比率)
           （FinMind 中為 `ForeignInvestmentShares`, `ForeignInvestmentSharesRatio`）

        :param df: pandas.DataFrame，包含上述欄位（可能部分欄位缺失）的籌碼面資料
        :param stock_id: 股票代碼，用於描述中
        :return: str，格式化的自然語言文字描述
        """

        descriptions = set()

        # 先以日期去重，避免同一個 date 重複描述
        df = df.drop_duplicates(subset=["date"])

        for _, row in df.iterrows():
            date_str = row["date"]

            # === 1. 融資融券 ===
            # 融資餘額 -> MarginPurchaseTodayBalance
            if "MarginPurchaseTodayBalance" in df.columns and pd.notna(
                row["MarginPurchaseTodayBalance"]
            ):
                descriptions.add(
                    f"截至 {date_str}，股票代碼 {stock_id} 的融資餘額為 {row['MarginPurchaseTodayBalance']:,} 張。"
                )
            # 融券餘額 -> ShortSaleTodayBalance
            if "ShortSaleTodayBalance" in df.columns and pd.notna(row["ShortSaleTodayBalance"]):
                descriptions.add(
                    f"截至 {date_str}，股票代碼 {stock_id} 的融券餘額為 {row['ShortSaleTodayBalance']:,} 張。"
                )
            # 當日融資買進 -> MarginPurchaseBuy
            if "MarginPurchaseBuy" in df.columns and pd.notna(row["MarginPurchaseBuy"]):
                descriptions.add(f"{date_str} 當日融資買進 {row['MarginPurchaseBuy']:,} 張。")
            # 當日融資償還 -> MarginPurchaseCashRepayment 或 MarginPurchaseSell
            if "MarginPurchaseCashRepayment" in df.columns and pd.notna(
                row["MarginPurchaseCashRepayment"]
            ):
                descriptions.add(
                    f"{date_str} 當日融資償還 {row['MarginPurchaseCashRepayment']:,} 張。"
                )
            elif "MarginPurchaseSell" in df.columns and pd.notna(row["MarginPurchaseSell"]):
                descriptions.add(f"{date_str} 當日融資償還 {row['MarginPurchaseSell']:,} 張。")

            # 當日融券賣出 -> ShortSaleSell
            if "ShortSaleSell" in df.columns and pd.notna(row["ShortSaleSell"]):
                descriptions.add(f"{date_str} 當日融券賣出 {row['ShortSaleSell']:,} 張。")

            # 當日融券回補 -> ShortSaleBuy, 或 ShortSaleCashRepayment
            if "ShortSaleBuy" in df.columns and pd.notna(row["ShortSaleBuy"]):
                descriptions.add(f"{date_str} 當日融券回補 {row['ShortSaleBuy']:,} 張。")
            elif "ShortSaleCashRepayment" in df.columns and pd.notna(row["ShortSaleCashRepayment"]):
                descriptions.add(f"{date_str} 當日融券回補 {row['ShortSaleCashRepayment']:,} 張。")

            # === 2. 三大法人買賣 ===
            # FinMind 預設: name = "Foreign_Investor" / "Investment_Trust" / "Dealer_Self" / "Dealer_Hedging" / ...
            # buy / sell = 當天買賣張數
            # 因為三大法人資料在同一天，會以多筆 (多列) 形式出現，如 name="Foreign_Investor"、name="Investment_Trust" 等
            # 若單純掃 df，每一列都會印一次。可以依需求看要怎麼去重或合併敘述。

            # 範例：若 name 為 "Foreign_Investor" 且 buy/sell 不為 NaN
            if (
                "name" in df.columns
                and "buy" in df.columns
                and pd.notna(row["buy"])
                and row["name"] == "Foreign_Investor"
            ):
                descriptions.add(f"{date_str} 外資買進 {row['buy']:,} 張。")
            if (
                "name" in df.columns
                and "sell" in df.columns
                and pd.notna(row["sell"])
                and row["name"] == "Foreign_Investor"
            ):
                descriptions.add(f"{date_str} 外資賣出 {row['sell']:,} 張。")

            # 投信
            if (
                "name" in df.columns
                and "buy" in df.columns
                and pd.notna(row["buy"])
                and row["name"] == "Investment_Trust"
            ):
                descriptions.add(f"{date_str} 投信買進 {row['buy']:,} 張。")
            if (
                "name" in df.columns
                and "sell" in df.columns
                and pd.notna(row["sell"])
                and row["name"] == "Investment_Trust"
            ):
                descriptions.add(f"{date_str} 投信賣出 {row['sell']:,} 張。")

            # 自營商 (有些分 Dealer_Self、Dealer_Hedging 等)
            if (
                "name" in df.columns
                and "buy" in df.columns
                and pd.notna(row["buy"])
                and row["name"] in ["Dealer_Self", "Dealer_Hedging"]
            ):
                descriptions.add(f"{date_str} 自營商買進 {row['buy']:,} 張。")

            if (
                "name" in df.columns
                and "sell" in df.columns
                and pd.notna(row["sell"])
                and row["name"] in ["Dealer_Self", "Dealer_Hedging"]
            ):
                descriptions.add(f"{date_str} 自營商賣出 {row['sell']:,} 張。")

            # === 3. 外資持股 ===
            # ForeignInvestmentShares (外資持股張數)
            if "ForeignInvestmentShares" in df.columns and pd.notna(row["ForeignInvestmentShares"]):
                descriptions.add(
                    f"截至 {date_str}，外資持有 {row['ForeignInvestmentShares']:,} 張 {stock_id}。"
                )

            # ForeignInvestmentSharesRatio (外資持股比率)
            if "ForeignInvestmentSharesRatio" in df.columns and pd.notna(
                row["ForeignInvestmentSharesRatio"]
            ):
                descriptions.add(f"外資持股比率約為 {row['ForeignInvestmentSharesRatio']:.2f}%。")

        return "\n".join(descriptions)


if __name__ == "__main__":
    # 測試
    cdata = ChiplData(config.FINMIND_API_TOKEN)
    df = cdata.get_chip_df("2330", "2023-01-01")
    print(df.tail(10))

    # 觀察抓到的 df 狀況
    print("df shape:", df.shape)
    print(df.head(10))
    print(df.columns)

    # 生成描述
    text_summary = cdata.summarize_chip_data(df, "2330")
    print(text_summary)
    print(len(text_summary.split("\n")))
