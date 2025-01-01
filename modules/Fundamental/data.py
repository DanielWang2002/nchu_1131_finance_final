import datetime
from time import sleep
from typing import Optional

import pandas as pd
from FinMind.data import DataLoader

sleep_time = 2


class FundamentalData:
    def __init__(self, api_token):
        """
        初始化 FundamentalData，並完成登入

        :param api_token: API 金鑰
        """
        self.api = DataLoader()
        self.api.login_by_token(api_token=api_token)

    def get_fundamental_df(
        self, stock_id: str, start_date: str, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        抓取指定股票的基本面資料

        :param stock_id: 股票代碼
        :param start_date: 起始日期 (YYYY-MM-DD)
        :param end_date: 結束日期 (YYYY-MM-DD)，預設為 start_date 的一個月後
        :return: 包含所有基本面資料的 DataFrame
        """

        # 如果沒有給定結束日期，預設為start_date的一個月後
        if end_date is None:
            date_obj = datetime.datetime.strptime(start_date, "%Y-%m-%d") + datetime.timedelta(
                days=90
            )
            end_date = date_obj.strftime("%Y-%m-%d")

        # 綜合損益表
        fs = self.api.taiwan_stock_financial_statement(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date,
        )
        print("fs:", fs)
        sleep(sleep_time)

        # 資產負債表
        bs = self.api.taiwan_stock_balance_sheet(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date,
        )
        print("bs:", bs)
        sleep(sleep_time)

        # 現金流量表
        cf = self.api.taiwan_stock_cash_flows_statement(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date,
        )
        print("cf:", cf)
        sleep(sleep_time)

        # 股利政策表
        di = self.api.taiwan_stock_dividend(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date,
        )
        print("di:", di)
        sleep(sleep_time)

        # 除權除息結果表
        dr = self.api.taiwan_stock_dividend_result(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date,
        )
        print("dr:", dr)
        sleep(sleep_time)

        # 月營收表
        mr = self.api.taiwan_stock_month_revenue(
            stock_id=stock_id,
            start_date=start_date,
            end_date=end_date,
        )
        print("mr:", mr)
        sleep(sleep_time)

        # 合併所有有資料的表格
        data_frames = [fs, bs, cf, di, dr, mr]

        # 過濾掉空的 DataFrame
        data_frames = [df for df in data_frames if not df.empty]

        # 如果 data_frames 不為空，則依序合併所有表格
        if data_frames:
            merged_df = data_frames[0]
            for next_df in data_frames[1:]:
                merged_df = merged_df.merge(next_df, on=['date', 'stock_id'], how='outer')
        else:
            # 如果所有表格皆為空，則回傳一個空的 DataFrame
            merged_df = pd.DataFrame()

        return merged_df

    def get_quarter(self, date: datetime.datetime) -> datetime.datetime:
        """
        根據日期，找出前一季的日期，每一季的日期為 3/31、6/30、9/30、12/31。

        :param date: 日期
        :return: 前一季的日期
        """
        if date.month <= 3:
            return datetime.datetime(date.year - 1, 12, 31)
        elif date.month <= 6:
            return datetime.datetime(date.year, 3, 31)
        elif date.month <= 9:
            return datetime.datetime(date.year, 6, 30)
        else:
            return datetime.datetime(date.year, 9, 30)

    def summarize_financial_data(self, df: pd.DataFrame) -> str:
        """
        將財務資料的 DataFrame 轉換為自然語言格式，並最小化重複資料輸出。

        :param df: pandas.DataFrame，包含財務資料，必須有以下欄位：
        :param date: 日期
        :param stock_id: 股票代碼
        :param type_x, value_x, origin_name_x: 第一組財務項目類型、數值和描述
        :param type_y, value_y, origin_name_y: 第二組財務項目類型、數值和描述
        :param type, value, origin_name: 第三組財務項目類型、數值和描述
        :return: str，格式化的自然語言文字描述。
        """
        # 將輸出的文字存儲在一個集合中以去除重複的描述
        descriptions = set()

        # 創建副本，避免影響原始 DataFrame
        df = df.drop_duplicates(subset=["origin_name_x", "origin_name_y", "origin_name"])

        for _, row in df.iterrows():
            # 描述第一組財務數據
            descriptions.add(
                f"截至 {row['date']}，股票代碼 {row['stock_id']} 的 {row['origin_name_x']} 為 {row['value_x']:.2f} 元。"
            )
            # 描述第二組財務數據
            descriptions.add(f"其 {row['origin_name_y']} 為 {row['value_y']:.2f} 元。")
            # 描述第三組財務數據
            descriptions.add(f"投資活動中，{row['origin_name']} 為 {row['value']:.2f} 元。")

        # 將描述合併為完整文字輸出
        return "\n".join(descriptions)
