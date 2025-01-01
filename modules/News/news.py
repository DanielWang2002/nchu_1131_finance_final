import json
from datetime import datetime
import urllib.request

import pandas as pd


class GNews:
    def __init__(self, api_key: str):
        """
        初始化 GNewsAPI 類，設置 API 金鑰。

        :param api_key: str，GNews API 的金鑰。
        """
        self.api_key = api_key
        self.base_url = "https://gnews.io/api/v4/search"

    def fetch_news(
        self, query: str, lang: str = "tw", country: str = "tw", max_results: int = 10
    ) -> pd.DataFrame:
        """
        從 GNews API 抓取新聞資料，並返回一個 DataFrame。

        :param query: str，新聞關鍵字。
        :param lang: str，新聞語言，預設為英文 ("en")。
        :param country: str，新聞國家，預設為美國 ("us")。
        :param max_results: int，最多返回的新聞數量，預設為 10。
        :return: pd.DataFrame，包含新聞標題、描述、日期、內容等。
        """
        url = f"{self.base_url}?q={query}&lang={lang}&country={country}&max={max_results}&apikey={self.api_key}"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
            articles = data.get("articles", [])

        # 將每篇新聞的資料整理成 DataFrame 格式
        news_data = []
        for article in articles:
            news_data.append(
                {
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "publishedAt": datetime.strptime(
                        article.get("publishedAt", ""), "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "content": article.get("content", ""),
                }
            )

        return pd.DataFrame(news_data)

    def news_to_natural_language(self, df: pd.DataFrame) -> str:
        """
        將新聞的 DataFrame 轉換為自然語言描述。

        :param df: pd.DataFrame，必須包含以下欄位：
            - title: 新聞標題
            - description: 新聞描述
            - publishedAt: 發佈日期 (datetime 格式)
            - content: 新聞內容
        :return: str，格式化的自然語言描述。
        """
        descriptions = []
        for _, row in df.iterrows():
            published_date = row["publishedAt"].strftime("%Y年%m月%d日 %H:%M")
            descriptions.append(
                f"於 {published_date} 發佈的新聞：{row['title']}。摘要如下：{row['description']}。\n"
                f"內容為：{row['content'][:200]}..."
            )

        return "\n\n".join(descriptions)


# 測試範例
if __name__ == "__main__":
    api_key = "819599a52833bb05cb624f44a64ac86e"
    gnews = GNews(api_key)

    # 查詢關鍵字 "example"，語言為英文
    df = gnews.fetch_news(query="2330", max_results=100)

    # 顯示 DataFrame 的前幾筆資料
    print(df.head())
    print(df.shape)
    df.to_csv('news.csv', index=False)
