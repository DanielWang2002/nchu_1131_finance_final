import google.generativeai as genai


class GeminiAPI:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        """
        初始化 GeminiAPI 類，設置 API 金鑰及模型名稱。

        :param api_key: str，Google Gemini API 的金鑰。
        :param model_name: str，模型名稱，預設為 "gemini-1.5-pro"。
        """
        self.api_key = api_key
        self.model_name = model_name
        self._configure_api()

    def _configure_api(self):
        """
        設定 API 金鑰。
        """
        genai.configure(api_key=self.api_key)

    def generate_content(self, prompt: str) -> str:
        """
        使用 Gemini API 生成內容。

        :param prompt: str，用於生成內容的提示語。
        :return: str，生成的內容。
        """
        model = genai.GenerativeModel(model_name=self.model_name)
        response = model.generate_content(prompt)
        return response.text


# 測試範例
if __name__ == "__main__":
    # 替換為您的 API 金鑰
    api_key = "AIzaSyDnx0ZF42PItw_lmMNfs1fdIVoprGTDx1g"
    gemini_api = GeminiAPI(api_key)

    # 使用提示語生成內容
    prompt = "你好，使用繁體中文解釋什麼是深度學習、機器學習的差別"
    content = gemini_api.generate_content(prompt)
    print("生成的內容：")
    print(content)
