import os
from openai import OpenAI


class DeepSeek:
    def __init__(self, api_key: str = "密钥", base_url: str = "https://api.deepseek.com"):
        """
        初始化 DeepSeek。

        :param api_key: DeepSeek API 密钥
        :param base_url: DeepSeek API 的基础 URL，默认为 "https://api.deepseek.com"
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def call(
            self,
            prompt: str,
            model: str = "deepseek-chat",  # DeepSeek 的默认模型
            # model: str = "deepseek-reasoner",
            temperature: float = 1.0,
            max_tokens: int = 30,
            top_p: float = 1.0,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            logprobs: bool = False,
            n: int = 1,
    ):
        return self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            n=n,
        )
