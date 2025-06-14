import os
import openai


class GPT4:
    def __init__(self):
        self.client = openai.OpenAI(api_key="密钥")

    def call(self,
             prompt,
             model="gpt-4o-mini",
             temperature=1.,
             max_tokens=30,
             top_p=1.,
             frequency_penalty=0,
             presence_penalty=0,
             logprobs=False,
             n=1):
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
