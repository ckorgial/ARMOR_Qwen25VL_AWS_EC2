from openai import OpenAI
from typing import List, Dict, Any, Optional

class VLLMClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def chat_vision(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        kwargs = dict(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except TypeError:
            # Some vLLM builds may not support response_format; retry without it
            kwargs.pop("response_format", None)
            resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""