import requests
import asyncio
from typing import List

from .exceptions import VLLMGenerationException

class vLLMClient:
    def __init__(self, model: str, base_url: str, generation_args: dict, api_key: str = 'sk-no-key-needed'):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        self.generation_args = generation_args

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _chat_single(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            **self.generation_args
        }
        try:
            response = requests.post(f"{self.base_url}", json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()['choices'][0]['text'].strip()
        except:
            return VLLMGenerationException("VLLM generation failed.")


    async def chat(self, prompts: List[str]) -> List[str]:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, self._chat_single, prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)