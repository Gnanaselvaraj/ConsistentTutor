"""
llm.py: LLM interface and wrappers with streaming support
"""
from typing import Any, Generator
import requests
import json

class OllamaLLM:
    def __init__(self, model: str = "llama3", temperature: float = 0.2, url: str = "http://localhost:11434/api/generate", num_predict: int = 512):
        self.url = url
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict  # Max tokens to generate

    def invoke(self, prompt: str, max_tokens: int = None) -> str:
        """Non-streaming call for backward compatibility"""
        r = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "num_predict": max_tokens or self.num_predict,
                "stream": False
            },
            timeout=180
        )
        r.raise_for_status()
        return r.json()["response"].strip()
    
    def stream(self, prompt: str, max_tokens: int = None) -> Generator[str, None, None]:
        """Streaming generator that yields tokens as they're generated"""
        try:
            r = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "num_predict": max_tokens or self.num_predict,
                    "stream": True
                },
                timeout=180,
                stream=True
            )
            r.raise_for_status()
            
            for line in r.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            yield f"[Error: {str(e)}]"
