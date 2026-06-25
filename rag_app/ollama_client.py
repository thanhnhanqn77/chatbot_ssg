from __future__ import annotations

from typing import Any

import requests


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 120,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def tags(self) -> dict[str, Any]:
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                headers=self._headers(),
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama chat request failed: {exc}") from exc

        data = response.json()
        message = data.get("message") or {}
        content = message.get("content")
        if not content:
            raise OllamaError(f"Ollama returned an empty response: {data}")
        return content
