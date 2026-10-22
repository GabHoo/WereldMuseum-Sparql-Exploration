"""
HostedChatModel — a chat model behind an OpenAI-compatible HTTP API.

Most hosting services (OpenAI, Groq, Together, Mistral, OpenRouter, the
Hugging Face router, ...) accept the same request:
    POST <base_url>/chat/completions   {"model": ..., "messages": [...]}
so one class covers them; only the base URL, key and model name differ.

Configure via .env:
    CHAT_MODEL=hosted
    CHAT_API_BASE_URL=https://api.openai.com/v1
    CHAT_API_KEY=...
    CHAT_API_MODEL=gpt-4o-mini
"""
import json
import urllib.error
import urllib.request

from modules.base import ChatModel, ChatModelError


class HostedChatModel(ChatModel):
    def __init__(self, base_url: str, api_key: str, model: str,
                 max_tokens: int = 512, timeout: int = 60):
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout

    def chat(self, messages: list[dict]) -> str:
        if not self._key:
            raise ChatModelError("No API key set. Add CHAT_API_KEY to .env and restart the app.")
        body = json.dumps({"model": self._model, "messages": messages,
                           "max_tokens": self._max_tokens}).encode("utf-8")
        req = urllib.request.Request(self._url, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._key}",
        })
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise ChatModelError(f"The chat service answered {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise ChatModelError(f"Could not reach the chat service at {self._url}: {e.reason}") from e
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as e:
            raise ChatModelError(f"Unexpected reply from the chat service: {str(data)[:300]}") from e
