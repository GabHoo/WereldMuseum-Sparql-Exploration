"""
NebulaChatModel — the VU Nebula LLM service (OpenAI-compatible API).

Follows networkinstitutevu/nebula-api-examples/nebula-text-prompt.py: the
official `openai` client pointed at Nebula's base URL with your Nebula key
(Nebula → Settings → Account → API Keys → Show).

Configure via .env:
    CHAT_MODEL=nebula
    NEBULA_API_KEY=...
    NEBULA_MODEL=deepseek-r1:1.5b

List the models your key can use:
    python -m modules.chat_model.nebula
"""
import re

import openai
from openai import OpenAI

from modules.base import ChatModel, ChatModelError

NEBULA_BASE_URL = "https://nebula.cs.vu.nl/api/"

# Reasoning models (e.g. deepseek-r1) put their thinking in <think>…</think>;
# only the answer after it is shown to the curator.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class NebulaChatModel(ChatModel):
    def __init__(self, api_key: str, model: str = "deepseek-r1:1.5b",
                 base_url: str = NEBULA_BASE_URL, max_tokens: int = 1024):
        self._client = OpenAI(base_url=base_url, api_key=api_key) if api_key else None
        self._base_url = base_url
        self._model = model
        self._max_tokens = max_tokens

    def list_models(self) -> list[str]:
        if self._client is None:
            raise ChatModelError("No Nebula API key set. Add NEBULA_API_KEY to .env and restart the app.")
        return [m.id for m in self._client.models.list().data]

    def chat(self, messages: list[dict]) -> str:
        if self._client is None:
            raise ChatModelError("No Nebula API key set. Add NEBULA_API_KEY to .env and restart the app.")
        try:
            response = self._client.chat.completions.create(
                model=self._model, messages=messages, max_tokens=self._max_tokens,
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
            raise ChatModelError("Nebula rejected the API key. Check NEBULA_API_KEY in .env.") from e
        except (openai.NotFoundError, openai.BadRequestError) as e:
            raise ChatModelError(
                f"Nebula could not run model {self._model!r} ({e.message}). "
                "List the available models with: python -m modules.chat_model.nebula"
            ) from e
        except openai.APIConnectionError as e:
            raise ChatModelError(f"Could not reach Nebula at {self._base_url}.") from e
        except openai.APIError as e:
            raise ChatModelError(f"Nebula error: {e}") from e

        text = response.choices[0].message.content or ""
        return _THINK.sub("", text).strip()


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    model = NebulaChatModel(api_key=os.getenv("NEBULA_API_KEY", ""),
                            base_url=os.getenv("NEBULA_BASE_URL", NEBULA_BASE_URL))
    try:
        print("Nebula models:", ", ".join(model.list_models()))
    except Exception as e:
        print("Could not list Nebula models:", e)
