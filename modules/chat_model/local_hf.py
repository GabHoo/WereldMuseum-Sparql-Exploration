"""
LocalHFChatModel — runs a Hugging Face chat model on this machine.

The model is loaded on the first chat, not at startup. Gated models (such as
meta-llama/*) need a Hugging Face login with access to the model; at startup
ensure_hf_login() looks for a token and, if there is none and a terminal is
attached, starts the Hugging Face login prompt. Without a login the app keeps
running and the chat explains what to do instead of crashing.

Configure via .env:
    CHAT_MODEL=local_hf
    CHAT_LOCAL_MODEL=meta-llama/Llama-3.2-1B-Instruct
    HF_TOKEN=hf_...            (optional; otherwise a saved login or the prompt is used)
"""
import os
import sys

from modules.base import ChatModel, ChatModelError


def ensure_hf_login() -> str | None:
    """Return a Hugging Face token: HF_TOKEN, a saved login, or a new interactive login."""
    from huggingface_hub import get_token, login

    token = os.getenv("HF_TOKEN") or get_token()
    if token:
        return token
    if sys.stdin.isatty():
        print("\nThe local chat model needs a Hugging Face login "
              "(create a token at https://huggingface.co/settings/tokens).")
        try:
            login()  # prompts for the token and saves it for next time
        except (KeyboardInterrupt, EOFError, Exception) as e:
            print(f"Hugging Face login skipped ({type(e).__name__}); the chat will ask again later.")
        return get_token()
    return None


def _default_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LocalHFChatModel(ChatModel):
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
                 device: str | None = None, max_new_tokens: int = 256):
        self._model_id = model_id
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._pipe = None
        self._token = ensure_hf_login()

    def _pipeline(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline

            self._token = self._token or ensure_hf_login()
            device = self._device or _default_device()
            try:
                self._pipe = pipeline(
                    "text-generation",
                    model=self._model_id,
                    dtype=torch.bfloat16 if device in ("mps", "cuda") else torch.float32,
                    device=device,
                    token=self._token,
                )
            except Exception as e:
                text = str(e).lower()
                if any(w in text for w in ("gated", "401", "403", "unauthorized", "authenticat")):
                    raise ChatModelError(
                        f"No access to {self._model_id}. Log in with a Hugging Face token "
                        f"(set HF_TOKEN in .env, or restart the app in a terminal to get the login "
                        f"prompt) and request access at https://huggingface.co/{self._model_id}."
                    ) from e
                raise ChatModelError(f"Could not load {self._model_id}: {e}") from e
        return self._pipe

    def chat(self, messages: list[dict]) -> str:
        outputs = self._pipeline()(messages, max_new_tokens=self._max_new_tokens)
        return str(outputs[0]["generated_text"][-1]["content"])
