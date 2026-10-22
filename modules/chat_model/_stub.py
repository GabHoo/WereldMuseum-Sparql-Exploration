from modules.base import ChatModel, ChatModelError


class MyChatModel(ChatModel):
    """
    Stub — copy this file, rename the class, implement chat(), then:
      1. Add an entry to REGISTRY["chat_model"] in config.py
      2. If it needs constructor args, add a branch in _build_chat_model()
      3. Set CHAT_MODEL=<your-key> in .env
    """

    def chat(self, messages: list[dict]) -> str:
        """
        messages: [{"role": "system" | "user" | "assistant", "content": str}, ...]
        Return the reply text. On failure raise ChatModelError("readable reason").
        """
        raise NotImplementedError
