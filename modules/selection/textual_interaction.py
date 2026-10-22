"""
Textual interaction: lets the curator chat with a language model about the
current selection (e.g. "summary" → commonalities and differences).

Model-agnostic: the actual model is a ChatModel (modules/chat_model/), chosen
with CHAT_MODEL in .env and passed in by app.py.
"""
from modules.base import ChatModel

SYSTEM_PROMPT = (
    "You are a helpful English-speaking assistant to a museum curator. "
    "When asked for a summary of the selected artifacts, do not describe the "
    "individual artifacts; summarise their commonalities and differences. Be concise."
)


def generate_chat_response(model: ChatModel, user_message: str) -> str:
    """
    user_message: the curator's question, optionally prefixed with the
    selection data (app.py adds the exported exhibition JSON).
    """
    if not user_message:
        return "Geen bericht ontvangen."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    return model.chat(messages)
