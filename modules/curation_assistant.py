"""
CurationAssistant — answers the curator's chat messages about the current
selection (e.g. "summary" → the recurring themes and connections in it).

Model-agnostic: the actual model is a ChatModel (modules/chat_model/), chosen
with CHAT_MODEL in .env; config.py builds the assistant around it.

Every message is sent together with the selection as compact text (one line
per item, from the Artifact fields in CHAT_CONTEXT_FIELDS) and the exhibition
theme, if the curator gave one. Stateless: earlier messages are not resent.
"""
from modules.base import ChatModel
from modules.selection.utils import metadata_linearization

DEFAULT_CONTEXT_FIELDS = "title,type,material,place,date,maker,intended_use"

SYSTEM_PROMPT = (
    "You are an assistant to a museum curator who is putting together an exhibition. "
    "You are given the curator's current selection of objects, one per line, and "
    "possibly the theme of the exhibition.\n\n"
    "When the curator asks for a summary, describe the selection as a whole, not the "
    "objects one by one. Name the two to four strongest recurring themes and the "
    "connections between the objects: shared functions, materials, places, periods, "
    "makers or cultural practices. For each theme, write a short heading and one or two "
    "sentences on which kinds of objects share it and what links them. Mention a single "
    "object only as a brief example of a theme. If an exhibition theme is given, say how "
    "these connections relate to it and which objects do not fit it.\n\n"
    "Answer other questions about the selection directly. Be concise and answer in English."
)


class CurationAssistant:
    def __init__(self, model: ChatModel, context_fields: str = DEFAULT_CONTEXT_FIELDS):
        self._model = model
        self._linearize = metadata_linearization(context_fields)

    def reply(self, message: str, items: list, exhibition_theme: str = "") -> str:
        """
        message          : the curator's chat message
        items            : the selected pool items ({ id, title, properties })
        exhibition_theme : the curator's framing of the exhibition, may be empty
        Raises ChatModelError when the model cannot answer.
        """
        return self._model.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._user_content(message, items, exhibition_theme)},
        ])

    def _user_content(self, message: str, items: list, exhibition_theme: str) -> str:
        parts = []
        if exhibition_theme:
            parts.append(f"Exhibition theme: {exhibition_theme}")
        if items:
            lines = [f"{i}. {self._linearize(item) or item.get('title', 'Untitled')}"
                     for i, item in enumerate(items, 1)]
            parts.append(f"Selected objects ({len(items)}):\n" + "\n".join(lines))
        else:
            parts.append("No objects are selected yet.")
        parts.append(f"Curator: {message}")
        return "\n\n".join(parts)
