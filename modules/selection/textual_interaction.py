import json

from huggingface_hub import InferenceClient
import torch
from transformers import pipeline

# 1. Definieer het model (zorgt voor automatische download bij de eerste keer draaien)
model_id = "meta-llama/Llama-3.2-1B-Instruct"

# 2. Zet de pipeline op en dwing het gebruik van de Mac GPU (MPS) af
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,  # Geoptimaliseerd voor moderne hardware
    device="mps",                # Dit activeert de Apple Silicon GPU versnelling
    token=TOKEN_HUGGINGFACE,
)

chat_history = [
    {"role": "system", "content": "You are a helpful English-speaking assistant to a museum curator."},
]

def generate_chat_response(user_message: str) -> str:
    with open("exhibition.json", "r") as f:
        selection = json.load(f)

    # 3. Formatteer je chat-prompt
    chat_history = [
        {"role": "system", "content": "You are a helpful English-speaking assistant to a museum curator."},
        {
            "role": "user",
            "content": f"Given the following JSON file: {selection}, give a summary of the selected artifacts. Do not describe the individual artifacts, just give a summary of the commonalities and differences. Be concise."
        },
    ]

    """Verwerkt het bericht, voegt het toe aan de historie en genereert een antwoord."""
    if not user_message:
        return "Geen bericht ontvangen."

    # Voeg gebruikersvraag toe aan de geschiedenis
    chat_history.append({"role": "user", "content": user_message})

    # Genereer antwoord via de pipeline
    outputs = pipe(
        chat_history,
        max_new_tokens=256,  # Kortere limiet houdt het snel op de Mac GPU
    )

    response_text = outputs[0]["generated_text"][-1]["content"]

    # Voeg assistent-antwoord toe aan geschiedenis voor context behoud
    chat_history.append({"role": "assistant", "content": response_text})
    return str(response_text)