# backend/opponent.py
from .config import MODEL_OPPONENT
from typing import List, Dict # Added for type hinting
from .utils import call_openrouter, sanitize_topic, clean_model_output,load_pdf_context


SYSTEM_MESSAGE = "You are an opposing debater. Your job is to rebut the last argument concisely, using clear reasoning and evidence where possible."

def build_opponent_prompt(last_argument: str, topic: str) -> List[Dict]:
    topic = sanitize_topic(topic)
    pdf_context = load_pdf_context()
    
    # --- THIS IS THE FIX ---
    # Create a clear instruction for the LLM
    content = (
        f"Topic: {topic}\n\n"
        f"You must rebut the following argument from your opponent:\n"
        f"\"\"\"\n{last_argument}\n\"\"\"\n\n"
        f"Provide your counter-argument."
    )
    # -----------------------
    
    # Conditionally add PDF context
    if pdf_context.strip(): # Check if context is not just empty space
        content += f"\n\nReference Material (from uploaded PDF):\n{pdf_context[:3000]}"

    messages = [
        {"role":"system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": content,
        }
    ]
    return messages

def generate_opponent_argument(last_argument: str, topic: str) -> str:
    messages = build_opponent_prompt(last_argument, topic)
    raw = call_openrouter(messages, MODEL_OPPONENT)
    cleaned = clean_model_output(raw)
    return cleaned.strip() if cleaned else ""