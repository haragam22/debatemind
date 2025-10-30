# backend/opponent.py
from .utils import call_openrouter, sanitize_topic, load_pdf_context
from .config import MODEL_OPPONENT

SYSTEM_MESSAGE = "You are an opposing debater. Your job is to rebut the last argument concisely, using clear reasoning and evidence where possible."

def build_opponent_prompt(last_argument: str, topic: str) -> list:
    topic = sanitize_topic(topic)
    pdf_context = load_pdf_context()
    messages = [
        {"role":"system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": (
                f"Instruction: {last_argument}\n"
                f"Topic: {topic}\n\n"
                f"Reference Material (from uploaded PDF):\n{pdf_context[:3000]}"
            ),
        }
    ]
    return messages

def generate_opponent_argument(last_argument: str, topic: str) -> str:
    messages = build_opponent_prompt(last_argument, topic)
    raw = call_openrouter(messages, MODEL_OPPONENT)
    return raw.strip()
