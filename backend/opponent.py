# backend/opponent.py
from .utils import call_openrouter, sanitize_topic
from .config import MODEL_OPPONENT

SYSTEM_MESSAGE = "You are an opposing debater. Your job is to rebut the last argument concisely."

def build_opponent_prompt(last_argument: str, topic: str) -> list:
    topic = sanitize_topic(topic)
    messages = [
        {"role":"system", "content": SYSTEM_MESSAGE},
        {"role":"user", "content": f"Topic: {topic}\nArgument to rebut:\n{last_argument}\n\nProvide a 3-5 sentence rebuttal addressing weaknesses and counterpoints."}
    ]
    return messages

def generate_opponent_argument(last_argument: str, topic: str) -> str:
    messages = build_opponent_prompt(last_argument, topic)
    raw = call_openrouter(messages, MODEL_OPPONENT)
    return raw.strip()
