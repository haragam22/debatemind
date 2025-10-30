# backend/debater.py
from .utils import call_openrouter, sanitize_topic, load_pdf_context
from .config import MODEL_COACHED
from typing import List, Dict

SYSTEM_MESSAGE = "You are an expert debater. Produce a concise, structured argument. Keep it 3-6 sentences."

def build_coached_prompt(template_instruction: str, topic: str, previous: List[str]=None) -> List[Dict]:
    topic = sanitize_topic(topic)
    pdf_context = load_pdf_context()
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": (
                f"Instruction: {template_instruction}\n"
                f"Topic: {topic}\n\n"
                f"Reference Material (from uploaded PDF):\n{pdf_context[:3000]}"
            ),
        }
    ]
    
    if previous:
        # attach last n previous arguments to maintain context (short)
        prev_text = "\n\n".join(previous[-2:])
        messages.append({"role":"user", "content": f"Previous rounds (last two):\n{prev_text}"})
    return messages

def generate_coached_argument(template_instruction: str, topic: str, previous: List[str]=None) -> str:
    messages = build_coached_prompt(template_instruction, topic, previous)
    raw = call_openrouter(messages, MODEL_COACHED)
    return raw.strip()