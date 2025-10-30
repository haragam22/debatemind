# backend/judge.py
from .utils import call_openrouter, sanitize_topic, parse_judge_json, load_pdf_context
from .config import MODEL_JUDGE

# Judge system prompt: must return JSON only
JUDGE_SYSTEM = "You are an objective debate judge. Evaluate two short arguments."

JUDGE_PROMPT_TEMPLATE = """
Topic: {topic}

COACHED ARGUMENT:
{coached}

OPPONENT ARGUMENT:
{opponent}

Reference Material (from uploaded PDF):
{pdf_context}

Score each argument on integers 1-10 for:
- logic
- relevance
- clarity
- persuasiveness

Return ONLY valid JSON with keys:
{{ "logic_coached", "relevance_coached", "clarity_coached", "persuasiveness_coached",
   "total_coached", "notes_coached",
   "logic_opponent", "relevance_opponent", "clarity_opponent", "persuasiveness_opponent",
   "total_opponent", "notes_opponent" }}
Be terse in notes.
"""

def evaluate(coached: str, opponent: str, topic: str) -> dict:
    topic = sanitize_topic(topic)
    pdf_context = load_pdf_context()
    content = JUDGE_PROMPT_TEMPLATE.format(topic=topic, coached=coached, opponent=opponent, pdf_context=pdf_context[:3000])
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": content}
    ]
    raw = call_openrouter(messages, MODEL_JUDGE)
    parsed = parse_judge_json(raw)
    return parsed
