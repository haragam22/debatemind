# backend/judge.py
from .utils import call_openrouter, sanitize_topic, parse_judge_json, load_pdf_context
from .config import MODEL_JUDGE

# Judge system prompt: must return JSON only
JUDGE_SYSTEM = "You are a highly analytical and impartial debate judge. Your sole purpose is to evaluate two competing arguments based on a specific rubric and provide actionable feedback. You must follow all instructions and return ONLY the specified JSON format."

JUDGE_PROMPT_TEMPLATE = """
Topic: {topic}

---
COACHED ARGUMENT:
{coached}
---
OPPONENT ARGUMENT:
{opponent}
---
Reference Material (from uploaded PDF):
{pdf_context}
---

INSTRUCTIONS:
Evaluate both arguments based on the following SCORING RUBRIC.
All scores must be an integer from 1 (poor) to 10 (excellent).

SCORING RUBRIC:
1.  **Logic (1-10):** Is the argument sound, well-reasoned, and free of fallacies?
2.  **Relevance (1-10):** Does the argument directly address the topic?
3.  **Clarity (1-10):** Is the argument easy to understand, concise, and well-structured?
4.  **Persuasiveness (1-10):** Is the argument compelling? Does it use effective rhetoric?
5.  **Evidence Use (1-10):** How well did the argument use facts or evidence? (Score 5 if no evidence was needed or used. Score higher if it used the Reference Material effectively. Score lower if it ignored or contradicted the Reference Material.)

OUTPUT FORMAT:
You must return ONLY a single, valid JSON object. Do not include any other text, preambles, or explanations.
Use the nested structure below.
The "notes" MUST be 1-2 sentences of specific, constructive feedback explaining the *reason* for the scores.

JSON_KEYS_REQUIRED:
{{
    "coached": {{
        "logic": <int>,
        "relevance": <int>,
        "clarity": <int>,
        "persuasiveness": <int>,
        "evidence_use": <int>,
        "notes": "<string, 1-2 sentences of feedback>"
    }},
    "opponent": {{
        "logic": <int>,
        "relevance": <int>,
        "clarity": <int>,
        "persuasiveness": <int>,
        "evidence_use": <int>,
        "notes": "<string, 1-2 sentences of feedback>"
    }}
}}
"""

# backend/judge.py

# backend/judge.py

# backend/judge.py

def evaluate(coached: str, opponent: str, topic: str) -> dict:
    topic = sanitize_topic(topic)
    # Load context from a PDF
    pdf_context = load_pdf_context()
    
    # --- THIS IS THE FIX ---
    # Conditionally create the reference section
    reference_section = ""
    if pdf_context.strip():
        reference_section = f"Reference Material (from uploaded PDF):\n{pdf_context[:3000]}\n---"
    
    # Format the prompt, using the new 'reference_section'
    # The key in .format() MUST match the placeholder in your template
    content = JUDGE_PROMPT_TEMPLATE.format(
        topic=topic,
        coached=coached,
        opponent=opponent,
        pdf_context=reference_section # Pass the conditional section to the {pdf_context} placeholder
    )
    # -----------------------

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": content}
    ]
    raw = call_openrouter(messages, MODEL_JUDGE)
    parsed = parse_judge_json(raw)
    return parsed