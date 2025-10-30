# backend/utils.py
import json
import re
from typing import Dict, Any
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL, HTTP_TIMEOUT
import httpx
import os

def sanitize_topic(topic: str) -> str:
    # basic sanitization to avoid huge payloads or control char injection
    s = topic.strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > 400:
        s = s[:400]
    return s

def call_openrouter(messages, model):
    """
    messages: list of {"role": "user"/"system", "content": "..."}
    model: model name via OpenRouter
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in environment")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            resp = client.post(OPENROUTER_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # compatibility: OpenRouter uses choices[0].message.content
            return data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        # bubble up error without leaking secrets
        raise RuntimeError(f"LLM API error: {e.response.status_code} {e.response.text[:200]}")
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {str(e)}")

def parse_judge_json(raw: str) -> Dict[str, Any]:
    """
    Robustly extract JSON object from the judge's text output.
    Expect keys like logic_coached, clarity_coached, etc.
    """
    try:
        # find first { ... } block
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found")
        blob = raw[start:end+1]
        data = json.loads(blob)
        # minimal validation: ensure numeric totals exist
        for k in ["total_coached", "total_opponent"]:
            if k not in data:
                # attempt compute from component scores
                coach_vals = [data.get(x, 0) for x in ("logic_coached","relevance_coached","clarity_coached","persuasiveness_coached")]
                opp_vals = [data.get(x, 0) for x in ("logic_opponent","relevance_opponent","clarity_opponent","persuasiveness_opponent")]
                if k=="total_coached" and any(coach_vals):
                    data["total_coached"] = sum(coach_vals)/len(coach_vals)
                if k=="total_opponent" and any(opp_vals):
                    data["total_opponent"] = sum(opp_vals)/len(opp_vals)
        return data
    except Exception as e:
        # fallback: neutral scores
        return {
            "logic_coached":5, "relevance_coached":5, "clarity_coached":5, "persuasiveness_coached":5,
            "total_coached":5.0, "notes_coached":"fallback",
            "logic_opponent":5, "relevance_opponent":5, "clarity_opponent":5, "persuasiveness_opponent":5,
            "total_opponent":5.0, "notes_opponent":"fallback"
        }

def load_pdf_context(file_path="data/extracted_text.txt"):
    """Load extracted PDF text as context for debates."""
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()