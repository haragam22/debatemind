# backend/utils.py
import json
import re
from typing import Dict, Any
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL, HTTP_TIMEOUT
import httpx
import os

import re

def clean_model_output(text: str) -> str:
    """
    Remove common special tokens and noisy markers returned by LLMs,
    normalize whitespace, and return a clean string for UI display.
    """
    if text is None:
        return text
    s = str(text)

    # 1) Remove common start/end/pad tokens and variants
    s = re.sub(r'(<s>|<\/s>|<pad>|<\|endoftext\|>|\[CLS\]|\[SEP\])', '', s, flags=re.IGNORECASE)

    # 2) Remove other common artifacts sometimes returned by tokenizers
    #    e.g., SentencePiece underscores (▁) — optionally collapse them to spaces
    s = s.replace('▁', ' ')

    # 3) Remove / normalize repeated HTML-like angle brackets if accidentally present
    s = re.sub(r'\s*<\s*>', '', s)

    # 4) Strip leading/trailing punctuation garbage like repeated dashes or pipes
    s = re.sub(r'^[\s\-\|:;,_]+', '', s)
    s = re.sub(r'[\s\-\|:;,_]+$', '', s)

    # 5) Normalize newlines and repeated whitespace
    s = re.sub(r'\r\n?', '\n', s)
    s = re.sub(r'\n{3,}', '\n\n', s)           # keep at most one blank line
    s = re.sub(r'[ \t]{2,}', ' ', s)
    s = s.strip()

    # 6) If it still contains only non-word tokens, return empty string to trigger fallback
    if re.fullmatch(r'[\W_]+', s):
        return ""

    return s

def sanitize_topic(topic: str) -> str:
    """
    Removes invalid filesystem characters from a topic string
    and converts it to a clean, URL-friendly format.
    """
    if not topic:
        return "untitled"

    # 1. Convert to lowercase
    text = topic.lower()

    # 2. Remove invalid characters ( \ / : * ? " < > | )
    text = re.sub(r'[\\/:*?"<>|]', '', text)

    # 3. Replace spaces and other separators with a hyphen
    text = re.sub(r'[\s_.]+', '-', text)

    # 4. Remove any leading/trailing hyphens
    text = text.strip('-')

    # 5. Limit length to avoid "filename too long" errors
    text = text[:50] # Keep it reasonably short

    # 6. Final check in case it's now empty
    if not text:
        return "sanitized-topic"

    return text

# backend/utils.py  -> replace call_openrouter(...) with this version

def call_openrouter(messages, model):
    """
    Robust LLM caller for OpenRouter-compatible endpoints.
    Tries multiple response patterns, saves last raw JSON to data/llm_last_response.json for debugging.
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

            # Save raw response for debugging (safe: non-sensitive but don't commit to git)
            try:
                os.makedirs("data", exist_ok=True)
                with open("data/llm_last_response.json", "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, ensure_ascii=False)
            except Exception:
                pass

            # 1) OpenAI-like response (choices[0].message.content)
            try:
                return data["choices"][0]["message"]["content"]
            except Exception:
                pass

            # 2) OpenRouter / Anthropic-ish output: "output"[0]["content"][0]["text"] or "output_text"
            try:
                out = data.get("output")
                if isinstance(out, list) and out:
                    # path: output[0].get('content') -> list of dicts with 'text' or 'type' 'output_text'
                    first = out[0]
                    if isinstance(first, dict):
                        # search for nested text fields
                        if "content" in first and isinstance(first["content"], list) and first["content"]:
                            for c in first["content"]:
                                if isinstance(c, dict) and "text" in c:
                                    return c["text"]
                                if isinstance(c, dict) and "type" in c and c["type"] == "output_text" and "text" in c:
                                    return c["text"]
                        # direct text key
                        if "text" in first:
                            return first["text"]
                # fallback nested top-level key
                if "output_text" in data:
                    return data["output_text"]
            except Exception:
                pass

            # 3) Some routers return a top-level "result" or "message"
            for key in ("result", "message", "response", "text"):
                if key in data and isinstance(data[key], str):
                    return data[key]

            # If we reach here, nothing parsed — return stringified JSON as last resort
            return json.dumps(data)
    except httpx.HTTPStatusError as e:
        # write response body to log for debugging
        text = ""
        try:
            text = e.response.text[:400]
        except Exception:
            text = str(e)
        raise RuntimeError(f"LLM API HTTP error: {e.response.status_code} — {text}")
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {str(e)}")


def parse_judge_json(raw: str) -> Dict[str, Any]:
    """
    Robustly extract JSON object from the judge's text output and return a flat dict.
    Accepts:
      - fully nested JSON: {"coached": {...}, "opponent": {...}}
      - flat JSON: {"logic_coached": 7, "notes_coached": "...", ...}
    Normalizes numeric scores to ints, ensures notes fields exist and are strings.
    """
    def to_int_safe(v, default=5):
        try:
            # handle "7", "7.0", 7.0, 7
            if isinstance(v, (int, float)):
                return int(round(v))
            if isinstance(v, str):
                v2 = v.strip()
                if not v2:
                    return default
                return int(round(float(v2)))
        except Exception:
            return default
        return default

    def pick_note(blob, *keys):
        # Look in nested blob for possible note keys; return first non-empty string
        for k in keys:
            if not blob:
                continue
            val = blob.get(k) if isinstance(blob, dict) else None
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    # default safe flat structure
    fallback = {
        "logic_coached":5, "relevance_coached":5, "clarity_coached":5, "persuasiveness_coached":5, "evidence_use_coached":5,
        "total_coached":5.0, "notes_coached":"Fallback: Error parsing judge response.",
        "logic_opponent":5, "relevance_opponent":5, "clarity_opponent":5, "persuasiveness_opponent":5, "evidence_use_opponent":5,
        "total_opponent":5.0, "notes_opponent":"Fallback: Error parsing judge response."
    }

    try:
        # 1) Try direct JSON parse
        try:
            nested = json.loads(raw)
        except Exception:
            # 2) fallback: extract first {...} block from text
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object found in judge output")
            blob = raw[start:end+1]
            nested = json.loads(blob)

        flat = {}

        # If nested contains 'coached' and 'opponent' nested dicts
        if isinstance(nested, dict) and "coached" in nested and "opponent" in nested:
            coached_blob = nested.get("coached", {}) or {}
            opponent_blob = nested.get("opponent", {}) or {}

            coach_keys = ["logic", "relevance", "clarity", "persuasiveness", "evidence_use"]
            opp_keys = coach_keys

            coach_vals = []
            for k in coach_keys:
                score = coached_blob.get(k, coached_blob.get(f"{k}_score", 5))
                s = to_int_safe(score, default=5)
                flat[f"{k}_coached"] = s
                coach_vals.append(s)

            opp_vals = []
            for k in opp_keys:
                score = opponent_blob.get(k, opponent_blob.get(f"{k}_score", 5))
                s = to_int_safe(score, default=5)
                flat[f"{k}_opponent"] = s
                opp_vals.append(s)

            # notes: look in many possible key names
            notes_c = pick_note(coached_blob, "notes", "note", "feedback", "comment")
            notes_o = pick_note(opponent_blob, "notes", "note", "feedback", "comment")

            flat["notes_coached"] = notes_c or "No notes provided."
            flat["notes_opponent"] = notes_o or "No notes provided."

            flat["total_coached"] = sum(coach_vals)/len(coach_vals) if coach_vals else 5.0
            flat["total_opponent"] = sum(opp_vals)/len(opp_vals) if opp_vals else 5.0

            return flat

        # If nested is a flat mapping with keys like 'logic_coached', 'notes_coached'
        if isinstance(nested, dict):
            # attempt to detect flat style
            keys_lower = {k.lower(): k for k in nested.keys()}
            # scoring keys we expect
            core_keys = ["logic_coached","relevance_coached","clarity_coached","persuasiveness_coached","evidence_use_coached",
                         "logic_opponent","relevance_opponent","clarity_opponent","persuasiveness_opponent","evidence_use_opponent"]
            found_core = any(k in keys_lower for k in core_keys)
            if found_core:
                # pull each expected key with sensible defaults
                coach_vals = []
                for k in ["logic_coached","relevance_coached","clarity_coached","persuasiveness_coached","evidence_use_coached"]:
                    rawv = nested.get(k, nested.get(k.lower(), 5))
                    v = to_int_safe(rawv, default=5)
                    flat[k] = v
                    coach_vals.append(v)

                opp_vals = []
                for k in ["logic_opponent","relevance_opponent","clarity_opponent","persuasiveness_opponent","evidence_use_opponent"]:
                    rawv = nested.get(k, nested.get(k.lower(), 5))
                    v = to_int_safe(rawv, default=5)
                    flat[k] = v
                    opp_vals.append(v)

                # notes may appear as notes_coached / notes_opponent or other variants
                notes_c = nested.get("notes_coached") or nested.get("coach_notes") or nested.get("coached_notes") or nested.get("notes") or ""
                notes_o = nested.get("notes_opponent") or nested.get("opponent_notes") or ""
                notes_c = notes_c if isinstance(notes_c, str) and notes_c.strip() else ""
                notes_o = notes_o if isinstance(notes_o, str) and notes_o.strip() else ""

                flat["notes_coached"] = notes_c or "No notes provided."
                flat["notes_opponent"] = notes_o or "No notes provided."

                flat["total_coached"] = sum(coach_vals)/len(coach_vals) if coach_vals else 5.0
                flat["total_opponent"] = sum(opp_vals)/len(opp_vals) if opp_vals else 5.0

                return flat

        # If nothing matched, return fallback (ensures notes exist)
        print("parse_judge_json: Unexpected JSON structure, returning fallback.")
        return fallback

    except Exception as e:
        # keep fallback and avoid crashing main app
        print(f"Error parsing judge JSON, using fallback. Error: {e}")
        return fallback

    except Exception as e:
        # fallback: neutral scores (this part is unchanged)
        print(f"Error parsing judge JSON, using fallback. Error: {e}")
        return {
            "logic_coached":5, "relevance_coached":5, "clarity_coached":5, "persuasiveness_coached":5, "evidence_use_coached": 5,
            "total_coached":5.0, "notes_coached":"Fallback: Error parsing judge response.",
            "logic_opponent":5, "relevance_opponent":5, "clarity_opponent":5, "persuasiveness_opponent":5, "evidence_use_opponent": 5,
            "total_opponent":5.0, "notes_opponent":"Fallback: Error parsing judge response."
        }

def load_pdf_context(file_path="data/extracted_text.txt"):
    """Load extracted PDF text as context for debates."""
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    
# backend/utils.py

# ... (keep all your existing functions like load_pdf_context)

def clear_pdf_context(file_path="data/extracted_text.txt"):
    """Deletes the extracted PDF text file, if it exists."""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Cleared old PDF context: {file_path}")
        except Exception as e:
            print(f"Error deleting PDF context: {e}")