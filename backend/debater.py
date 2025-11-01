# backend/debater.py
from .config import MODEL_COACHED
from typing import List, Dict
import json, os, time
from .utils import call_openrouter, sanitize_topic, clean_model_output


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
        prev_text = "\n\n".join(previous[-2:])
        messages.append({"role":"user", "content": f"Previous rounds (last two):\n{prev_text}"})
    return messages

def _robust_extract_text_from_llm(raw):
    """
    Attempts multiple heuristics to extract human text from various LLM response shapes.
    Returns the extracted string (may be empty).
    """
    if raw is None:
        return ""
    # if it's already a plain string with content, return it
    if isinstance(raw, str):
        s = raw.strip()
        # if the string looks like JSON, try parse below
        try:
            parsed = json.loads(s)
        except Exception:
            return s
        # if parsed, continue with parsed object
        data = parsed
    else:
        data = raw

    # Now data likely a dict/list - test common shapes
    try:
        # OpenAI-like
        if isinstance(data, dict):
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                ch = data["choices"][0]
                # message.content
                msg = ch.get("message") or ch.get("message", {})
                if isinstance(msg, dict) and "content" in msg and isinstance(msg["content"], str) and msg["content"].strip():
                    return msg["content"].strip()
                # text
                if "text" in ch and isinstance(ch["text"], str) and ch["text"].strip():
                    return ch["text"].strip()
            # OpenRouter/Anthropic-ish: output -> list -> content
            if "output" in data and isinstance(data["output"], list) and data["output"]:
                first = data["output"][0]
                if isinstance(first, dict):
                    if "content" in first and isinstance(first["content"], list):
                        for c in first["content"]:
                            if isinstance(c, dict) and "text" in c and isinstance(c["text"], str) and c["text"].strip():
                                return c["text"].strip()
                    if "text" in first and isinstance(first["text"], str) and first["text"].strip():
                        return first["text"].strip()
            # simple keys
            for key in ("output_text", "result", "response", "message", "text"):
                if key in data and isinstance(data[key], str) and data[key].strip():
                    return data[key].strip()
            # nested coached/opponent outputs sometimes have 'output'->'text'
            # search shallow for any string leaf
            def find_first_str_leaf(obj):
                if isinstance(obj, str) and obj.strip():
                    return obj.strip()
                if isinstance(obj, dict):
                    for v in obj.values():
                        found = find_first_str_leaf(v)
                        if found:
                            return found
                if isinstance(obj, list):
                    for v in obj:
                        found = find_first_str_leaf(v)
                        if found:
                            return found
                return None
            leaf = find_first_str_leaf(data)
            if leaf:
                return leaf
    except Exception:
        pass

    # Fallback: stringify the raw object
    try:
        text = json.dumps(data, ensure_ascii=False)
        return text if text.strip() else ""
    except Exception:
        return str(data) if str(data).strip() else ""

def generate_coached_argument(template_instruction: str, topic: str, previous: List[str]=None, retries=2, retry_delay=1.0) -> str:
    """
    Robust coached-argument generator.
    - Calls LLM and extracts text using multiple heuristics.
    - Saves raw LLM response to data/llm_last_response.json for debugging.
    - Retries a couple times if the result is empty.
    """
    messages = build_coached_prompt(template_instruction, topic, previous)

    # quick config checks
    if not MODEL_COACHED:
        raise RuntimeError("MODEL_COACHED is not set in backend.config")

    last_raw = None
    for attempt in range(1, retries + 1):
        try:
            raw = call_openrouter(messages, MODEL_COACHED)
        except Exception as e:
            # persist error to file for debugging and re-raise as descriptive runtime error
            os.makedirs("data", exist_ok=True)
            with open("data/llm_last_response.json", "w", encoding="utf-8") as fh:
                json.dump({"error": str(e), "attempt": attempt}, fh, indent=2, ensure_ascii=False)
            raise RuntimeError(f"LLM call failed on attempt {attempt}: {e}")

        last_raw = raw
        # save raw response for inspection
        try:
            os.makedirs("data", exist_ok=True)
            with open("data/llm_last_response.json", "w", encoding="utf-8") as fh:
                json.dump({"raw": raw}, fh, indent=2, ensure_ascii=False)
        except Exception:
            pass

        extracted = _robust_extract_text_from_llm(raw)
        if extracted and extracted.strip():
            return extracted.strip()

        # if empty, wait and retry (except after last attempt)
        if attempt < retries:
            time.sleep(retry_delay)

        extracted = _robust_extract_text_from_llm(raw)   # OR however you get 'raw' -> 'extracted'
        cleaned = clean_model_output(extracted)
        if not cleaned:
            raise RuntimeError("LLM returned empty/garbage for coached argument. Raw response saved.")
        return cleaned


    # if we reach here, all attempts returned empty. Save a helpful file and raise.
    try:
        os.makedirs("data", exist_ok=True)
        with open("data/llm_last_response.json", "w", encoding="utf-8") as fh:
            json.dump({"raw": last_raw, "note": "Empty text extracted after retries."}, fh, indent=2, ensure_ascii=False)
    except Exception:
        pass

    raise RuntimeError("LLM returned empty string for coached argument. Raw response saved to data/llm_last_response.json")
