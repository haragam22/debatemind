# backend/config.py
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env if present

# Use OpenRouter (set OPENROUTER_API_KEY in your env or .env)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")

if not OPENROUTER_API_KEY:
    raise EnvironmentError("⚠️ Missing OPENROUTER_API_KEY. Please set it in your .env file.")

# Models: you can change to any model available via OpenRouter
MODEL_COACHED = os.getenv("MODEL_COACHED", "meta-llama/llama-3.1-8b-instruct")
MODEL_OPPONENT = os.getenv("MODEL_OPPONENT", "meta-llama/llama-3.1-8b-instruct")
MODEL_JUDGE = os.getenv("MODEL_JUDGE", "mistralai/mistral-7b-instruct") 

# Timeouts (seconds)
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))

# RL policy / template choices
TEMPLATES = [
    "Be concise and focus on logical structure and evidence.",
    "Emphasize emotional appeal and human impact; be persuasive.",
    "Focus on counter-arguments and rebuttals; attack assumptions.",
    "Use analogies and examples to illustrate the point clearly.",
    "Provide legal/ethical reasoning and reference principles.",
    "Prioritize clarity and brevity with a strong summary."
]

# Round limit
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "5"))