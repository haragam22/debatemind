# main_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from backend.debater import generate_coached_argument
from backend.opponent import generate_opponent_argument
from backend.judge import evaluate
from backend.utils import load_pdf_context

app = FastAPI(title="DebateMind API", description="API for the DebateMind RL debate system", version="1.0.0")

# --- Input Schemas ---

class CoachedInput(BaseModel):
    topic: str
    instruction: str
    previous: Optional[List[str]] = None

class OpponentInput(BaseModel):
    topic: str
    last_argument: str

class JudgeInput(BaseModel):
    topic: str
    coached: str
    opponent: str


# --- Routes ---

@app.get("/")
def home():
    """Root endpoint"""
    return {
        "message": "Welcome to DebateMind API 👋",
        "available_endpoints": ["/generate-coached", "/generate-opponent", "/judge", "/pdf-context"]
    }


@app.get("/pdf-context")
def get_pdf_context():
    """
    Returns the first 1000 characters of the current PDF context
    to verify successful ingestion.
    """
    pdf_text = load_pdf_context()
    return {"preview": pdf_text[:1000], "length": len(pdf_text)}


@app.post("/generate-coached")
def generate_coached(data: CoachedInput):
    """
    Generates a coached debater argument using the current PDF context.
    """
    response = generate_coached_argument(
        template_instruction=data.instruction,
        topic=data.topic,
        previous=data.previous
    )
    return {"coached_argument": response}


@app.post("/generate-opponent")
def generate_opponent(data: OpponentInput):
    """
    Generates an opponent debater argument in response to the last argument.
    """
    response = generate_opponent_argument(
        last_argument=data.last_argument,
        topic=data.topic
    )
    return {"opponent_argument": response}


@app.post("/judge")
def judge(data: JudgeInput):
    """
    Evaluates two arguments (coached and opponent) and returns a JSON score.
    """
    response = evaluate(
        coached=data.coached,
        opponent=data.opponent,
        topic=data.topic
    )
    return response
