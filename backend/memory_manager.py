
# backend/memory_manager.py
import os
import threading
import pandas as pd
import json
from typing import Dict, Any

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(BASE_DIR, exist_ok=True)

DEBATE_CSV = os.path.join(BASE_DIR, "debate_memory.csv")
JUDGE_CSV = os.path.join(BASE_DIR, "judge_summary.csv")
RL_JSON = os.path.join(BASE_DIR, "rl_memory.json")

_lock = threading.Lock()

def init_files():
    # ensure csv files have headers
    with _lock:
        if not os.path.exists(DEBATE_CSV):
            df = pd.DataFrame(columns=["round","speaker","coached_argument","opponent_argument","action","reward"])
            df.to_csv(DEBATE_CSV, index=False)
        if not os.path.exists(JUDGE_CSV):
            df = pd.DataFrame(columns=["round","total_coached","total_opponent","notes_coached","notes_opponent"])
            df.to_csv(JUDGE_CSV, index=False)
        if not os.path.exists(RL_JSON):
            with open(RL_JSON, "w", encoding="utf-8") as f:
                json.dump({"template_stats": {}, "epsilon": 0.25}, f)

def append_round(round_entry: Dict[str, Any]):
    with _lock:
        df = pd.read_csv(DEBATE_CSV)
        df = pd.concat([df, pd.DataFrame([round_entry])], ignore_index=True)
        df.to_csv(DEBATE_CSV, index=False)

def append_judge(round_num: int, judge_dict: Dict[str, Any]):
    with _lock:
        df = pd.read_csv(JUDGE_CSV)
        entry = {
            "round": round_num,
            "total_coached": judge_dict.get("total_coached", 0.0),
            "total_opponent": judge_dict.get("total_opponent", 0.0),
            "notes_coached": judge_dict.get("notes_coached", ""),
            "notes_opponent": judge_dict.get("notes_opponent", "")
        }
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        df.to_csv(JUDGE_CSV, index=False)

def read_debate():
    with _lock:
        return pd.read_csv(DEBATE_CSV)

def read_judge():
    with _lock:
        return pd.read_csv(JUDGE_CSV)

def read_rl_memory():
    with _lock:
        with open(RL_JSON, "r", encoding="utf-8") as f:
            return json.load(f)

def write_rl_memory(data: Dict[str, Any]):
    with _lock:
        with open(RL_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
