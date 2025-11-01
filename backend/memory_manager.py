
import pandas as pd
import os
import time
import shutil
from .utils import sanitize_topic
import json

# --- 1. NEW: Define the main data directory ---
DATA_DIR = "data"
DEBATE_FILENAME = "debate.csv"
JUDGE_FILENAME = "judge.csv"
RL_MEMORY_FILE = "rl_memory.json" # <-- ADD THIS

# --- 2. MODIFIED: This function now creates the main 'data' folder ---
def init_storage():
    """
    Ensures the main 'data' directory exists.
    This is called by the "Reset Storage" button to wipe ALL debates.
    """
    # If the folder exists, wipe it completely
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    
    # Create a fresh, empty 'data' folder
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Storage initialized at {DATA_DIR}")

# --- 3. NEW: Function to create a NEW debate session ---
# backend/memory_manager.py

# backend/memory_manager.py

# backend/memory_manager.py

def create_new_debate(topic: str) -> str:
    """
    Creates a new, unique folder for a specific debate.
    Returns the ID of the new debate (which is its folder name).
    """
    # Create a unique ID from the topic and current time
    safe_topic = sanitize_topic(topic)
    timestamp = int(time.time())
    debate_id = f"{safe_topic}_{timestamp}" # e.g., "remote-work_1678886400"
    
    debate_path = os.path.join(DATA_DIR, debate_id)
    os.makedirs(debate_path, exist_ok=True)
    
    # Define file paths
    debate_file = os.path.join(debate_path, DEBATE_FILENAME)
    judge_file = os.path.join(debate_path, JUDGE_FILENAME)
    
    # Create the empty CSVs *inside* the new folder
    if not os.path.exists(debate_file):
        pd.DataFrame(columns=[
            "round", "speaker", "coached_argument", 
            "opponent_argument", "action", "reward"
        ]).to_csv(debate_file, index=False)

    # --- THIS IS THE FIX ---
    # The judge file MUST have all the columns that parse_judge_json creates.
    if not os.path.exists(judge_file):
        pd.DataFrame(columns=[
            "round", 
            "logic_coached", "relevance_coached", "clarity_coached", 
            "persuasiveness_coached", "evidence_use_coached",
            "total_coached", "notes_coached",
            "logic_opponent", "relevance_opponent", "clarity_opponent",
            "persuasiveness_opponent", "evidence_use_opponent",
            "total_opponent", "notes_opponent"
        ]).to_csv(judge_file, index=False)
        
    return debate_id # Return the new ID
# --- 4. NEW: Function to list all saved debates ---
def list_debates() -> list:
    """
    Scans the DATA_DIR and returns a list of all debate IDs (folder names).
    """
    if not os.path.exists(DATA_DIR):
        return []
    
    # List all entries in the data directory
    entries = os.listdir(DATA_DIR)
    
    # Filter for directories only
    debate_ids = [entry for entry in entries if os.path.isdir(os.path.join(DATA_DIR, entry))]
    
    # Sort by name (which will be timestamp)
    debate_ids.sort(reverse=True)
    return debate_ids

# --- 5. MODIFIED: All file functions now require a 'debate_id' ---

def get_debate_path(debate_id: str) -> str:
    """Helper to get the full path to a debate's CSV."""
    return os.path.join(DATA_DIR, debate_id, DEBATE_FILENAME)

def get_judge_path(debate_id: str) -> str:
    """Helper to get the full path to a judge's CSV."""
    return os.path.join(DATA_DIR, debate_id, JUDGE_FILENAME)

def read_debate(debate_id: str) -> pd.DataFrame:
    """Reads the debate.csv for a specific debate."""
    if not debate_id:
        return pd.DataFrame()
    try:
        return pd.read_csv(get_debate_path(debate_id))
    except FileNotFoundError:
        return pd.DataFrame()

def read_judge(debate_id: str) -> pd.DataFrame:
    """Reads the judge.csv for a specific debate."""
    if not debate_id:
        return pd.DataFrame()
    try:
        return pd.read_csv(get_judge_path(debate_id))
    except FileNotFoundError:
        return pd.DataFrame()

def append_round(debate_id: str, round_data: dict):
    """Appends a round to the correct debate.csv, aligning columns to the file header."""
    if not debate_id:
        return

    debate_path = get_debate_path(debate_id)
    new_row = pd.DataFrame([round_data])

    # If file doesn't exist create with expected columns (safe default)
    if not os.path.exists(debate_path):
        # create file with appropriate columns in the expected order
        cols = ["round", "speaker", "coached_argument", "opponent_argument", "action", "reward"]
        pd.DataFrame(columns=cols).to_csv(debate_path, index=False)

    # Read header columns (no data) and reindex new_row to that column order
    try:
        existing_header = pd.read_csv(debate_path, nrows=0).columns.tolist()
        # Ensure new_row has all header columns (add missing with NaN)
        new_row = new_row.reindex(columns=existing_header)
    except Exception:
        # Fallback if reading header fails — just append with current row columns
        pass

    # Append row (header=False because file already has header)
    new_row.to_csv(debate_path, mode='a', header=False, index=False)


def append_judge(debate_id: str, judge_data: dict):
    """
    Appends judge scores to the correct judge.csv, aligning columns to the file header
    so fields like notes_coached/notes_opponent don't get misaligned.
    """
    if not debate_id:
        return

    judge_path = get_judge_path(debate_id)
    new_row = pd.DataFrame([judge_data])

    # If file doesn't exist, create it with the canonical judge columns
    if not os.path.exists(judge_path):
        cols = [
            "round",
            "logic_coached", "relevance_coached", "clarity_coached",
            "persuasiveness_coached", "evidence_use_coached",
            "total_coached", "notes_coached",
            "logic_opponent", "relevance_opponent", "clarity_opponent",
            "persuasiveness_opponent", "evidence_use_opponent",
            "total_opponent", "notes_opponent"
        ]
        pd.DataFrame(columns=cols).to_csv(judge_path, index=False)

    # Read header columns and reindex the new row to match them exactly
    try:
        header_cols = pd.read_csv(judge_path, nrows=0).columns.tolist()
        # Ensure notes keys exist in the row (so reindex doesn't drop them)
        # Provide default empty strings for notes if missing
        if "notes_coached" not in new_row.columns:
            new_row["notes_coached"] = new_row.apply(lambda _: judge_data.get("notes_coached", judge_data.get("notes", "")), axis=1)
        if "notes_opponent" not in new_row.columns:
            new_row["notes_opponent"] = new_row.apply(lambda _: judge_data.get("notes_opponent", judge_data.get("notes_opponent", judge_data.get("opponent_notes", ""))), axis=1)

        # Reindex to header columns (missing columns become NaN; that's fine)
        new_row = new_row.reindex(columns=header_cols)
    except Exception:
        # If header read fails for some reason, proceed with current new_row as-is
        pass

    # Optionally coerce numeric totals to float before writing (safer for reads later)
    for num_col in ["total_coached", "total_opponent"]:
        if num_col in new_row.columns:
            try:
                new_row[num_col] = pd.to_numeric(new_row[num_col], errors='coerce').fillna(0.0)
            except Exception:
                pass

    # Append to file (header=False — file already has header)
    new_row.to_csv(judge_path, mode='a', header=False, index=False)

# --- 6. NEW: RL Agent Memory Functions ---

# --- 6. NEW: RL Agent Memory Functions (JSON Version) ---

def get_rl_memory_path():
    """Helper to get the full path to the RL memory file."""
    return os.path.join(DATA_DIR, RL_MEMORY_FILE)

def read_rl_memory() -> dict:
    """
    Reads the RL agent's JSON memory file from the data directory.
    If not found, returns an empty dictionary.
    """
    path = get_rl_memory_path()
    if not os.path.exists(path):
        return {}  # Return an empty dict if the file doesn't exist
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            return data
    except (json.JSONDecodeError, FileNotFoundError):
        return {} # Return empty dict on error

def write_rl_memory(mem: dict):
    """Saves the RL agent's memory dictionary to a JSON file."""
    path = get_rl_memory_path()
    try:
        with open(path, 'w') as f:
            json.dump(mem, f, indent=4)
    except Exception as e:
        print(f"Error saving RL memory: {e}")
