import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import json
import os
from PyPDF2 import PdfReader

# --- Preserve Backend Imports ---
from backend.memory_manager import (
    init_storage, create_new_debate, list_debates, # NEW
    read_debate, read_judge, append_round, append_judge # MODIFIED
)
from backend.rl_agent import RLAgent
from backend.debater import generate_coached_argument
from backend.opponent import generate_opponent_argument
from backend.judge import evaluate
from backend.utils import sanitize_topic, load_pdf_context, clear_pdf_context
from backend.config import MAX_ROUNDS 

def format_score_as_points(score_val):
    """Formats a score (expected 0-10) to points, assuming it's already clamped."""
    try:
        score = float(score_val)
        return f"{score:.1f}" # Display one decimal place for clarity
    except (ValueError, TypeError):
        return "N/A"

# --- 1. Initialization and Setup ---
rl = RLAgent()

st.set_page_config(
    page_title="DebateMind: AI Coach",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS / Styling (includes chatbox + bubbles) ---
# [UPDATED] Added CSS for the Interruption Judge Box
STYLING_CSS = """
    <style>
        /* General Dark Theme */
        body, .stApp {
            background-color: #0d1117;
            color: #c9d1d9;
        }
        h1, h2, h3, h4, h5, h6, p, label {
            color: #f0f4f8 !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #30363d;
            padding-top: 1rem;
        }
        .sidebar-divider { margin: 8px 0; border-top: 1px solid #30363d; }

        /* Chat container centered column */
        .chat-wrapper {
            display: flex;
            justify-content: center;
            margin-top: 8px;
            margin-bottom: 8px;
        }
        .chat-box {
            width: 80%;
            max-width: 880px;
            min-width: 340px;
            background: linear-gradient(180deg, rgba(18,20,24,0.6), rgba(14,16,20,0.4));
            border: 1px solid #22272b;
            border-radius: 12px;
            padding: 18px;
            box-shadow: 0 8px 30px rgba(2,6,23,0.6);
            overflow: hidden;
        }
        
        /* --- NEW CHAT STYLES --- */
        
        /* Each message row (now uses flex) */
        .message-row {
            display: flex;
            margin: 16px 0; /* More spacing */
            width: 100%;
            clear: both;
        }

        /* Avatar styling */
        .avatar {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 1.1rem;
            flex-shrink: 0;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }
        .avatar-coach {
            background-color: #3b82f6; /* Blue */
            color: white;
            border: 1px solid #5aa7ff;
        }
        .avatar-opponent {
            background-color: #ef4444; /* Red */
            color: white;
            border: 1px solid #ff7b7b;
        }

        /* Message content (bubble + meta) */
        .message-content {
            display: flex;
            flex-direction: column;
            max-width: 82%; /* Max width for the content block */
        }
        
        /* Align message content left/right */
        .message-row-left {
            justify-content: flex-start;
        }
        .message-row-left .message-content {
            margin-left: 12px;
            align-items: flex-start;
        }

        .message-row-right {
            justify-content: flex-end;
        }
        .message-row-right .message-content {
            margin-right: 12px;
            align-items: flex-end; /* This makes the bubble align right */
        }

        /* Metadata (now outside bubble) */
        .msg-meta {
            font-size: 0.82rem;
            color: #9aa4b2;
            margin-bottom: 5px;
            padding: 0 4px;
        }

        /* Bubbles (Redesigned) */
        .bubble-left {
            background: rgba(59,130,246,0.1);
            border: 1px solid rgba(59,130,246,0.3);
            color: #e6f0ff;
            padding: 12px 16px;
            border-radius: 16px;
            border-top-left-radius: 6px; /* 'tail' */
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            font-size: 0.97rem;
            line-height: 1.45;
            word-wrap: break-word;
        }
        
        .bubble-right {
            background: rgba(239,68,68,0.08);
            border: 1px solid rgba(239,68,68,0.25);
            color: #ffecec;
            padding: 12px 16px;
            border-radius: 16px;
            border-top-right-radius: 6px; /* 'tail' */
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            font-size: 0.97rem;
            line-height: 1.45;
            word-wrap: break-word;
        }
        
        /* --- END NEW CHAT STYLES --- */

        /* Judge box centered and matching chat width */
        .judge-wrapper {
            display: flex;
            justify-content: center;
            margin-top: 10px;
            margin-bottom: 24px;
        }
        .judge-box {
            width: 80%;
            max-width: 880px;
            min-width: 340px;
            background: rgba(251,191,36,0.06);
            border: 1px solid rgba(245,158,11,0.18);
            padding: 12px 16px;
            border-radius: 10px;
            color: #fff7e6;
            box-shadow: 0 6px 18px rgba(0,0,0,0.35);
            font-size: 0.95rem;
            line-height: 1.45;
        }
        
        /* --- Interruption Judge Box Style --- */
        .judge-interruption {
            background: rgba(56, 189, 248, 0.1); /* Light Blue/Teal */
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: #e6f0ff;
        }
        /* --- END Interruption Judge Box Style --- */

        /* --- NEW WINNER POPUP STYLES (BLUE/BLACK THEME) --- */
        .winner-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: rgba(0, 0, 0, 0.9); /* Opaque black overlay */
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            opacity: 0; 
            animation: fadeIn 0.5s forwards;
        }

        .winner-content {
            /* Bluish-Black Theme - Set to a reasonable, considerable size */
            background: #0d1117; 
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            /* Blue shiny border and glow */
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.8), inset 0 0 10px rgba(56, 189, 248, 0.4); 
            transform: scale(0.8);
            animation: zoomIn 0.6s cubic-bezier(0.68, -0.55, 0.27, 1.55) forwards;
            border: 3px solid #38bdf8;
            width: 80%;
            max-width: 500px; /* Considerable size */
        }

        .winner-content h1 {
            font-size: 3.5em !important;
            color: #38bdf8 !important; /* Blue for the winner message */
            margin-bottom: 20px;
            text-shadow: 0 0 10px rgba(56, 189, 248, 0.5); 
        }
        .winner-content h2 {
            font-size: 2em !important;
            color: white !important; /* Final Scores text in white */
            margin-top: 10px;
        }
        .winner-content p {
            font-size: 1.4em !important;
            color: white !important; /* Score lines in white */
            line-height: 1.8;
        }
        
        /* Animation keyframes */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes zoomIn {
            from { transform: scale(0.8); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
        @keyframes pulse {
            0% { transform: scale(1); text-shadow: 0 0 20px rgba(255, 255, 255, 0.8); }
            50% { transform: scale(1.05); text-shadow: 0 0 30px rgba(255, 255, 255, 1); }
            100% { transform: scale(1); text-shadow: 0 0 20px rgba(255, 255, 255, 0.8); }
        }

        /* subtle separator */
        .chat-sep { height: 1px; background: #1f2937; margin: 10px 0; border-radius: 2px; }

        /* small muted */
        .small-muted { color: #9aa4b2; font-size: 0.9em; }

        /* responsivity */
        @media (max-width: 600px) {
            .chat-box, .judge-box { width: 94%; padding: 12px; }
            .bubble-left, .bubble-right { max-width: 100%; padding: 10px; }
            .avatar { width: 32px; height: 32px; font-size: 1rem; }
            .message-content { max-width: 80%; }
        }
        
        /* --- BUTTON COLOR FIX (Crucial Change) --- */
        /* Targets ALL primary buttons and forces them to the blue theme color */
        
        .stButton button[kind="primary"] {
            background-color: #38bdf8 !important; /* Blue background */
            border-color: #38bdf8 !important; /* Blue border */
            color: #0d1117 !important; /* Dark text for contrast */
        }
        
        .stButton button[kind="primary"]:hover {
            background-color: #1e88e5 !important; /* Slightly darker blue on hover */
            border-color: #1e88e5 !important;
        }
        /* --- END BUTTON COLOR FIX --- */
    </style>
"""
st.markdown(STYLING_CSS, unsafe_allow_html=True)

# --- 3. Header ---
def render_header():
    st.markdown(
        """
        <div style="display:flex; align-items:center; padding:6px 0 0 0;">
            <h1 style="
                margin:0; font-size:2.6em; font-weight:800;
                background: linear-gradient(90deg,#38bdf8,#818cf8);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            ">DebateMind</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div style='height:3px; width:100%; margin:6px 0 14px; background: radial-gradient(circle at center, #38bdf8 0%, rgba(22,27,34,0) 70%);'></div>", unsafe_allow_html=True)

render_header()


# New state for winner popup
if "show_winner_popup" not in st.session_state:
    st.session_state.show_winner_popup = False
if "winner_info" not in st.session_state:
    st.session_state.winner_info = None
if "latest_judge_data" not in st.session_state:
    st.session_state.latest_judge_data = None
    
if "mid_argument_counter" not in st.session_state:
    st.session_state.mid_argument_counter = 0

if "latest_judge_interruption_data" not in st.session_state:
    st.session_state.latest_judge_interruption_data = None


# --- PDF Auto-Extraction Section ---
st.markdown("## Add custom context to the Debate")

st.markdown("""
    <style>
    /* Change the border color and background of file uploader */
    div[data-testid="stFileUploader"] > section {
        background-color: #1f2937;
        border-left: 4px solid #38bdf8;
        border-radius: 10px;
        padding: 10px;
    }

    /* Change the text color and font */
    div[data-testid="stFileUploader"] label {
        /* Adjusted color to be visible on dark background, assuming Streamlit handles file uploader label style */
        color: #f0f4f8 !important; 
        font-weight: 600;
    }

    /* On hover */
    div[data-testid="stFileUploader"] > section:hover {
        background-color: #1f2937; /* lighter hover */
        border-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_pdf:
    try:
        # Define your output path (ensure /data exists)
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "extracted_text.txt")

        # Extract text
        pdf_reader = PdfReader(uploaded_pdf)
        extracted_text = ""

        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"

        # Save text automatically
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        
        # --- START: Pop-up Confirmation Feature ---
        st.success("✅ **PDF context saved!** The judge can now use this material to evaluate arguments.")
        with st.expander("Preview Extracted Text"):
            # Show the first 1000 characters for confirmation
            preview = extracted_text[:1000] if extracted_text else "No extractable text found on the PDF pages."
            st.code(preview)
        # --- END: Pop-up Confirmation Feature ---

    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")


# --- 4. Session State Setup for chat + animation ---
if "page" not in st.session_state:
    st.session_state.page = "Debate Arena"

if "current_debate_id" not in st.session_state: # <-- ADD THIS
    st.session_state.current_debate_id = None

if "debate_active" not in st.session_state:
    st.session_state.debate_active = False
    st.session_state.round = 0
    st.session_state.topic = ""
    st.session_state.history = []

if "view_round_idx" not in st.session_state:
    st.session_state.view_round_idx = None
if "history_search" not in st.session_state:
    st.session_state.history_search = ""

# Chat-related states:
# chat_history: list of dicts {"speaker":"coach"/"opponent", "text":..., "round":int, "type":"round"/"input"}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# [MODIFIED] New states for streaming text
if "stream_round_idx" not in st.session_state:
    st.session_state.stream_round_idx = -1 # -1 means not streaming
if "stream_speaker" not in st.session_state:
    st.session_state.stream_speaker = None # 'coach' or 'opponent'

# Helper: rebuild chat history from stored CSV (read_debate)
def rebuild_chat_from_storage(debate_id: str):
    if not debate_id: # <-- ADD THIS CHECK
        return []
    df = read_debate(debate_id)
    # Ensure chronological order by round ascending
    if df is None or df.empty:
        return []
    rows = []
    for _, r in df.sort_values("round", ascending=True).iterrows():
        # keep the same structure as append_chat_messages expects
        rows.append({"speaker": "coach", "text": str(r.get("coached_argument","")), "round": int(r["round"]), "type": "round"})
        rows.append({"speaker": "opponent", "text": str(r.get("opponent_argument","")), "round": int(r["round"]), "type": "round"})
    return rows

# If no chat_history yet, populate from storage
# [MODIFIED] No longer sets animation targets
if not st.session_state.chat_history:
    st.session_state.chat_history = rebuild_chat_from_storage(st.session_state.current_debate_id)

# Helper to append new round messages (UI-only)
# [MODIFIED] This function just appends to the list now
def append_chat_for_round(coached: str, opponent: str, round_internal: int, type_val: str = "round"):
    # Append to session chat history (two messages)
    st.session_state.chat_history.append({"speaker":"coach", "text": coached, "round": round_internal, "type": type_val})
    st.session_state.chat_history.append({"speaker":"opponent", "text": opponent, "round": round_internal, "type": type_val})


# --- 5. Sidebar (unchanged functionality, trimmed UI controls removed) ---
with st.sidebar:
    st.markdown("<h2 style='color:#f0f4f8; margin-bottom:0px;'>Control Panel</h2>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("<h3>Debate Topic:</h3>", unsafe_allow_html=True)
    topic_input = st.text_input(
        label="",
        value=st.session_state.topic or "Is remote work better than office work?",
        placeholder="Enter your debate topic here...",
        key="topic_input_key"
    )

    st.markdown("<h3>Total Rounds</h3>", unsafe_allow_html=True)
    max_rounds_input = st.number_input(
        label="",
        min_value=1,
        max_value=MAX_ROUNDS,
        value=MAX_ROUNDS,
        key="max_rounds_input_key"
    )

    if st.button("Start / Reset Simulation", use_container_width=True, key="start_debate_btn", type="primary"):
        if topic_input:
            st.cache_data.clear()
            clear_pdf_context()
            topic = sanitize_topic(topic_input)
            new_debate_id = create_new_debate(topic)
            st.session_state.current_debate_id = new_debate_id
            # reset everything (storage kept as is; we reset UI & session)
            st.session_state.debate_active = True
            st.session_state.round = 0
            st.session_state.topic = topic
            st.session_state.history = []
            st.session_state.view_round_idx = None
            st.session_state.history_search = ""
            st.session_state.show_winner_popup = False # Reset popup
            st.session_state.winner_info = None
            # reset chat UI
            st.session_state.chat_history = []
            st.session_state.mid_argument_counter = 0 
            st.session_state.latest_judge_interruption_data = None # RESET INTERRUPT JUDGE
            
            # [MODIFIED] Reset streaming state
            st.session_state.stream_round_idx = -1
            st.session_state.stream_speaker = None

            with st.spinner("Starting new simulation and drafting Round 1 arguments..."):
                if 1 <= max_rounds_input:
                    round_no = st.session_state.round
                    st.session_state.round += 1

                    template_idx, template_text = rl.select()
                    df = read_debate(st.session_state.current_debate_id) 
                    prev_args = df["coached_argument"].dropna().astype(str).tolist() if not df.empty else []

                    try:
                        coached = generate_coached_argument(template_text, st.session_state.topic, previous=prev_args)
                    except Exception as e:
                        # show the real error to the UI so you can debug (also saved in data/llm_last_response.json by call_openrouter)
                        coached = f"Error generating coached argument: {e}"
                        # optional: log to file
                        try:
                            os.makedirs("data", exist_ok=True)
                            with open("data/llm_coach_error.txt", "a", encoding="utf-8") as fh:
                                fh.write(f"{time.ctime()}: {str(e)}\n")
                        except Exception:
                            pass


                    try:
                        opponent = generate_opponent_argument(coached, st.session_state.topic)
                    except Exception as e:
                        opponent = f"Opponent generation failed: {e}"

                    try:
                        judge_scores = evaluate(coached, opponent, st.session_state.topic)
                    except Exception as e:
                        judge_scores = {
                            "total_coached": 5.0,
                            "total_opponent": 5.0,
                            "notes_coached": f"Judge error: {e}",
                            "notes_opponent": "Fallback evaluation."
                        }

                    reward = float(judge_scores.get("total_coached", 0)) - float(judge_scores.get("total_opponent", 0))

                    append_round(st.session_state.current_debate_id,{
                        "round": round_no,
                        "speaker": "coached",
                        "coached_argument": coached,
                        "opponent_argument": opponent,
                        "action": str(template_idx),
                        "reward": reward
                    })
                    judge_scores['round'] = round_no # Add the round number to the dictionary
                    append_judge(st.session_state.current_debate_id, judge_scores) # Pass only two arguments
                    st.session_state.latest_judge_data = judge_scores
                    st.session_state.latest_judge_interruption_data = None # RESET INTERRUPT JUDGE

                    try:
                        rl.update(template_idx, reward)
                    except Exception as e:
                        st.warning(f"RL update failed: {e}")

                    # Append chat messages UI-only
                    append_chat_for_round(coached, opponent, round_no)
                    
                    # [MODIFIED] Set streaming state to start animation
                    st.session_state.stream_round_idx = round_no
                    st.session_state.stream_speaker = 'coach'

                    st.toast("Simulation started and Round 1 generated.", icon="✅")
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.error("Total Rounds must be at least 1.")
        else:
            st.error("Please enter a debate topic to start.")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Navigation buttons
    if st.session_state.page == "Dashboard":
        st.button("Go to Dashboard", use_container_width=True, key="nav_dashboard_btn_active", disabled=True)
    else:
        if st.button("Go to Dashboard", use_container_width=True, key="nav_dashboard_btn_inactive"):
            st.session_state.page = "Dashboard"
            st.session_state.show_winner_popup = False
            st.rerun()

    if st.session_state.page == "Debate Arena":
        st.button("Back to Arena", use_container_width=True, key="nav_arena_btn_active", disabled=True)
    else:
        if st.button("Back to Arena", use_container_width=True, key="nav_arena_btn_inactive"):
            st.session_state.page = "Debate Arena"
            st.session_state.show_winner_popup = False
            st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Reset storage (recreate CSVs & clear history)
    if st.button("Delete Chat History", use_container_width=True, key="reset_storage_btn", type="primary"):
        init_storage()
        st.cache_data.clear()
        st.session_state.debate_active = False
        st.session_state.round = 0
        st.session_state.topic = ""
        st.session_state.history = []
        st.session_state.view_round_idx = None
        st.session_state.history_search = ""
        st.session_state.chat_history = []
        st.session_state.show_winner_popup = False # Reset popup
        st.session_state.winner_info = None
        st.session_state.current_debate_id = None
        st.session_state.mid_argument_counter = 0 
        st.session_state.latest_judge_interruption_data = None # RESET INTERRUPT JUDGE
        
        # [MODIFIED] Reset streaming state
        st.session_state.stream_round_idx = -1
        st.session_state.stream_speaker = None
        
        st.toast("Storage reset successful.")
        st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # History quick jump (keeps existing behaviour)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # --- NEW: Debate History Loader ---
    st.markdown("<h3>Load Past Debate</h3>", unsafe_allow_html=True)
    
    debate_ids = list_debates()
    
    if not debate_ids:
        st.info("No past debates found. Start a new one!")
    else:
        # Create a dictionary to map clean names to ugly IDs
        # "remote-work (10-31-2025)" -> "remote-work_1678886400"
        debate_options = {}
        for debate_id in debate_ids:
            try:
                parts = debate_id.rsplit('_', 1)
                topic_name = parts[0].replace('-', ' ').title()
                timestamp = int(parts[1])
                date = pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M')
                display_name = f"{topic_name} ({date})"
                debate_options[display_name] = debate_id
            except Exception:
                # Fallback for any weird folder names
                debate_options[debate_id] = debate_id

        selected_display_name = st.selectbox(
            label="Select a debate to load:",
            options=debate_options.keys()
        )
        
        if st.button("Load Debate", use_container_width=True, key="load_debate_btn"):
            selected_debate_id = debate_options[selected_display_name]
            
            st.cache_data.clear() # Clear cache
            
            # Load this debate into session state
            st.session_state.current_debate_id = selected_debate_id
            st.session_state.debate_active = True
            
            # Get topic and round from the loaded data
            df = read_debate(selected_debate_id)
            jd = read_judge(selected_debate_id)
            
            # Rebuild chat
            st.session_state.chat_history = rebuild_chat_from_storage(selected_debate_id)
            
            # Set topic
            st.session_state.topic = debate_options[selected_display_name].split(' (')[0]
            
            # Set round number
            st.session_state.round = int(df["round"].max() + 1) if not df.empty else 0
            
            # Reset streaming state
            st.session_state.stream_round_idx = -1
            st.session_state.stream_speaker = None
            st.session_state.latest_judge_interruption_data = None # RESET INTERRUPT JUDGE
            
            st.session_state.page = "Debate Arena"
            st.toast(f"Loaded debate: {selected_display_name}", icon="📚")
            st.rerun()

    # Data export
    st.markdown("<h3>Data Export</h3>", unsafe_allow_html=True)
    df_hist = read_debate(st.session_state.current_debate_id)
    if not df_hist.empty:
        csv_export = df_hist.to_csv(index=False)
        st.download_button("Download History (CSV)", csv_export, file_name="debate_history.csv", mime="text/csv", use_container_width=True, type="primary")
        json_export = json.dumps(json.loads(df_hist.to_json(orient="records")), indent=2)
        st.download_button("Download History (JSON)", json_export, file_name="debate_history.json", mime="application/json", use_container_width=True, type="primary")

# --- 6. Main content (Debate Arena with centered chat + judge box) ---

# [NEW] Helper to render a full, non-streaming message
def render_full_message_bubble(m: dict):
    """Renders a complete message bubble (not streaming)."""
    speaker = m["speaker"]
    text = m["text"]
    rnd = int(m.get("round", 0)) + 1
    msg_type = m.get("type", "round")
    
    # Sanitize text for HTML rendering
    text = text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    
    # Use a custom label for mid-argument inputs
    round_label = f"Round {rnd}" if msg_type == "round" else f"**User Interruption** ({st.session_state.mid_argument_counter})"

    if speaker == "coach":
        meta_html = f"<div class='msg-meta'>Coach • {round_label}</div>"
        html = f"""
        <div class='message-row message-row-left'>
            <div class='avatar avatar-coach'>C</div>
            <div class='message-content'>
                {meta_html}
                <div class='bubble-left'>
                    <div>{text}</div>
                </div>
            </div>
        </div>
        """
    else: # Opponent
        meta_html = f"<div class='msg-meta'>Opponent • {round_label}</div>"
        html = f"""
        <div class='message-row message-row-right'>
            <div class='message-content'>
                {meta_html}
                <div class='bubble-right'>
                    <div>{text}</div>
                </div>
            </div>
            <div class='avatar avatar-opponent'>O</div>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)

# [NEW] Helper generator to stream text as HTML
def stream_html_generator(full_text: str, speaker: str, round_num: int, msg_type: str = "round", word_delay=0.03):
    """Yields complete HTML bubbles with progressively more text."""
    
    streamed_text = ""
    # Sanitize and split
    words = full_text.replace("<", "&lt;").replace(">", "&gt;").split(' ')
    
    # Use a custom label for mid-argument inputs
    round_label = f"Round {round_num}" if msg_type == "round" else f"**User Interruption** ({st.session_state.mid_argument_counter})"

    if speaker == "coach":
        meta_html = f"<div class='msg-meta'>Coach • {round_label}</div>"
        for word in words:
            streamed_text += word + " "
            html = f"""
            <div class='message-row message-row-left'>
                <div class='avatar avatar-coach'>C</div>
                <div class='message-content'>
                    {meta_html}
                    <div class='bubble-left'>
                        <div>{streamed_text.replace("\n", "<br>")}</div>
                    </div>
                </div>
            </div>
            """
            yield html
            time.sleep(word_delay)
            
    else: # Opponent
        meta_html = f"<div class='msg-meta'>Opponent • {round_label}</div>"
        for word in words:
            streamed_text += word + " "
            html = f"""
            <div class='message-row message-row-right'>
                <div class='message-content'>
                    {meta_html}
                    <div class='bubble-right'>
                        <div>{streamed_text.replace("\n", "<br>")}</div>
                    </div>
                </div>
                <div class='avatar avatar-opponent'>O</div>
            </div>
            """
            yield html
            time.sleep(word_delay)
            
# <<< CORRECTED MID-ARGUMENT HANDLER FUNCTION >>>
def handle_mid_argument_input(input_point: str, topic: str):
    """Generates and streams a short response based on user input and calls the judge."""
    st.session_state.mid_argument_counter += 1
    input_count = st.session_state.mid_argument_counter
    
    # 1. Coach responds to the input
    with st.spinner(f"Coach is responding to user input #{input_count}..."):
        try:
            # Use a short argument generation prompt specific to the interruption
            coach_prompt = f"The user introduced the point: '{input_point}'. Respond briefly (max 80 words) by integrating this point into the debate topic: '{topic}' and support your side."
            coached_response = generate_coached_argument(coach_prompt, topic, previous=[st.session_state.chat_history[-1]['text']] if st.session_state.chat_history else [])
        except Exception as e:
            coached_response = f"Coach failed to process input: {e}"

    # 2. Opponent responds to the Coach's response + input
    with st.spinner(f"Opponent is countering user input #{input_count}..."):
        try:
            opponent_prompt = f"The user introduced the point: '{input_point}'. The Coach just responded: '{coached_response}'. Counter this point briefly (max 80 words) and defend your side."
            opponent_response = generate_opponent_argument(opponent_prompt, topic)
        except Exception as e:
            opponent_response = f"Opponent failed to process input: {e}"
            
    # 3. Judge re-evaluates the two new arguments
    with st.spinner(f"Judge is evaluating the interruption..."):
        try:
            # FIX: Create a single, augmented topic string for the judge
            # This passes the necessary context without the illegal 'context' argument.
            augmented_topic_for_judge = f"TOPIC: {topic}. CONTEXT: The user just interrupted the debate with the point: '{input_point}'. Judge the following arguments based on this new point and the main topic."
            
            judge_scores = evaluate(coached_response, opponent_response, augmented_topic_for_judge)
            
        except Exception as e:
            # Fallback evaluation, ensuring the error is recorded in the notes
            judge_scores = {
                "total_coached": 5.0,
                "total_opponent": 5.0,
                "notes_coached": f"Judge ERROR: evaluate() function failed. Check backend/judge.py. Error: {e}",
                "notes_opponent": "Fallback evaluation."
            }

    # Store the judge data in the new dedicated state
    st.session_state.latest_judge_interruption_data = judge_scores
    
    # Use a dummy round number (e.g., current round number + 1000) for UI ordering but label it as input
    dummy_round_num = st.session_state.round + 1000 + input_count
    
    st.session_state.chat_history.append({"speaker":"coach", "text": coached_response, "round": dummy_round_num, "type": "input"})
    st.session_state.chat_history.append({"speaker":"opponent", "text": opponent_response, "round": dummy_round_num, "type": "input"})

    # Set streaming state to start animation for the Coach's response
    st.session_state.stream_round_idx = dummy_round_num
    st.session_state.stream_speaker = 'coach'
    
    st.toast(f"User input received and debated in Interruption {input_count}.", icon="💬")
    st.rerun()
# <<< END CORRECTED MID-ARGUMENT HANDLER FUNCTION >>>


if st.session_state.page == "Debate Arena":
    st.markdown("<h2>DEBATE ARENA: Real-Time Simulation</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a6b1bf;'>Monitor the arguments and the judge's real-time evaluations.</p>", unsafe_allow_html=True)
    st.divider()

    if not st.session_state.debate_active:
        st.info("Start a new debate from the sidebar to begin the simulation and view the Round Viewer.")
        # Clear interruption data if debate is inactive
        st.session_state.latest_judge_interruption_data = None
    else:
        st.subheader(f"Topic: **{st.session_state.topic}**")
        
        # Clear interruption data if we are generating a new full round
        if st.session_state.stream_round_idx > -1 and st.session_state.chat_history and st.session_state.chat_history[-1].get("type", "round") == "round":
             st.session_state.latest_judge_interruption_data = None


        max_r = float(max_rounds_input)
        display_current_round = st.session_state.round
        progress_val = (display_current_round / max_r) if max_r > 0 else 0.0
        progress_val = min(max(progress_val, 0.0), 1.0)

        progress_container = st.container()
        with progress_container:
            col_p1, col_p2 = st.columns([3, 1])
            with col_p1:
                st.metric(
                    label="Current Round Progress",
                    value=f"Round {display_current_round} / {int(max_r)}",
                    delta=f"{progress_val*100:.0f}% Complete"
                )
                next_round_num = display_current_round + 1
                progress_text = f"Simulating Round {next_round_num}..." if display_current_round < max_r else "Simulation complete."
                st.progress(progress_val, text=progress_text)
            with col_p2:
                is_finished = display_current_round >= max_r
                
                # [MODIFIED] Disable "NEXT ROUND" button while streaming
                is_streaming = st.session_state.stream_round_idx != -1
                
                if is_finished:
                    st.success("Simulation complete! View results on the Dashboard.")
                    if st.button("View Final Results", use_container_width=True, key="view_final_arena", type="primary"):
                        st.session_state.page = "Dashboard"
                        # --- WINNER CALCULATION AND POPUP TRIGGER ---
                        jd_final = read_judge(st.session_state.current_debate_id) # Read final judge data
                        if not jd_final.empty:
                            total_coached_score = pd.to_numeric(jd_final["total_coached"], errors='coerce').sum()
                            total_opponent_score = pd.to_numeric(jd_final["total_opponent"], errors='coerce').sum()
                            
                            winner_message = ""
                            if total_coached_score > total_opponent_score:
                                winner_message = "Coach Debater Wins!"
                            elif total_opponent_score > total_coached_score:
                                winner_message = "Opponent Wins!"
                            else:
                                winner_message = "It's a Tie!"
                            
                            st.session_state.winner_info = {
                                "winner_message": winner_message,
                                "coach_score": total_coached_score,
                                "opponent_score": total_opponent_score
                            }
                            st.session_state.show_winner_popup = True
                        else:
                            st.warning("No judge data available to determine a winner.")
                        st.rerun()
                else:
                    if st.button("NEXT ROUND", use_container_width=True, key="next_round_btn_primary", type="primary", disabled=is_streaming):
                        with st.spinner(f"Drafting Round {display_current_round + 1} arguments..."):
                            round_to_generate_internal = st.session_state.round
                            template_idx, template_text = rl.select()
                            df = read_debate(st.session_state.current_debate_id)
                            prev_args = df["coached_argument"].dropna().astype(str).tolist() if not df.empty else []

                            try:
                                coached = generate_coached_argument(template_text, st.session_state.topic, previous=prev_args)
                                if not coached:
                                    coached = "Coached model returned no valid response."
                            except Exception as e:
                                coached = f"Error generating coached argument: {e}"

                            try:
                                opponent = generate_opponent_argument(coached, st.session_state.topic)
                            except Exception as e:
                                opponent = f"Opponent generation failed: {e}"

                            try:
                                judge_scores = evaluate(coached, opponent, st.session_state.topic)
                            except Exception as e:
                                judge_scores = {
                                    "total_coached": 5.0,
                                    "total_opponent": 5.0,
                                    "notes_coached": f"Judge error: {e}",
                                    "notes_opponent": "Fallback evaluation."
                                }

                            reward = float(judge_scores.get("total_coached", 0)) - float(judge_scores.get("total_opponent", 0))

                            append_round(st.session_state.current_debate_id,{
                                "round": round_to_generate_internal,
                                "speaker": "coached",
                                "coached_argument": coached,
                                "opponent_argument": opponent,
                                "action": str(template_idx),
                                "reward": reward
                            })
                            judge_scores['round'] = round_to_generate_internal # Add the round number to the dictionary
                            append_judge(st.session_state.current_debate_id, judge_scores) # Pass only two arguments

                            st.session_state.latest_judge_data = judge_scores
                            st.session_state.latest_judge_interruption_data = None # RESET INTERRUPT JUDGE


                            try:
                                rl.update(template_idx, reward)
                            except Exception as e:
                                st.warning(f"RL update failed: {e}")

                            # increment completed round counter
                            st.session_state.round += 1

                            # Append chat messages UI-only
                            append_chat_for_round(coached, opponent, round_to_generate_internal)
                            
                            # [MODIFIED] Set streaming state
                            st.session_state.stream_round_idx = round_to_generate_internal
                            st.session_state.stream_speaker = 'coach'

                            st.toast(f"Round {display_current_round + 1} completed.", icon="✅")
                            time.sleep(0.2)
                            st.session_state.view_round_idx = None
                            st.session_state.show_winner_popup = False # Ensure popup is off
                            st.session_state.winner_info = None # Clear winner info
                            st.rerun()
                            
        # <<< NEW MID-ARGUMENT FEATURE UI >>>
        if not is_finished and not is_streaming:
            st.divider()
            st.markdown("<h3>🎯 Inject a Mid-Argument Point</h3>", unsafe_allow_html=True)
            col_input, col_button = st.columns([4, 1])
            
            with col_input:
                # Use a specific key here to ensure the value resets on inject button press
                user_input_point = st.text_input(
                    label="Enter a new key point to introduce to the debate:",
                    placeholder="e.g., 'The environmental cost of daily commuting should be factored in.'",
                    key="mid_argument_input_key"
                )
            
            with col_button:
                st.markdown("<div style='height: 29px;'>&nbsp;</div>", unsafe_allow_html=True) # Vertical alignment spacer
                if st.button("Inject Point", use_container_width=True, key="inject_point_btn", type="primary", disabled=not user_input_point):
                    if st.session_state.current_debate_id:
                        handle_mid_argument_input(user_input_point, st.session_state.topic)
                    else:
                        st.error("Please start a debate first.")
        # <<< END NEW MID-ARGUMENT FEATURE UI >>>
        
        st.divider()

        # --- [MODIFIED] Chat Rendering with Streaming ---
        st.markdown("<div class='chat-wrapper'><div class='chat-box'>", unsafe_allow_html=True)

        messages_to_render = st.session_state.chat_history
        streaming_round = st.session_state.stream_round_idx
        streaming_speaker = st.session_state.stream_speaker
        
        # Flags to trigger a rerun after a stream completes
        coach_stream_complete = False
        opponent_stream_complete = False

        for m in messages_to_render:
            msg_round = m['round']
            msg_speaker = m['speaker']
            
            # Check if this is the exact message we are supposed to be streaming
            is_streaming_message = (msg_round == streaming_round) and (msg_speaker == streaming_speaker)

            if is_streaming_message:
                # Render the message using the streaming generator
                placeholder = st.empty()
                stream_generator = stream_html_generator(
                    full_text=m["text"],
                    speaker=msg_speaker,
                    round_num=int(m.get("round", 0)) + 1, # Pass the original round number
                    msg_type=m.get("type", "round") 
                )
                
                # This loop will run, replacing the content of `placeholder`
                for html_chunk in stream_generator:
                    placeholder.markdown(html_chunk, unsafe_allow_html=True)
                
                # Stream for this message is done. Update state.
                if msg_speaker == "coach":
                    # If this was a mid-argument input, the next message to stream is the opponent's
                    next_message_speaker = 'opponent'
                    st.session_state.stream_speaker = next_message_speaker
                    coach_stream_complete = True

                elif msg_speaker == "opponent":
                    # Opponent stream finishes the interruption/round
                    st.session_state.stream_round_idx = -1
                    st.session_state.stream_speaker = None
                    opponent_stream_complete = True
            
            else:
                # This is not the message being streamed, so render it fully
                render_full_message_bubble(m)

        st.markdown("</div></div>", unsafe_allow_html=True)  # close chat-box & wrapper
        
        # --- [NEW] Interruption Judge Panel Logic ---
        if st.session_state.stream_round_idx == -1 and st.session_state.latest_judge_interruption_data:
            latest_judge = st.session_state.latest_judge_interruption_data
            
            judge_html = f"""
                <div class='judge-wrapper'>
                    <div class='judge-box judge-interruption'>
                        <div style='font-weight:700; margin-bottom:6px;'>Judge Re-evaluation — User Interruption ({st.session_state.mid_argument_counter})</div>
                        <div>
                            <b>Coach Score:</b> <span style="color:#38bdf8; font-weight:700;">{format_score_as_points(latest_judge.get('total_coached', 'N/A'))}/10</span> 
                            &nbsp; | &nbsp; 
                            <b>Opponent Score:</b> <span style="color:#ef4444; font-weight:700;">{format_score_as_points(latest_judge.get('total_opponent', 'N/A'))}/10</span>
                        </div>
                        <div class='chat-sep'></div>
                        <div><b>Judge Notes (Coach):</b> {latest_judge.get('notes_coached','No notes provided.')}</div>
                        <div style='margin-top:6px;'><b>Judge Notes (Opponent):</b> {latest_judge.get('notes_opponent','No notes provided.')}</div>
                    </div>
                </div>
            """
            st.markdown(judge_html, unsafe_allow_html=True)


        # --- [MODIFIED] Regular Round Judge Panel Logic (only show if latest message was a full round) ---
        jd = read_judge(st.session_state.current_debate_id)
        df = read_debate(st.session_state.current_debate_id)
        
        is_latest_message_round = (not st.session_state.chat_history) or (st.session_state.chat_history[-1].get("type", "round") == "round")
        
        # Only show the judge box if we are NOT streaming AND the latest content was a full round
        if st.session_state.stream_round_idx == -1 and st.session_state.stream_speaker is None and is_latest_message_round:
            if (not jd.empty) and (len(st.session_state.chat_history) > 0):
                # determine latest stored round number
                latest_round_internal = int(df["round"].max()) if not df.empty else None
                if latest_round_internal is not None:
                    # fetch latest judge entry (match by round)
                    latest_judge_row = jd[jd["round"].astype(int) == latest_round_internal]
                    if latest_judge_row.empty:
                        latest_judge = jd.iloc[-1].to_dict()  # fallback
                    else:
                        latest_judge = latest_judge_row.iloc[-1].to_dict()

                    # Render judge box centered and same width as chatbox
                    judge_html = f"""
                        <div class='judge-wrapper'>
                            <div class='judge-box'>
                                <div style='font-weight:700; margin-bottom:6px;'>Judge Evaluation — Round {int(latest_judge.get('round', 0)) + 1}</div>
                                <div><b>Coach:</b> {format_score_as_points(latest_judge.get('total_coached', 'N/A'))}/10 &nbsp; | &nbsp; <b>Opponent:</b> {format_score_as_points(latest_judge.get('total_opponent', 'N/A'))}/10</div>
                                <div class='chat-sep'></div>
                                <div><b>Notes (Coach):</b> {latest_judge.get('notes_coached','No notes provided.')}</div>
                                <div style='margin-top:6px;'><b>Notes (Opponent):</b> {latest_judge.get('notes_opponent','No notes provided.')}</div>
                            </div>
                        </div>
                    """
                    st.markdown(judge_html, unsafe_allow_html=True)

        # --- [NEW] Rerun Trigger ---
        # If a stream just finished, trigger a rerun to stream the next part or show the judge
        if coach_stream_complete or opponent_stream_complete:
            st.rerun()

        # quick fallback history expander (keeps previous quick-history view)
        with st.expander("Quick history (latest rounds)", expanded=False):
            df = read_debate(st.session_state.current_debate_id)
            recent = df.tail(5).reset_index(drop=True)
            for _, r in recent.iterrows():
                st.markdown(f"**Round {int(r['round'] + 1)}** --- <span class='small-muted'>{str(r.get('coached_argument',''))[:120].replace(chr(10),' ') + ('...' if len(str(r.get('coached_argument','')))>120 else '')}</span>", unsafe_allow_html=True)
            st.download_button("Download full history (CSV)", df.to_csv(index=False), file_name="debate_history.csv", mime="text/csv", use_container_width=True, key="arena_download_full_hist", type="primary")


# --- Dashboard (unchanged except style) ---
elif st.session_state.page == "Dashboard":
    st.markdown("<h2>PROGRESS DASHBOARD: AI Learning Metrics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a6b1bf;'>Monitor the Reinforcement Learning Agent's performance and strategy effectiveness.</p>", unsafe_allow_html=True)
    st.divider()

    jd = read_judge(st.session_state.current_debate_id)
    df = read_debate(st.session_state.current_debate_id)

    if st.session_state.current_debate_id is None:
        st.info("Please start a new debate or load a past debate from the sidebar.")
    elif jd.empty or df.empty:
        st.warning("This debate has no rounds yet. Run a few rounds in the Arena.")
    else:
        avg_coached = pd.to_numeric(jd["total_coached"], errors='coerce').mean()
        avg_opponent = pd.to_numeric(jd["total_opponent"], errors='coerce').mean()
        avg_reward = pd.to_numeric(df["reward"], errors='coerce').mean()
        total_rounds = len(jd)

        st.subheader("Key Performance Indicators")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                label="Avg. Debater Score (Coach)",
                value=f"{avg_coached:.2f}",
                delta=f"{avg_coached - avg_opponent:+.2f} vs Opponent",
                delta_color="normal" if avg_coached >= avg_opponent else "inverse"
            )

        with c2:
            st.metric(
                label="Avg. Strategy Success (Reward)",
                value=f"{avg_reward:+.2f}",
                delta="Success Rate",
                delta_color="off"
            )

        with c3:
            st.metric(
                label="Total Simulated Rounds",
                value=f"{total_rounds}",
                delta="History Size",
                delta_color="off"
            )

        st.divider()

        st.subheader("Performance Visualizations")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.caption("Score Trend: Debater vs Opponent (By Round)")
            plot_df = jd.copy()
            plot_df["round_display"] = plot_df["round"] + 1
            chart_data = plot_df[["round_display", "total_coached", "total_opponent"]].melt(
                "round_display", var_name="Agent", value_name="Score"
            )
            chart_data["Agent"] = chart_data["Agent"].map({
                "total_coached": "Coach Debater (Blue)",
                "total_opponent": "Opponent (Red)"
            })
            st.line_chart(
                chart_data.set_index("round_display"),
                y="Score",
                color="Agent",
                height=350
            )

        with col_c2:
            st.caption("Average Reward by Strategy (Action)")
            strat_df = df.groupby("action")["reward"].mean().reset_index()
            strat_df.columns = ["Strategy Index", "Avg Reward"]
            st.bar_chart(strat_df.set_index("Strategy Index"), height=350)

        st.divider()

        with st.expander("Detailed Judge Summaries (By Round)", expanded=False):
            for index, row in jd.iloc[::-1].iterrows():
                
                # --- THIS IS THE FIX ---
                # Robustly convert scores to numbers, defaulting to 0.0 if it's bad text
                coach_score = pd.to_numeric(row['total_coached'], errors='coerce')
                opponent_score = pd.to_numeric(row['total_opponent'], errors='coerce')

                # Handle NaN values if conversion failed, set to 0.0
                coach_score = 0.0 if pd.isna(coach_score) else coach_score
                opponent_score = 0.0 if pd.isna(opponent_score) else opponent_score
                # -----------------------

                with st.container():
                    col_h1, col_h2, col_h3 = st.columns([1, 1, 1])
                    with col_h1:
                        st.markdown(f"**Round:** **{int(row['round'] + 1)}**")
                    with col_h2:
                        # Use the new, safe numeric variable
                        st.markdown(f"**Coach Score:** <span style='color:#38bdf8;'>**{coach_score:.2f}**</span>", unsafe_allow_html=True)
                    with col_h3:
                        # Use the new, safe numeric variable
                        st.markdown(f"**Opponent Score:** <span style='color:#ef4444;'>**{opponent_score:.2f}**</span>", unsafe_allow_html=True)

                    st.markdown("<hr style='border-top: 1px solid #30363d;'>", unsafe_allow_html=True)
                    st.markdown(f"**Judge's Feedback (Coach):** {row['notes_coached']}")
                    st.markdown(f"**Judge's Feedback (Opponent):** {row['notes_opponent']}")

        st.divider()

        st.subheader("Raw Simulation Data")
        with st.expander("View Full Round Details Table"):
            st.dataframe(df.assign(Round_Display=df["round"] + 1), use_container_width=True) 

# --- 7. WINNER POPUP RENDERING (STREAMLIT NATIVE DISMISS) ---
if st.session_state.show_winner_popup and st.session_state.winner_info:
    
    # 1. Prepare Data
    winner_info = st.session_state.winner_info
    winner_message = winner_info['winner_message']
    winner_message = winner_message.replace("🌟", "").replace("💥", "").replace("🤝", "").strip()
    
    # 2. Render Custom HTML Overlay (Visual Only)
    st.markdown(
        f"""
        <div class="winner-overlay" id="winner-overlay">
            <div class="winner-content">
                <h1>{winner_message}</h1>
                <h2 style='color: white !important;'>Final Scores:</h2> 
                <p>
                    <span style='color: #38bdf8;'>Coach Debater:</span> <b>{format_score_as_points(winner_info['coach_score'])}</b> <br>
                    <span style='color: #ef4444;'>Opponent:</span> <b>{format_score_as_points(winner_info['opponent_score'])}</b>
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 3. Render the Functional Streamlit Button (Visible and clickable when popup is active)
    # The button is rendered outside the HTML overlay but becomes the only clickable element
    # (since the overlay covers the rest of the screen).
    
    # Use st.container() and centering to make the button look good when it's the only element
    st.markdown("""
        <div style='display: flex; justify-content: center; margin-top: 20px; z-index: 1001;'>
            <div id='dismiss-button-container-outer' style='width: 300px;'></div>
        </div>
        """, unsafe_allow_html=True)

    # Use a placeholder to inject the button in the visible area
    button_placeholder = st.empty()
    with button_placeholder.container():
        # Adds space below the overlay for cleaner separation
        st.markdown("<h3 style='height: 40px;'>&nbsp;</h3>", unsafe_allow_html=True) 
        
        # Center the button using Streamlit columns
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            if st.button("Go to Dashboard & Dismiss Results", key="dismiss_winner_popup_btn", type="primary", use_container_width=True):
                # Functional Logic: Clears popup state and redirects page
                st.session_state.show_winner_popup = False
                st.session_state.winner_info = None
                st.session_state.page = "Dashboard"
                st.rerun()