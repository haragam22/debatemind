import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import json
import os
from PyPDF2 import PdfReader

# --- Preserve Backend Imports ---
from backend.memory_manager import init_files, read_debate, read_judge, append_round, append_judge
from backend.rl_agent import RLAgent
from backend.debater import generate_coached_argument
from backend.opponent import generate_opponent_argument
from backend.judge import evaluate
# IMPORTANT: Updated import to include load_pdf_context for clarity (though it's used in judge.py)
from backend.utils import sanitize_topic, load_pdf_context 
from backend.config import MAX_ROUNDS 

# --- Helper function for score formatting ---
def format_score_as_points(score_val):
    """Formats a score (expected 0-10) to points, assuming it's already clamped."""
    try:
        score = float(score_val)
        return f"{score:.1f}" # Display one decimal place for clarity
    except (ValueError, TypeError):
        return "N/A"

# --- 1. Initialization and Setup ---
init_files() 
rl = RLAgent() 

st.set_page_config(
    page_title="DebateMind: AI Coach", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. Custom Styling and Theme (UPDATED with Blue/Black Popup Styles) ---
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
        /* Sidebar Styling: Reduced Top Padding & Consistent Spacing */
        section[data-testid="stSidebar"] {
            background-color: #161b22; 
            border-right: 1px solid #30363d;
            padding-top: 1rem; 
        }
        /* Custom horizontal rule for clean separation */
        .sidebar-divider {
            margin: 8px 0; 
            border-top: 1px solid #30363d;
        }

        /* Metric Cards */
        [data-testid="stMetric"] {
            background: #1f2937;
            border-left: 4px solid #38bdf8;
            border-radius: 8px;
            padding: 1.2em;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }

        /* Input Fields */
        div[data-testid="stTextInput"] > div > input, 
        div[data-testid="stNumberInput"] input {
            background-color: #161b22 !important;
            color: #f0f4f8 !important;
            border-radius: 6px;
            border: 1px solid #4a5568;
            padding: 10px;
        }
        
        /* Navigation/History Buttons - General (Primary style remains bright blue) */
        div.stButton button {
            background-color: #21262d; /* Inactive buttons */
            color: #c9d1d9;
            border-radius: 8px;
            border: 1px solid #30363d;
            padding: 0.75rem 1rem;
            font-weight: 500;
            font-size: 1.1em;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
            height: 50px; 
            margin-bottom: 5px; 
        }
        div.stButton button:hover {
            background-color: #30363d; 
            color: #f0f4f8;
            border-color: #38bdf8; 
        }

        /* PRIMARY ACTION STYLING (BRIGHT BLUE) */
        div.stButton > button[kind="primary"] {
            background-color: #3b82f6;
            color: white;
            border: none;
            padding: 0.75rem 1.5rem; 
            font-weight: 700;
            font-size: 1.1em;
            height: auto; 
            margin-top: 5px; 
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #2563eb;
            border: none;
        }
        
        /* CUSTOM STYLING FOR ACTIVE NAVIGATION BUTTON (DISABLED STATE) */
        div.stButton > button[data-testid*="active"] {
            background: linear-gradient(90deg, #38bdf8, #3b82f6) !important;
            color: black !important;
            border: 2px solid #38bdf8 !important; 
            font-weight: 900 !important;
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.6);
            transform: scale(1.01);
        }

        /* Debate Dialogue Boxes */
        .debate-box {
            border-radius: 12px;
            padding: 1.5em;
            margin-top: 1.2em;
            line-height: 1.6;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        }
        .coach-box {
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid #3b82f6;
        }
        .opponent-box { 
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid #ef4444;
        }
        .judge-box { 
            background: rgba(251, 191, 36, 0.1);
            border: 1px solid #f59e0b;
        }
        .small-muted { color: #9aa4b2; font-size: 0.9em; }
        
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
        
    </style>
"""
st.markdown(STYLING_CSS, unsafe_allow_html=True)

# --- 3. Header Bar with Title (DebateMind) ---
def render_header():
    """Renders a sleek, bold header bar with the new project name."""
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; padding: 10px 0 0px 0;">
            <h1 style="
                margin: 0; 
                font-size: 3em; 
                font-weight: 800; 
                letter-spacing: 2px;
                background: linear-gradient(90deg, #38bdf8, #818cf8); 
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0px 0px 15px rgba(56, 189, 248, 0.5); 
            ">
            DebateMind 
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
        <div style="
            height: 3px; 
            background: radial-gradient(circle at center, #38bdf8 0%, rgba(22, 27, 34, 0) 70%); 
            width: 100%; 
            margin: 5px 0 20px 0;
        "></div>
    """, unsafe_allow_html=True)

render_header()

# --- 4. State Management ---
if "page" not in st.session_state:
    st.session_state.page = "Debate Arena"

if "debate_active" not in st.session_state:
    st.session_state.debate_active = False
    st.session_state.round = 0 # Internal counter for rounds completed (0 means before Round 1)
    st.session_state.topic = ""
    st.session_state.history = []

if "view_round_idx" not in st.session_state:
    st.session_state.view_round_idx = None
if "history_search" not in st.session_state:
    st.session_state.history_search = ""
    
# New state for winner popup
if "show_winner_popup" not in st.session_state:
    st.session_state.show_winner_popup = False
if "winner_info" not in st.session_state:
    st.session_state.winner_info = None

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


# --- 5. Sidebar Control Panel ---
with st.sidebar:
    st.markdown("<h2 style='color:#f0f4f8; margin-bottom: 0px;'>Control Panel</h2>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # --- 5.1 SIMULATION SETUP ---
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
    
    # START / RESET SIMULATION Button (Now triggers Round 1 generation)
    if st.button("Start / Reset Simulation", use_container_width=True, key="start_debate_btn", type="primary"):
        if topic_input:
            topic = sanitize_topic(topic_input)
            
            # --- Reset State ---
            st.session_state.debate_active = True
            st.session_state.round = 0 # Pre-start state
            st.session_state.topic = topic
            st.session_state.history = []
            st.session_state.view_round_idx = None
            st.session_state.history_search = ""
            st.session_state.show_winner_popup = False # Reset popup
            st.session_state.winner_info = None # Reset winner info

            # --- Auto-start Round 1 Logic (Copied from the NEXT ROUND button) ---
            with st.spinner("Starting new simulation and drafting Round 1 arguments..."):
                
                # Check if Round 1 is allowed
                if 1 <= max_rounds_input:
                    # Rerunning the NEXT ROUND logic inline:
                    
                    round_no = st.session_state.round # 0 for the first round
                    st.session_state.round += 1 # Increment to 1 immediately for next round state

                    # Generate arguments for Round 1
                    template_idx, template_text = rl.select()
                    df = read_debate()
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
                    
                    # Append Round 1 Data
                    append_round({
                        "round": round_no, 
                        "speaker": "coached", 
                        "coached_argument": coached, 
                        "opponent_argument": opponent, 
                        "action": str(template_idx), 
                        "reward": reward
                    })
                    append_judge(round_no, judge_scores)
                    
                    try:
                        rl.update(template_idx, reward)
                    except Exception as e:
                        st.warning(f"RL update failed: {e}")
                    
                    st.toast("Simulation started and Round 1 generated.", icon="✅")
                    time.sleep(0.5)
                    st.rerun() 
                else:
                    st.error("Total Rounds must be at least 1.")

        else:
            st.error("Please enter a debate topic to start.")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # --- 5.2 NAVIGATION BUTTONS ---
    
    # Go to Dashboard button
    if st.session_state.page == "Dashboard":
        st.button("Go to Dashboard", use_container_width=True, key="nav_dashboard_btn_active", disabled=True)
    else:
        if st.button("Go to Dashboard", use_container_width=True, key="nav_dashboard_btn_inactive"):
            st.session_state.page = "Dashboard"
            st.session_state.show_winner_popup = False # Close popup if navigating away
            st.rerun()

    # Back to Arena button
    if st.session_state.page == "Debate Arena":
        st.button("Back to Arena", use_container_width=True, key="nav_arena_btn_active", disabled=True)
    else:
        if st.button("Back to Arena", use_container_width=True, key="nav_arena_btn_inactive"):
            st.session_state.page = "Debate Arena"
            st.session_state.show_winner_popup = False # Close popup if navigating away
            st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    # --- 5.3 STORAGE (CSV) SECTION ---
    st.markdown("<h3>Storage (CSV)</h3>", unsafe_allow_html=True)
    st.markdown("<p>If previous runs persist, recreate the CSV files and clear caches.</p>", unsafe_allow_html=True)

    # Reset Storage Button
    if st.button("Reset Storage (recreate CSVs & clear history)", use_container_width=True, key="reset_storage_btn"):
        init_files() 
        st.session_state.debate_active = False
        st.session_state.round = 0 
        st.session_state.topic = ""
        st.session_state.history = []
        st.session_state.view_round_idx = None
        st.session_state.history_search = ""
        st.session_state.show_winner_popup = False # Reset popup
        st.session_state.winner_info = None # Reset winner info
        st.toast("Storage reset successful.")
        st.rerun()
        
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # --- 5.4 HISTORY QUICK JUMP ---
    st.markdown("<h3>History Quick Jump</h3>", unsafe_allow_html=True)
    df_hist = read_debate()
    
    if df_hist.empty:
        st.info("No rounds yet to jump to.")
    else:
        st.markdown("<h3>Open round:</h3>", unsafe_allow_html=True)
        
        # Build options
        def preview_text(row):
            txt = str(row.get("coached_argument", "") or "")
            txt = txt.replace("\n", " ")
            return (txt[:80] + "...") if len(txt) > 80 else txt

        options = []
        mapping = {}
        for _, r in df_hist.sort_values(by="round", ascending=False).iterrows():
            idx = int(r["round"])
            label = f"Round {idx + 1}" # Display as Round 1, 2, etc.
            options.append(label)
            mapping[label] = idx # Map back to internal 0-indexed round
        
        # Selectbox
        sel_label = st.selectbox(label="", options=options, key="sidebar_round_select_jump")

        # Open/Clear Buttons
        if st.button("Open selected", use_container_width=True, key="open_selected_btn_jump"): 
            st.session_state.view_round_idx = mapping[sel_label]
            st.session_state.page = "Debate Arena"
            st.session_state.show_winner_popup = False # Close popup if navigating away
            st.rerun()

        if st.button("Clear selection", use_container_width=True, key="back_to_live_sidebar_btn_jump"): 
            st.session_state.view_round_idx = None
            st.session_state.page = "Debate Arena"
            st.session_state.show_winner_popup = False # Close popup if navigating away
            st.rerun()
            
    # --- 5.5 DOWNLOAD HISTORY ---
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown("<h3>Data Export</h3>", unsafe_allow_html=True)
    
    if not df_hist.empty:
        csv_export = df_hist.to_csv(index=False)
        st.download_button("Download History (CSV)", csv_export, file_name="debate_history.csv", mime="text/csv", use_container_width=True, type="primary")
        
        json_export = json.dumps(json.loads(df_hist.to_json(orient="records")), indent=2)
        st.download_button("Download History (JSON)", json_export, file_name="debate_history.json", mime="application/json", use_container_width=True, type="primary")


# --- 6. Main Content Pages ---

## Debate Arena Page
if st.session_state.page == "Debate Arena":
    st.markdown("<h2>DEBATE ARENA: Real-Time Simulation</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a6b1bf;'>Monitor the arguments and the judge's real-time evaluations.</p>", unsafe_allow_html=True)
    st.divider()

    if not st.session_state.debate_active:
        # Show instruction only if debate is not active
        st.info("Start a new debate from the sidebar to begin the simulation and view the Round Viewer.")
    else:
        st.subheader(f"Topic: **{st.session_state.topic}**")

        # Use MAX_ROUNDS from sidebar input
        max_r = float(max_rounds_input)
        
        # Display current round (the one that has just been GENERATED)
        # Note: st.session_state.round is the count of rounds COMPLETED (starting at 1 after the first generation)
        display_current_round = st.session_state.round
        
        # Determine progress calculation based on completed rounds vs max rounds
        progress_val = (display_current_round / max_r) if max_r > 0 else 0.0
        progress_val = min(max(progress_val, 0.0), 1.0)
        
        progress_container = st.container(border=True)
        with progress_container:
            col_p1, col_p2 = st.columns([3, 1])
            with col_p1:
                st.metric(
                    label="Current Round Progress", 
                    value=f"Round {display_current_round} / {int(max_r)}",
                    delta=f"{progress_val*100:.0f}% Complete"
                )
                # The text displays the NEXT round number that will be generated (unless finished)
                next_round_num = display_current_round + 1
                progress_text = f"Simulating Round {next_round_num}..." if display_current_round < max_r else "Simulation complete."
                st.progress(progress_val, text=progress_text)
            
            with col_p2:
                # Check if the number of rounds completed is equal to the max rounds
                is_finished = display_current_round >= max_r
                
                if is_finished:
                    st.success("Simulation complete! View results on the Dashboard.")
                    
                    if st.button("View Final Results", use_container_width=True, key="view_final_arena", type="primary"): 
                        st.session_state.page = "Dashboard"
                        # --- WINNER CALCULATION AND POPUP TRIGGER ---
                        jd_final = read_judge() # Read final judge data
                        if not jd_final.empty:
                            total_coached_score = jd_final["total_coached"].astype(float).sum()
                            total_opponent_score = jd_final["total_opponent"].astype(float).sum()
                            
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
                    
                # The NEXT ROUND button should only appear if the simulation is NOT finished
                elif st.button("NEXT ROUND", use_container_width=True, key="next_round_btn_primary", type="primary"):
                    
                    with st.spinner(f"Drafting Round {display_current_round + 1} arguments..."):
                        
                        round_to_generate_internal = st.session_state.round # 1-indexed to 0-indexed conversion
                        
                        # Generate arguments for the next round
                        template_idx, template_text = rl.select()
                        df = read_debate()
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
                        
                        # Append the new round's data using the internal counter
                        append_round({
                            "round": round_to_generate_internal,
                            "speaker": "coached", 
                            "coached_argument": coached, 
                            "opponent_argument": opponent, 
                            "action": str(template_idx), 
                            "reward": reward
                        })
                        append_judge(round_to_generate_internal, judge_scores)
                        
                        try:
                            rl.update(template_idx, reward)
                        except Exception as e:
                            st.warning(f"RL update failed: {e}")

                        # IMPORTANT: Increment the COMPLETED round counter
                        st.session_state.round += 1
                        
                        st.toast(f"Round {display_current_round + 1} completed.", icon="✅")
                        time.sleep(0.5) 
                        st.session_state.view_round_idx = None
                        st.session_state.show_winner_popup = False # Ensure popup is off
                        st.session_state.winner_info = None # Clear winner info
                        st.rerun() 

        # --- Display Latest/Selected Round Arguments and Evaluation ---
        st.header("Round Viewer")
        
        df = read_debate()
        jd = read_judge()
        
        if df.empty or jd.empty:
            # This should only happen if state is active but Round 1 failed to generate
            st.warning("Round 1 generation failed. Check backend logs or try resetting simulation.")
        else:
            if st.session_state.view_round_idx is not None:
                sel_round_internal = int(st.session_state.view_round_idx)
                sel_round_display = sel_round_internal + 1

                row_matches = df[df["round"] == sel_round_internal]
                judge_matches = jd[jd["round"] == sel_round_internal] if "round" in jd.columns else jd.iloc[[min(len(jd)-1, sel_round_internal)]]

                if row_matches.empty:
                    st.warning(f"Round {sel_round_display} not found. Reverting to latest.")
                    st.session_state.view_round_idx = None
                    st.rerun()
                else:
                    latest = row_matches.iloc[-1]
                    judge = judge_matches.iloc[-1] if not judge_matches.empty else {"total_coached": "N/A", "total_opponent": "N/A", "notes_coached": "", "notes_opponent": ""}
                    
                    colpv, coln, colb = st.columns(3) 
                    with colpv:
                        if st.button("Previous", use_container_width=True, key="arena_prev_btn", type="primary"): 
                            prev_idx = max(df["round"].min(), sel_round_internal - 1)
                            st.session_state.view_round_idx = prev_idx
                            st.session_state.show_winner_popup = False # Close popup
                            st.rerun()
                    with coln:
                        if st.button("Next", use_container_width=True, key="arena_next_btn", type="primary"): 
                            next_idx = min(df["round"].max(), sel_round_internal + 1)
                            st.session_state.view_round_idx = next_idx
                            st.session_state.show_winner_popup = False # Close popup
                            st.rerun()
                    with colb:
                        if st.button("Back to Live", use_container_width=True, key="arena_back_to_live_btn", type="primary"):
                            st.session_state.view_round_idx = None
                            st.session_state.show_winner_popup = False # Close popup
                            st.rerun()
                    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

                    st.markdown(f"**Viewing Round {sel_round_display}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"<div class='debate-box coach-box'><h3>Coached Argument</h3>{latest['coached_argument']}</div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<div class='debate-box opponent-box'><h3>Opponent</h3>{latest['opponent_argument']}</div>", unsafe_allow_html=True)

                    with st.expander("Judge Evaluation (Selected Round)", expanded=True):
                        st.markdown(f"""
                            <div class='debate-box judge-box'>
                                <h4>Scores</h4>
                                <p>Coach: <b>{judge.get('total_coached', 'N/A')}/10</b> | Opponent: <b>{judge.get('total_opponent', 'N/A')}/10</b></p>
                                <h4>Notes</h4>
                                <p><b>Coach:</b> {judge.get('notes_coached', '')}</p>
                                <p><b>Opponent:</b> {judge.get('notes_opponent', '')}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    one_round_csv = pd.DataFrame([latest]).to_csv(index=False)
                    st.download_button("Download this round (CSV)", one_round_csv, file_name=f"round_{sel_round_display}.csv", mime="text/csv", use_container_width=True, type="primary")

            else:
                # show latest round (default "live" mode)
                latest = df.iloc[-1]
                judge = jd.iloc[-1]
                st.markdown(f"**Viewing Latest Round {display_current_round}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"<div class='debate-box coach-box'><h3>Coached Argument</h3>{latest['coached_argument']}</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='debate-box opponent-box'><h3>Opponent</h3>{latest['opponent_argument']}</div>", unsafe_allow_html=True)

                with st.expander("Judge Evaluation (Latest)", expanded=True):
                    st.markdown(f"""
                        <div class='debate-box judge-box'>
                            <h4>Scores</h4>
                            <p>Coach: <b>{judge['total_coached']}/10</b> | Opponent: <b>{judge['total_opponent']}/10</b></p>
                            <h4>Notes</h4>
                            <p><b>Coach:</b> {judge['notes_coached']}</p>
                            <p><b>Opponent:</b> {judge['notes_opponent']}</p>
                        </div>
                    """, unsafe_allow_html=True)

                with st.expander("Quick history (latest {display_current_round} rounds)", expanded=False):
                    recent = df.tail(5).reset_index(drop=True)
                    for _, r in recent.iterrows():
                        # Display 1-indexed round in quick history
                        st.markdown(f"**Round {int(r['round'] + 1)}** --- <span class='small-muted'>{str(r.get('coached_argument',''))[:120].replace(chr(10),' ') + ('...' if len(str(r.get('coached_argument','')))>120 else '')}</span>", unsafe_allow_html=True)
                    st.download_button("Download full history (CSV)", df.to_csv(index=False), file_name="debate_history.csv", mime="text/csv", use_container_width=True, key="arena_download_full_hist", type="primary")


## Dashboard Page
elif st.session_state.page == "Dashboard":
    st.markdown("<h2>PROGRESS DASHBOARD: AI Learning Metrics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a6b1bf;'>Monitor the Reinforcement Learning Agent's performance and strategy effectiveness.</p>", unsafe_allow_html=True)
    st.divider()
    
    jd = read_judge()
    df = read_debate()

    if jd.empty or df.empty:
        st.warning("No debate data yet. Run a few rounds in the **Arena View** first!")
    else:
        avg_coached = jd["total_coached"].astype(float).mean()
        avg_opponent = jd["total_opponent"].astype(float).mean()
        avg_reward = df["reward"].astype(float).mean()
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
            # When plotting, use the 1-indexed round for clarity
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
                
                with st.container(border=True):
                    col_h1, col_h2, col_h3 = st.columns([1, 1, 1])
                    with col_h1:
                        st.markdown(f"**Round:** **{int(row['round'] + 1)}**") # Display 1-indexed
                    with col_h2:
                        st.markdown(f"**Coach Score:** <span style='color:#38bdf8;'>**{row['total_coached']:.2f}**</span>", unsafe_allow_html=True)
                    with col_h3:
                        st.markdown(f"**Opponent Score:** <span style='color:#ef4444;'>**{row['total_opponent']:.2f}**</span>", unsafe_allow_html=True)
                        
                    st.markdown("<hr style='border-top: 1px solid #30363d;'>", unsafe_allow_html=True)
                    
                    st.markdown(f"**Judge's Feedback (Coach):** {row['notes_coached']}")
                    st.markdown(f"**Judge's Feedback (Opponent):** {row['notes_opponent']}")
                    
        st.divider()

        st.subheader("Raw Simulation Data")
        
        with st.expander("View Full Round Details Table"):
            # When displaying raw dataframes, the internal 0-indexed round is fine
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