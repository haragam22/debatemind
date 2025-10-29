import streamlit as st
import pandas as pd
import numpy as np
import time
# --- Preserve Backend Imports ---
from backend.memory_manager import init_files, read_debate, read_judge, append_round, append_judge
from backend.rl_agent import RLAgent
from backend.debater import generate_coached_argument
from backend.opponent import generate_opponent_argument
from backend.judge import evaluate
from backend.utils import sanitize_topic
from backend.config import MAX_ROUNDS 

# --- 1. Initialization and Setup ---
# Line 1
init_files() 
# Line 2
rl = RLAgent() 

# Streamlit page configuration - Use 'wide' layout for a dashboard feel
# Line 3
st.set_page_config(
    page_title="DebateMind: AI Coach", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. Custom Styling and Theme (Deep Dark Mode & Enhanced Aesthetics) ---
# Line 1
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
            margin: 8px 0; /* Further reduced vertical margin for dividers */
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
        
        /* Navigation Buttons - Custom styling for card-like appearance */
        div.stButton button {
            background-color: #21262d;
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

        /* Primary Action/Active Buttons - TARGETS DISABLED (ACTIVE) NAV BUTTONS */
        div.stButton > button[kind="primary"] {
            background-color: #3b82f6; 
            color: white;
            border: none;
            padding: 1rem 1.5rem; 
            font-weight: 700;
            font-size: 1.2em;
            height: auto; 
            margin-top: 10px; 
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
    </style>
"""
# Line 2
st.markdown(STYLING_CSS, unsafe_allow_html=True)

# --- 3. Header Bar with Title (DebateMind) ---
def render_header():
    """Renders a sleek, bold header bar with the new project name."""
    # Line 1: Gradient Title with Glow
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; padding: 10px 0 0px 0;">
            <h1 style="
                margin: 0; 
                font-size: 3em; 
                font-weight: 800; 
                letter-spacing: 2px;
                /* Gradient Effect */
                background: linear-gradient(90deg, #38bdf8, #818cf8); 
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0px 0px 15px rgba(56, 189, 248, 0.5); /* Enhanced Glow */
            ">
            DebateMind 
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Line 2: Light Beam Divider
    st.markdown("""
        <div style="
            height: 3px; 
            background: radial-gradient(circle at center, #38bdf8 0%, rgba(22, 27, 34, 0) 70%); 
            width: 100%; 
            margin: 5px 0 20px 0;
        "></div>
    """, unsafe_allow_html=True)

render_header()

# --- 4. Session State Management ---
# Line 1
if "page" not in st.session_state:
    # Line 2
    st.session_state.page = "Debate Arena"
# Line 3
if "debate_active" not in st.session_state:
    # Line 4
    st.session_state.debate_active = False
    # Line 5
    st.session_state.round = -1
    # Line 6
    st.session_state.topic = ""
    # Line 7
    st.session_state.history = []

# --- 5. Sidebar Navigation and Setup (Condensed Layout, Verbose Spacing) ---
# Line 1
with st.sidebar:
    # Line 2
    st.markdown("<h2 style='color:#f0f4f8; margin-bottom: 0px;'>Control Panel</h2>", unsafe_allow_html=True)
    # Line 3
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True) # Subtle divider

    # --- Navigation Buttons (Vertical Stacked Rectangles) ---
    # Line 4
    st.markdown("<h3>Navigation</h3>", unsafe_allow_html=True)
    
    # ARENA VIEW Button
    # Line 5
    if st.session_state.page == "Debate Arena":
        # Line 6 (Active Button)
        st.button("ARENA VIEW", use_container_width=True, key="nav_arena_btn_active", disabled=True, type="primary")
    # Line 7
    else:
        # Line 8 (Inactive Button)
        if st.button("ARENA VIEW", use_container_width=True, key="nav_arena_btn_inactive"):
            # Line 9
            st.session_state.page = "Debate Arena"

    # Line 10 (No explicit spacer here for tight vertical stacking)

    # DASHBOARD Button 
    # Line 11
    if st.session_state.page == "Dashboard":
        # Line 12 (Active Button)
        st.button("DASHBOARD", use_container_width=True, key="nav_dashboard_btn_active", disabled=True, type="primary")
    # Line 13
    else:
        # Line 14 (Inactive Button)
        if st.button("DASHBOARD", use_container_width=True, key="nav_dashboard_btn_inactive"):
            # Line 15
            st.session_state.page = "Dashboard"

    # Line 16
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True) # Subtle divider

    # --- Simulation Setup ---
    # Line 17
    st.markdown("<h3>Simulation Setup</h3>", unsafe_allow_html=True)
    
    # Topic Input 
    # Line 18
    st.markdown("**Topic for Debate**")
    # Line 19
    topic_input = st.text_input(
        label="", 
        value="Is remote work better than office work?",
        placeholder="Enter your debate topic here...",
        key="topic_input_key"
    )
    
    # Line 20 (No explicit spacer)

    # Rounds Input
    # Line 21
    max_rounds_input = st.number_input(
        "**Total Rounds to Simulate**", 
        min_value=1, 
        max_value=MAX_ROUNDS, 
        value=MAX_ROUNDS,
        key="max_rounds_input_key"
    )
    
    # Line 22 (No explicit spacer)
    
    # Start Button
    # Line 23
    if st.button("START / RESET SIMULATION", use_container_width=True, key="start_debate_btn", type="primary"):
        # Line 24
        if topic_input:
            # Line 25
            topic = sanitize_topic(topic_input)
            # Line 26
            st.session_state.debate_active = True
            # Line 27
            st.session_state.round = -1
            # Line 28
            st.session_state.topic = topic
            # Line 29
            st.session_state.history = []
            # Line 30
            st.toast(f"Simulation started on: {topic}", icon="✅")
            # Line 31
            st.rerun()
        # Line 32
        else:
            # Line 33
            st.error("Please enter a debate topic to start.")

# --- 6. Main Content Pages ---

# ----------------------------------------------------------------------
## Debate Arena Page
# ----------------------------------------------------------------------
# Line 1
if st.session_state.page == "Debate Arena":
    # Line 2
    st.markdown("<h2>DEBATE ARENA: Real-Time Simulation</h2>", unsafe_allow_html=True)
    # Line 3
    st.markdown("<p style='color:#a6b1bf;'>Monitor the arguments and the judge's real-time evaluations.</p>", unsafe_allow_html=True)
    # Line 4
    st.divider()

    # Line 5
    if not st.session_state.debate_active:
        # Line 6
        st.info("Start a new debate from the sidebar to begin the simulation.")
    # Line 7
    else:
        # Line 8
        st.subheader(f"Topic: **{st.session_state.topic}**")

        # --- Round Progress and Next Round Button ---
        # Line 9
        max_r = float(max_rounds_input)
        # Line 10
        current_round = st.session_state.round + 1
        
        # Line 11
        progress_val = (current_round / max_r) if max_r > 0 else 0.0
        # Line 12
        progress_val = min(max(progress_val, 0.0), 1.0)
        
        # Line 13
        progress_container = st.container(border=True)
        # Line 14
        with progress_container:
            # Line 15
            col_p1, col_p2 = st.columns([3, 1])
            # Line 16
            with col_p1:
                # Line 17
                st.metric(
                    label="Current Round Progress", 
                    value=f"Round {current_round} / {int(max_r)}",
                    delta=f"{progress_val*100:.0f}% Complete"
                )
                # Line 18
                st.progress(progress_val, text=f"Simulating Round {current_round}...")
            
            # Line 19
            with col_p2:
                # Line 20
                is_finished = (st.session_state.round + 1) >= max_r
                
                # Line 21
                if is_finished:
                    # Line 22
                    st.success("Simulation complete! View results on the Dashboard.")
                    
                    # --- CRITICAL CHANGE: Make 'View Final Results' button switch page ---
                    # Line 23
                    if st.button("View Final Results", use_container_width=True, key="view_final_arena"):
                        # Line 24
                        st.session_state.page = "Dashboard"
                        # Line 25
                        st.rerun()
                    # --- END CRITICAL CHANGE ---
                    
                # Line 26
                elif st.button("NEXT ROUND", use_container_width=True, key="next_round_btn_primary", type="primary"):
                    
                    # Line 27
                    with st.spinner(f"Debater and Opponent are drafting arguments for Round {current_round}..."):
                        # Line 28
                        st.session_state.round += 1
                        # Line 29
                        round_no = st.session_state.round

                        # Line 30
                        template_idx, template_text = rl.select()
                        # Line 31
                        df = read_debate()
                        # Line 32
                        prev_args = df["coached_argument"].dropna().astype(str).tolist() if not df.empty else []

                        # Line 33
                        try:
                            # Line 34
                            coached = generate_coached_argument(template_text, st.session_state.topic, previous=prev_args)
                        # Line 35
                        except Exception as e:
                            # Line 36
                            st.error(f"Coached LLM error: {e}")
                            # Line 37
                            coached = "Error generating coached argument."

                        # Line 38
                        try:
                            # Line 39
                            opponent = generate_opponent_argument(coached, st.session_state.topic)
                        # Line 40
                        except Exception as e:
                            # Line 41
                            st.error(f"Opponent LLM error: {e}")
                            # Line 42
                            opponent = "Error generating opponent argument."

                    # Line 43
                    with st.spinner("Judge is evaluating the round..."):
                        # Line 44
                        try:
                            # Line 45
                            judge_scores = evaluate(coached, opponent, st.session_state.topic)
                        # Line 46
                        except Exception as e:
                            # Line 47
                            st.error(f"Judge error: {e}")
                            # Line 48
                            judge_scores = { "total_coached": 5.0, "total_opponent": 5.0, "notes_coached": "Judge fallback: Error during evaluation.", "notes_opponent": "Judge fallback: Error during evaluation." }

                        # Line 49
                        reward = float(judge_scores.get("total_coached", 0)) - float(judge_scores.get("total_opponent", 0))
                        
                        # Line 50
                        append_round({"round": round_no, "speaker": "rl_debater", "coached_argument": coached, "opponent_argument": opponent, "action": str(template_idx), "reward": reward})
                        # Line 51
                        append_judge(round_no, judge_scores)
                        
                        # Line 52
                        try:
                            # Line 53
                            rl.update(template_idx, reward)
                        # Line 54
                        except Exception as e:
                            # Line 55
                            st.error(f"RL update error: {e}")
                        
                        # Line 56
                        st.toast("Round successfully completed!", icon="🎉")
                        # Line 57
                        time.sleep(0.5) 
                        # Line 58
                        st.rerun() 

        # --- Display Last Round's Arguments and Evaluation ---
        # Line 59
        st.header("Latest Round Summary")
        
        # Line 60
        df = read_debate()
        # Line 61
        jd = read_judge()
        
        # Line 62
        if not df.empty and not jd.empty:
            # Line 63
            latest_round = df.iloc[-1]
            # Line 64
            latest_judge = jd.iloc[-1]
            
            # Line 65
            col_coach, col_opponent = st.columns(2)
            
            # Line 66
            with col_coach:
                # Line 67
                st.markdown(
                    f'<div class="debate-box coach-box"><h3>Coach (Debater) Argument</h3>'
                    f'<p>{latest_round["coached_argument"]}</p>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
            
            # Line 68
            with col_opponent:
                # Line 69
                st.markdown(
                    f'<div class="debate-box opponent-box"><h3>Opponent Argument</h3>'
                    f'<p>{latest_round["opponent_argument"]}</p>'
                    f'</div>', 
                    unsafe_allow_html=True
                )

            # Line 70
            with st.expander("Judge's Evaluation & Score Breakdown", expanded=True):
                # Line 71
                st.markdown(
                    f'<div class="debate-box judge-box">'
                    f'<h4>Final Scores:</h4>'
                    f'<ul>'
                    f'<li>**Coach (Debater):** <span style="font-weight:bold; color:#38bdf8;">{latest_judge["total_coached"]}/10</span></li>'
                    f'<li>**Opponent:** <span style="font-weight:bold; color:#ef4444;">{latest_judge["total_opponent"]}/10</span></li>'
                    f'</ul>'
                    f'<h4>Judge Notes:</h4>'
                    f'**For Coach/Debater:** {latest_judge["notes_coached"]}<br><br>'
                    f'**For Opponent:** {latest_judge["notes_opponent"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
        # Line 72
        else:
            # Line 73
            st.info("No debate arguments to display yet. Click **NEXT ROUND** to generate the first arguments.")


# ----------------------------------------------------------------------
## Dashboard Page
# ----------------------------------------------------------------------
# Line 1
elif st.session_state.page == "Dashboard":
    # Line 2
    st.markdown("<h2>PROGRESS DASHBOARD: AI Learning Metrics</h2>", unsafe_allow_html=True)
    # Line 3
    st.markdown("<p style='color:#a6b1bf;'>Monitor the Reinforcement Learning Agent's performance and strategy effectiveness.</p>", unsafe_allow_html=True)
    # Line 4
    st.divider()
    
    # Line 5
    jd = read_judge()
    # Line 6
    df = read_debate()

    # Line 7
    if jd.empty or df.empty:
        # Line 8
        st.warning("No debate data yet. Run a few rounds in the **Arena View** first!")
    # Line 9
    else:
        # Calculate Key Metrics
        # Line 10
        avg_coached = jd["total_coached"].astype(float).mean()
        # Line 11
        avg_opponent = jd["total_opponent"].astype(float).mean()
        # Line 12
        avg_reward = df["reward"].astype(float).mean()
        # Line 13
        total_rounds = len(jd)
        
        # --- A. Key Metrics (Top Row) ---
        # Line 14
        st.subheader("Key Performance Indicators")
        # Line 15
        col_m1, col_m2, col_m3 = st.columns(3)
        
        # Line 16
        with col_m1:
            # Line 17
            st.metric(
                label="Avg. Debater Score (Coach)", 
                value=f"{avg_coached:.2f}", 
                delta=f"{avg_coached - avg_opponent:+.2f} vs Opponent",
                delta_color="normal" if avg_coached >= avg_opponent else "inverse"
            )
        
        # Line 18
        with col_m2:
            # Line 19
            st.metric(
                label="Avg. Strategy Success (Reward)", 
                value=f"{avg_reward:+.2f}",
                delta="Success Rate",
                delta_color="off"
            )

        # Line 20
        with col_m3:
            # Line 21
            st.metric(
                label="Total Simulated Rounds", 
                value=f"{total_rounds}",
                delta="History Size",
                delta_color="off"
            )
        
        # Line 22
        st.divider()

        # --- B. Chart Section ---
        # Line 23
        st.subheader("Performance Visualizations")
        # Line 24
        col_c1, col_c2 = st.columns(2)
        
        # Line 25
        with col_c1:
            # Line 26
            st.caption("Score Trend: Debater vs Opponent (By Round)")
            # Line 27
            chart_data = jd[["round", "total_coached", "total_opponent"]].melt(
                "round", var_name="Agent", value_name="Score"
            )
            # Line 28
            chart_data["Agent"] = chart_data["Agent"].map({
                "total_coached": "Coach Debater (Blue)", 
                "total_opponent": "Opponent (Red)"
            })
            # Line 29
            st.line_chart(
                chart_data.set_index("round"), 
                y="Score", 
                color="Agent", 
                height=350
            )

        # Line 30
        with col_c2:
            # Line 31
            st.caption("Average Reward by Strategy (Action)")
            # Line 32
            strat_df = df.groupby("action")["reward"].mean().reset_index()
            # Line 33
            strat_df.columns = ["Strategy Index", "Avg Reward"]
            # Line 34
            st.bar_chart(strat_df.set_index("Strategy Index"), height=350)
            
        # Line 35
        st.divider()

        # --- C. Detailed Judge Summaries ---
        # Line 36
        with st.expander("Detailed Judge Summaries (By Round)", expanded=False):
            
            # Line 37
            for index, row in jd.iloc[::-1].iterrows():
                
                # Line 38
                with st.container(border=True):
                    # Line 39
                    col_h1, col_h2, col_h3 = st.columns([1, 1, 1])
                    # Line 40
                    with col_h1:
                        # Line 41
                        st.markdown(f"**Round:** **{int(row['round'])}**")
                    # Line 42
                    with col_h2:
                        # Line 43
                        st.markdown(f"**Coach Score:** <span style='color:#38bdf8;'>**{row['total_coached']:.2f}**</span>", unsafe_allow_html=True)
                    # Line 44
                    with col_h3:
                        # Line 45
                        st.markdown(f"**Opponent Score:** <span style='color:#ef4444;'>**{row['total_opponent']:.2f}**</span>", unsafe_allow_html=True)
                        
                    # Line 46
                    st.markdown("<hr style='border-top: 1px solid #30363d;'>", unsafe_allow_html=True)
                    
                    # Line 47
                    st.markdown(f"**Judge's Feedback (Coach):** {row['notes_coached']}")
                    # Line 48
                    st.markdown(f"**Judge's Feedback (Opponent):** {row['notes_opponent']}")
                    
        
        # Line 49
        st.divider()

        # --- D. Raw Data View ---
        # Line 50
        st.subheader("Raw Simulation Data")
        
        # Line 51
        with st.expander("View Full Round Details Table"):
            # Line 52
            st.dataframe(df, use_container_width=True)