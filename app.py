import streamlit as st
import pandas as pd
import numpy as np
import time
from backend.memory_manager import init_files, read_debate, read_judge, append_round, append_judge
from backend.rl_agent import RLAgent
from backend.debater import generate_coached_argument
from backend.opponent import generate_opponent_argument
from backend.judge import evaluate
from backend.utils import sanitize_topic
from backend.config import MAX_ROUNDS

# Initialize files
init_files()

# Streamlit setup
st.set_page_config(page_title="LLM Debate Coach", layout="wide")
st.sidebar.title("Navigation")

# -------------------- GLOBAL DARK THEME STYLING -------------------- #
st.markdown("""
    <style>
        body, .stApp {
            background-color: #0b1220;
            color: #e0e0e0;
        }
        h1, h2, h3, h4, h5, h6, p, label {
            color: #e6e6e6 !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #11182b;
            color: #f0f0f0;
        }
        div.stButton > button {
            background-color: #2563eb;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.6em 1.2em;
        }
        div.stButton > button:hover {
            background-color: #1d4ed8;
        }
        [data-testid="stMetric"] {
            background: #1e293b;
            border-radius: 12px;
            padding: 1em;
            color: #f9fafb !important;
        }
        table {
            background-color: #1e293b !important;
            color: #e5e7eb !important;
            border-radius: 10px;
        }
        input, textarea {
            background-color: #1f2937 !important;
            color: #e5e7eb !important;
            border-radius: 8px;
        }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-thumb {
            background: #2563eb;
            border-radius: 10px;
        }
        .debate-box {
            border-radius: 14px;
            padding: 1.2em;
            margin-top: 1em;
            font-size: 1.05em;
        }
        .coach-box {
            background: rgba(37, 99, 235, 0.15);
            border: 1px solid #2563eb;
        }
        .opponent-box {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid #ef4444;
        }
        .judge-box {
            background: rgba(234, 179, 8, 0.1);
            border: 1px solid #eab308;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.radio("Go to", ["Debate Simulator", "Dashboard"])

# -------------------- DEBATE SIMULATOR PAGE -------------------- #
if page == "Debate Simulator":
    st.markdown("<h1 style='text-align:center;'>🏛️ Debate Arena</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Debate Controls")
        topic = st.text_input("Debate topic", "Is remote work better than office work?")
        max_rounds = st.number_input("Rounds", min_value=1, max_value=MAX_ROUNDS, value=MAX_ROUNDS)
        start = st.button("🎯 Create / Start Debate")

    # Session state
    if "debate_active" not in st.session_state:
        st.session_state.debate_active = False
        st.session_state.round = 0
        st.session_state.topic = ""
        st.session_state.history = []

    if start:
        topic = sanitize_topic(topic)
        st.session_state.debate_active = True
        st.session_state.round = -1
        st.session_state.topic = topic
        st.session_state.history = []
        st.success(f"Debate created on: {topic}")

    rl = RLAgent()

    if st.session_state.debate_active:
        st.markdown(f"<h3 style='color:#94a3b8;'>Topic</h3><h2>{st.session_state.topic}</h2>", unsafe_allow_html=True)
        try:
                # ensure max_rounds is valid
                max_r = float(max_rounds) if max_rounds else 1.0
        except Exception:
                max_r = 1.0

        try:
                raw_progress = float(st.session_state.round) / max_r
        except Exception:
                raw_progress = 0.0

            # clamp between 0 and 1
        if not (0.0 <= raw_progress <= 1.0):
                raw_progress = min(max(raw_progress, 0.0), 1.0)

        progress = float(raw_progress)  # ensure it's a float, not tuple or numpy type

            # --- Display Progress ---
        st.markdown(
                f"### Round {st.session_state.round + 1} of {int(max_r)} "
                "<span style='color:#60a5fa;'>(Arguments in Progress)</span>",
                unsafe_allow_html=True,
        )
        st.progress(progress)

        if st.button("⏭️ Next Round", use_container_width=True):
            st.session_state.round += 1
            round_no = st.session_state.round

            # RL agent selection
            template_idx, template_text = rl.select()

            # Previous context
            df = read_debate()
            prev_args = df["coached_argument"].dropna().astype(str).tolist() if not df.empty else []

            # Generate coached + opponent
            try:
                coached = generate_coached_argument(template_text, st.session_state.topic, previous=prev_args)
            except Exception as e:
                st.error(f"Coached LLM error: {e}")
                coached = "Error generating coached argument."

            try:
                opponent = generate_opponent_argument(coached, st.session_state.topic)
            except Exception as e:
                st.error(f"Opponent LLM error: {e}")
                opponent = "Error generating opponent argument."

            # Judge evaluation
            try:
                judge_scores = evaluate(coached, opponent, st.session_state.topic)
            except Exception as e:
                st.error(f"Judge error: {e}")
                judge_scores = {
                    "total_coached": 5.0,
                    "total_opponent": 5.0,
                    "notes_coached": "judge fallback",
                    "notes_opponent": "judge fallback"
                }

            reward = float(judge_scores.get("total_coached", 0)) - float(judge_scores.get("total_opponent", 0))

            append_round({
                "round": round_no,
                "speaker": "rl_debater",
                "coached_argument": coached,
                "opponent_argument": opponent,
                "action": str(template_idx),
                "reward": reward
            })
            append_judge(round_no, judge_scores)

            try:
                rl.update(template_idx, reward)
            except Exception as e:
                st.error(f"RL update error: {e}")

            st.rerun()

        # Display last round
        df = read_debate()
        jd = read_judge()

        if not df.empty and not jd.empty:
            latest_round = df.iloc[-1]
            latest_judge = jd.iloc[-1]

            st.markdown(f'<div class="debate-box coach-box"><b>🧠 Coached Debater:</b><br>{latest_round["coached_argument"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="debate-box opponent-box"><b>⚔️ Opponent:</b><br>{latest_round["opponent_argument"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="debate-box judge-box"><b>⚖️ Judge Evaluation:</b><br>Debater: {latest_judge["total_coached"]} | Opponent: {latest_judge["total_opponent"]}<br><i>{latest_judge["notes_coached"]}</i></div>', unsafe_allow_html=True)

# -------------------- DASHBOARD PAGE -------------------- #
elif page == "Dashboard":
    st.markdown("<h1 style='text-align:center;'>📊 AI Learning Progress Dashboard</h1>", unsafe_allow_html=True)

    jd = read_judge()
    df = read_debate()

    if jd.empty or df.empty:
        st.warning("No debate data yet. Run a few rounds first!")
    else:
        st.subheader("Score Trend: Debater vs Opponent")

        avg_coached = jd["total_coached"].astype(float).mean()
        avg_opponent = jd["total_opponent"].astype(float).mean()
        avg_reward = df["reward"].astype(float).mean()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg. Debater Score", f"{avg_coached:.2f}", f"{avg_coached - avg_opponent:+.2f} vs Opponent")

            # FIXED chart (dark-compatible)
            chart_data = jd[["round", "total_coached", "total_opponent"]].melt("round", var_name="Type", value_name="Score")
            st.area_chart(chart_data, x="round", y="Score", color="Type", height=300)

        with col2:
            st.metric("Avg. Reward (Strategy Success)", f"{avg_reward:+.2f}", f"Across {len(jd)} rounds")
            strat_df = df.groupby("action")["reward"].mean().reset_index()
            strat_df.columns = ["Strategy", "Avg Reward"]
            st.bar_chart(strat_df.set_index("Strategy"), height=300)

        st.subheader("🧾 Judge Summary")
        for _, row in jd.iterrows():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"### Round {int(row['round'])}")
            with col2:
                st.markdown(f"**Verdict:** {row['notes_coached']}")
            st.divider()

        st.subheader("📘 Round Details")
        st.dataframe(df)
