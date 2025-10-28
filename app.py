# app.py
import streamlit as st
from backend.memory_manager import init_files, read_debate, read_judge, append_round, append_judge
from backend.rl_agent import RLAgent
from backend.debater import generate_coached_argument
from backend.opponent import generate_opponent_argument
from backend.judge import evaluate
from backend.utils import sanitize_topic
import time
import pandas as pd
from backend.config import MAX_ROUNDS

# initialize files
init_files()

st.set_page_config(page_title="LLM Debate Coach", layout="wide")
st.title("LLM Debate Coach — Demo")

# sidebar controls
with st.sidebar:
    st.header("Controls")
    topic = st.text_input("Debate topic", "AI systems should have legal rights")
    max_rounds = st.number_input("Rounds", min_value=1, max_value=MAX_ROUNDS, value=MAX_ROUNDS)
    start = st.button("Create / Start Debate")

# session state
if "debate_active" not in st.session_state:
    st.session_state.debate_active = False
    st.session_state.round = 0
    st.session_state.topic = ""
    st.session_state.history = []

if start:
    topic = sanitize_topic(topic)
    st.session_state.debate_active = True
    st.session_state.round = 0
    st.session_state.topic = topic
    st.session_state.history = []
    st.success(f"Debate created on: {topic}")

rl = RLAgent()

col1, col2 = st.columns([2,1])

with col1:
    if st.session_state.debate_active:
        st.subheader(f"Debate Topic: {st.session_state.topic}")
        if st.button("Run Next Round"):
            # run one round: RL -> coached arg -> opponent -> judge -> persist -> update policy
            st.session_state.round += 1
            round_no = st.session_state.round

            # select template via RL
            template_idx, template_text = rl.select()

            # get previous arguments for context
            df = read_debate()
            prev_args = df["coached_argument"].dropna().astype(str).tolist() if not df.empty else []

            # generate coached argument
            try:
                coached = generate_coached_argument(template_text, st.session_state.topic, previous=prev_args)
            except Exception as e:
                st.error(f"Coached LLM error: {e}")
                coached = "Error generating coached argument."

            # opponent rebuttal
            try:
                opponent = generate_opponent_argument(coached, st.session_state.topic)
            except Exception as e:
                st.error(f"Opponent LLM error: {e}")
                opponent = "Error generating opponent argument."

            # judge evaluation
            try:
                judge_scores = evaluate(coached, opponent, st.session_state.topic)
            except Exception as e:
                st.error(f"Judge error: {e}")
                judge_scores = {
                    "total_coached":5.0, "total_opponent":5.0,
                    "notes_coached":"judge fallback", "notes_opponent":"judge fallback"
                }

            # reward = difference
            reward = float(judge_scores.get("total_coached",0)) - float(judge_scores.get("total_opponent",0))

            # append memory
            append_round({
                "round": round_no,
                "speaker": "rl_debater",
                "coached_argument": coached,
                "opponent_argument": opponent,
                "action": str(template_idx),
                "reward": reward
            })
            append_judge(round_no, judge_scores)

            # RL update
            try:
                rl.update(template_idx, reward)
            except Exception as e:
                st.error(f"RL update error: {e}")

            st.rerun()

        # show transcripts
        df = read_debate()
        if not df.empty:
            for _, r in df.iterrows():
                st.markdown(f"### Round {int(r['round'])}")
                st.markdown(f"**Coached (RL)**: {r['coached_argument']}")
                st.markdown(f"**Opponent**: {r['opponent_argument']}")
                st.markdown(f"**Action (template)**: {r['action']}")
                st.markdown(f"**Reward**: {r['reward']}")
                st.markdown("---")

with col2:
    st.subheader("Metrics")
    jd = read_judge()
    if not jd.empty:
        jd_display = jd.copy()
        st.table(jd_display)
    else:
        st.info("No rounds yet. Start the debate to see judge scores.")
