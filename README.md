# 🧠 DebateMind — AI-Powered Debate Simulation with Reinforcement Learning

> A multi-agent debate arena where an RL-coached debater learns to argue better round by round, judged in real-time by an AI.

---

## What It Does

**DebateMind** simulates a structured debate between two AI agents — a **Coached Debater** and an **Opponent** — with a neutral **Judge** scoring each round. The twist: the coached debater is guided by a **Reinforcement Learning (RL) agent** that learns which argument strategies (logical, emotional, data-driven, etc.) work best for a given topic over multiple rounds.

Key capabilities:

- Enter any debate topic (e.g. *"Is remote work better than office work?"*) and watch two LLMs go head-to-head
- The RL agent selects a prompt strategy each round, receives reward based on judge scores, and updates its policy
- A Judge LLM evaluates both sides on **logic, relevance, clarity, and persuasiveness**, producing scores and written notes
- Real-time animated chat UI streams arguments as they're generated
- Upload a **PDF** to inject custom context into the debate (e.g. research papers, articles)
- A **Dashboard** visualises RL learning curves, score trends, and strategy performance across rounds
- Full debate history is persisted to CSV and can be re-loaded, browsed, or exported (CSV/JSON)

---

## Tech Stack

| Layer | Technology |
|---|---|
| **UI / Frontend** | [Streamlit](https://streamlit.io) |
| **LLM Backend** | OpenRouter API (any compatible model, e.g. GPT-4, Claude, Mistral) |
| **RL Agent** | Custom epsilon-greedy bandit (`rl_agent.py`) |
| **Memory / Storage** | CSV files via `pandas` (`debate_memory.csv`, `judge_summary.csv`) |
| **PDF Parsing** | PyPDF2 |
| **HTTP Client** | `httpx` |
| **Data Visualisation** | Altair + Streamlit metrics |
| **Language** | Python 3.10+ |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Debate Arena│  │  Dashboard   │  │ PDF Uploader │  │
│  └──────┬───────┘  └──────────────┘  └──────────────┘  │
└─────────┼───────────────────────────────────────────────┘
          │  per-round loop
          ▼
┌─────────────────────────────────────────────────────────┐
│                    Backend (backend/)                    │
│                                                         │
│  ┌────────────────────────────────────────────────┐     │
│  │  1. RL Agent (rl_agent.py)                     │     │
│  │     - Reads last reward from CSV               │     │
│  │     - Selects prompt strategy (ε-greedy)       │     │
│  │     - Updates policy weights after each round  │     │
│  └───────────────────┬────────────────────────────┘     │
│                      │ strategy / prompt template        │
│                      ▼                                   │
│  ┌────────────────────────────────────────────────┐     │
│  │  2. Coached Debater (debater.py)               │     │
│  │     - Builds prompt: topic + strategy +        │     │
│  │       previous rounds (from CSV)               │     │
│  │     - Calls LLM API → coached_argument         │     │
│  └───────────────────┬────────────────────────────┘     │
│                      │ coached_argument                  │
│                      ▼                                   │
│  ┌────────────────────────────────────────────────┐     │
│  │  3. Opponent LLM (opponent.py)                 │     │
│  │     - Receives topic + coached_argument        │     │
│  │     - Returns counter-argument (static policy) │     │
│  └───────────────────┬────────────────────────────┘     │
│                      │ both arguments                    │
│                      ▼                                   │
│  ┌────────────────────────────────────────────────┐     │
│  │  4. Judge LLM (judge.py)                       │     │
│  │     - Evaluates both sides (temp=0, JSON out)  │     │
│  │     - Scores: logic, relevance, clarity,       │     │
│  │       persuasiveness (0–10 each)               │     │
│  │     - Returns total_coached, total_opponent,   │     │
│  │       notes_coached, notes_opponent            │     │
│  └───────────────────┬────────────────────────────┘     │
│                      │ scores → reward signal            │
│                      ▼                                   │
│  ┌────────────────────────────────────────────────┐     │
│  │  5. Memory Manager (memory_manager.py)         │     │
│  │     - Appends round to debate_memory.csv       │     │
│  │     - Appends scores to judge_summary.csv      │     │
│  └────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────┐
│  data/                  │
│  ├── debate_memory.csv  │  ← round, arguments, reward
│  ├── judge_summary.csv  │  ← round scores + notes
│  └── rl_memory.json     │  ← RL policy state
└─────────────────────────┘
```

### Round Flow (per round)

```
RL Agent
  └─► selects strategy
        └─► Coached Debater generates argument
              └─► Opponent generates rebuttal
                    └─► Judge scores both
                          └─► reward = coached_score − opponent_score
                                └─► RL Agent updates policy
                                      └─► all data saved to CSV
```

---

## File Structure

```
debate-coach/
│
├── app.py                    # Streamlit UI — arena, dashboard, PDF upload
├── backend/
│   ├── rl_agent.py           # Epsilon-greedy RL agent & strategy selection
│   ├── debater.py            # Coached LLM interface
│   ├── opponent.py           # Opponent LLM interface
│   ├── judge.py              # Judge evaluation (JSON output, temp=0)
│   ├── memory_manager.py     # CSV read/write via pandas
│   ├── config.py             # API keys, model names, MAX_ROUNDS
│   └── utils.py              # Prompt builders, PDF loader, sanitizers
│
├── api/                      # (API route helpers)
│
├── data/
│   ├── debate_memory.csv
│   ├── judge_summary.csv
│   └── rl_memory.json
│
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone the repo

```bash
git clone https://github.com/haragam22/debatemind.git
cd debatemind
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.10+. It's recommended to use a virtual environment:
> ```bash
> python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
> ```

### 3. Set up your API key

Create a `.env` file in the project root (or set environment variables directly):

```env
OPENROUTER_API_KEY=your_key_here
```

> DebateMind uses [OpenRouter](https://openrouter.ai) to access LLMs. You can swap in any compatible model (GPT-4, Claude, Mistral, etc.) by editing `backend/config.py`.

### 4. Launch the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### 5. Start a debate

1. Enter a debate topic in the **sidebar** (e.g. *"AI will replace software engineers"*)
2. Set the number of rounds (default: 5)
3. Click **Start / Reset Simulation**
4. Click **NEXT ROUND** to advance — each round generates arguments, a rebuttal, and judge scores
5. After all rounds, click **View Final Results** to see the dashboard and winner

### Optional: Add PDF context

Upload a PDF (research paper, article, etc.) using the uploader on the main page. The extracted text will be injected into the judge and debater prompts for that session.

---

## Research Backing

This project draws on:

- **Du et al. (2023)** — *Improving Factuality and Reasoning through Multiagent Debate*
- **Liang et al. (2024, EMNLP)** — *Encouraging Divergent Thinking via Multi-Agent Debate (MAD)*
- **Kenton et al. (2024)** — *Scalable Oversight with Weak LLMs Judging Strong LLMs*
- **Wang et al. (2025)** — *RL for Reasoning in LLMs with One Training Example*
- **Zhang et al. (2025)** — *A Survey of RL for Large Reasoning Models*

---

## License

MIT © [haragam22](https://github.com/haragam22)
