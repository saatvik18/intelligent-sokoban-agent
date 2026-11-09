# 🎮 Intelligent Sokoban Puzzle-Solving Agent
### Production Hybrid Game AI Architecture with CNN+LSTM Perception, Deadlock Reasoning, A* Hungarian Planning, LLM/RAG Knowledge Grounding, and DRL

---

## 📌 1. Project Overview & Problem Statement

Sokoban is a classic PSPACE-complete puzzle where an agent must push boxes onto designated storage targets in a constrained grid. Because boxes can only be pushed (never pulled), innocent-looking moves frequently lead to **irreversible deadlock states** (e.g., pushing a box into an un-targeted corner or trapping multiple boxes along a wall).

Standard standalone Deep Reinforcement Learning (DRL) algorithms fail catastrophically on Sokoban because unguided random exploration rapidly drives boxes into irreversible deadlock traps within the first few steps.

To overcome this, this project implements the complete **Strict Hybrid Game AI Architecture**:

```
                              ┌────────────────────────┐
                              │    GAME ENVIRONMENT    │
                              │ (Sokoban Grid / Levels)│
                              └───────────┬────────────┘
                                          │ State S_t (6-Channel Tensor & Grid Graph)
                                          ▼
                              ┌────────────────────────┐
                              │       PERCEPTION       │
                              │  • CNN (Spatial Board) │ ──> Spatial Embedding z_cnn (128-dim)
                              │  • LSTM (Action Seq)   │ ──> Sequence State (64-dim)
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │   SEARCH & PLANNING    │
                              │ • Deadlock Detection   │ ──> Active Action Mask A_mask
                              │ • A* (Hungarian Match) │ ──> Minimum Bipartite Distance + Macro Plan
                              │ • Backtracking Search  │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │     LLM / SLM + RAG    │
                              │ • Vector Knowledge Base│ ──> Semantic TF-IDF Document Retrieval
                              │ • Strategy Advisor     │ ──> 16-dim Continuous Strategic Context Vector
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ DEEP RL (DQN / PPO)    │
                              │ • Dueling Double DQN   │ ──> Fuses Spatial + Sequence + Strategic Vector
                              │ • Action-Masked Policy │ ──> Masks deadlocking & illegal moves
                              └───────────┬────────────┘
                                          │ Primitive Action A_t (UP / DOWN / LEFT / RIGHT)
                                          ▼
                              ┌────────────────────────┐
                              │    GAME ENVIRONMENT    │
                              │  (Reward & Next State) │
                              └────────────────────────┘
```

---

## 🧠 2. What AI is Used & How It Works

### Tier 1: Perception AI (CNN + LSTM)
- **Multi-Channel CNN State Encoder** (`sokoban/perception/cnn_encoder.py`):
  - Converts the board state into a $6 \times 8 \times 8$ feature tensor:
    - **Channel 0**: Walls
    - **Channel 1**: Unplaced Boxes
    - **Channel 2**: Boxes on Target (Goal achieved)
    - **Channel 3**: Empty Targets
    - **Channel 4**: Player Location
    - **Channel 5**: Player Reachable Floor Mask (Flood-fill connected component)
  - Uses residual convolutional blocks and global spatial attention pooling to project the grid into a dense **128-dimensional spatial embedding vector** ($z_{\text{cnn}} \in \mathbb{R}^{128}$).
- **LSTM Sequence Tracker** (`sokoban/perception/lstm_model.py`):
  - Processes temporal sequences of moves and spatial embeddings ($B \times T \times 132$).
  - Tracks movement momentum, predicts future transition dynamics, and classifies tactical behavior phases.

---

### Tier 2: Search, Planning & Deadlock Detection
- **Explicit Deadlock Detection Engine** (`sokoban/reasoning/deadlock.py`):
  - Prevents fatal pushes before they happen by detecting 4 primary deadlock categories:
    1. **Corner Deadlocks**: Box pushed into an orthogonal wall intersection without a target.
    2. **Wall / Line Deadlocks**: Multiple boxes lined up against a continuous wall with fewer targets than boxes.
    3. **2x2 Square Deadlocks**: Any $2 \times 2$ cluster of boxes and walls with no way to push outward.
    4. **Static Deadlock Map**: Precomputed via reverse-pull flood fill from all goal targets.
  - Generates an active **Deadlock Action Mask** $A_{\text{mask}} \in \{0, 1\}^4$ that filters out illegal or deadlocking actions.
- **A\* Heuristic Search with Hungarian Bipartite Matching** (`sokoban/reasoning/astar.py`):
  - Computes global lower-bound solution cost using the **Hungarian Algorithm** (`scipy.optimize.linear_sum_assignment`) over the Manhattan distance matrix between all boxes and targets.
  - Combines with transposition table caching (state hashing) and deadlock pruning to find optimal step paths in milliseconds.
- **Depth-First Backtracking Solver** (`sokoban/reasoning/backtracking.py`):
  - Depth-bounded search with cycle detection and state transposition table.

---

### Tier 3: LLM / SLM + RAG Strategic Reasoning
- **Sokoban Domain Knowledge Base** (`sokoban/rag_llm/knowledge_base.py`):
  - Structured repository of tactical principles: *Corridor packing order, Alcove filling priority, Player mobility preservation, 2x2 cluster avoidance, and Sacrificial maneuvers*.
- **Semantic Vector Retriever** (`sokoban/rag_llm/retriever.py`):
  - Uses TF-IDF cosine similarity vector space retrieval with query caching to retrieve relevant tactical documents matching current board conditions.
- **Strategic Reasoner & Context Vector** (`sokoban/rag_llm/strategy_advisor.py`):
  - Synthesizes retrieved domain principles, target completion ratio, and Hungarian distance into a continuous **16-dimensional Strategic Context Vector** ($\mathbf{v}_{\text{strat}} \in \mathbb{R}^{16}$) fed directly to the DRL policy network.
  - Generates natural language explanations grounding each move.

---

### Tier 4: Deep Reinforcement Learning (Dueling Double DQN)
- **Multi-Modal Feature Fusion Network** (`sokoban/drl/dqn_agent.py`):
  - The DRL policy network receives:
    $$\text{Input} = [z_{\text{cnn}} \ (128\text{-d}) \parallel \mathbf{v}_{\text{strat}} \ (16\text{-d})]$$
  - Dueling architecture divides into:
    - **Value Stream** $V(s) \in \mathbb{R}^1$: Estimates state solvability and value.
    - **Advantage Stream** $A(s, a) \in \mathbb{R}^4$: Estimates relative value of actions `UP`, `DOWN`, `LEFT`, `RIGHT`.
    - Combined via: $Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')\right)$
- **Deadlock Action Masking**:
  - The Q-values of actions leading to deadlocks are masked with $-\infty$:
    $$Q(s, a) \leftarrow -\infty \quad \text{if } A_{\text{mask}}(s, a) = 0$$
  - Ensures the agent never selects moves that freeze the game during exploration or execution.

---

## 🎓 3. How the AI Was Trained

```
                  ┌───────────────────────────────────────────────┐
                  │          TRAINING & DATASET PIPELINE          │
                  └───────────────────────┬───────────────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        ▼                                 ▼                                 ▼
┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│     CNN ENCODER      │      │    LSTM TRAJECTORY   │      │    DUELING D3QN      │
│   (Perception AI)    │      │    (Sequence AI)     │      │       (DRL AI)       │
├──────────────────────┤      ├──────────────────────┤      ├──────────────────────┤
│ • Dataset: Solvable  │      │ • Dataset: A* action │      │ • Prioritized Replay │
│   vs Deadlock States │      │   sequences          │      │   Buffer (PER)       │
│ • Loss: CrossEntropy │      │ • Loss: MSE (Dist) + │      │ • Loss: Huber Smooth │
│ • Optimizer: AdamW   │      │   CrossEntropy (Act) │      │   L1 Bellman Error   │
│ • Result: 94.7% Acc  │      │ • Result: MAE = 4.25 │      │ • Target Net Update  │
└──────────────────────┘      └──────────────────────┘      └──────────────────────┘
```

1. **CNN Perception Training** (`sokoban/perception/perception_eval.py`):
   - Trained on thousands of board configurations generated by the procedural reverse-pull simulator and benchmark states labeled by solvability and deadlock status.
   - Trained with Cross-Entropy Loss and `AdamW` optimizer ($\text{lr}=10^{-3}$, weight decay $=10^{-4}$).
   - Achieved **94.74% Accuracy** and **97.30% F1-Score**.

2. **LSTM Sequence Training**:
   - Trained on sequential trajectory data from optimal A* solutions and exploratory walks.
   - Jointly trained with MSE Loss for distance-to-goal regression and Cross-Entropy Loss for action prediction and tactical phase classification.
   - Achieved **MAE = 4.25**, **RMSE = 4.81**, and **84.21% Behavior Phase Accuracy**.

3. **Dueling Double DQN Training** (`sokoban/drl/drl_eval.py`):
   - Transitions $(s, a, r, s', \text{done}, A_{\text{mask}}, \mathbf{v}_{\text{strat}})$ stored in a **Prioritized Experience Replay (PER)** buffer.
   - Loss function:
     $$\mathcal{L}(\theta) = \mathbb{E}\left[\left(r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^{-}) - Q(s, a; \theta)\right)^2\right]$$
   - Target network soft updates: $\theta^{-} \leftarrow \tau \theta + (1-\tau)\theta^{-}$ with $\tau = 0.005$.

---

## 📊 4. Comprehensive Evaluation Benchmark Results

The entire pipeline was benchmarked using `python run_experiments.py` across all 7 evaluation tiers specified in the requirements:

### Primary Ablation Benchmark: Hybrid AI vs. DRL-Only
$$\text{Performance Gain} = \text{Performance}_{\text{Hybrid}} - \text{Performance}_{\text{DRL-only}}$$

| Metric | DRL-Only Baseline | Search-Only (A*) | Search + DRL (no deadlock mask) | **Full Hybrid Agent** | **Advantage (Delta)** |
|---|---|---|---|---|---|
| **Win / Solve Rate** | `0.00%` (trapped in deadlocks) | `100.00%` | `0.00%` | **`100.00%`** | **`+100.00%`** |
| **Average Reward** | `-4.03` | n/a (planner only) | `-1.20` | **`+18.40`** | **`+22.43`** |
| **Average Step Count** | `60.00` (Timeout) | `17.50` | `60.00` (Timeout) | **`17.50`** | **`42.50 steps faster`** |

> Values from the verified run stored in `experiment_results.json` (Windows 11, Python 3.11, CPU). The DRL-only reward varies slightly between runs (about -2.2 to -5.7) because exploration is not seeded; the win-rate and step results are stable.

---

### Detailed Tier-by-Tier Evaluation Metrics

| Tier | Evaluation Metric | Achieved Value | Description / Interpretation |
|---|---|---|---|
| **Tier 1: DRL** | Average Reward | `-2.35` | Baseline exploration reward without deadlock guidance |
| | Reward Variance | `0.64` | Consistency of episode returns |
| | Episode Length | `80.0` | Average steps per episode |
| | Success / Win Rate | `0.00%` | Unmasked exploration rapidly enters fatal corner deadlocks |
| | Convergence Episodes | `40` | Training episode benchmark |
| **Tier 2: Search & Planning** | A\* Path Length | `16.86` steps | Optimal shortest path length across benchmark suite |
| | A\* Expanded Nodes | `228.57` nodes | High pruning efficiency via Hungarian heuristic |
| | A\* Search Time | `~8-20 ms` | Millisecond-level solving speed |
| | Backtracking Path Length | `34.83` steps | Depth-first search path length |
| | Backtracking Nodes | `506.17` nodes | Search tree exploration size |
| | Backtracking Time | `~2-3 ms` | DFS traversal speed |
| | Average Optimality Gap | `1.214` | Backtracking path length overhead relative to optimal A\* |
| **Tier 3: CNN Perception** | Classification Accuracy | **`94.74%`** | Solvability & deadlock classification accuracy |
| | Precision | **`94.74%`** | Positive predictive value |
| | Recall | **`100.00%`** | Sensitivity / true positive detection |
| | F1-Score | **`97.30%`** | Harmonic mean of precision & recall |
| **Tier 4: LSTM Modeling** | Mean Absolute Error (MAE) | `4.71` | Remaining step prediction error |
| | Root Mean Squared Error (RMSE) | `5.24` | Distance regression error |
| | Action Prediction Accuracy | `36.84%` | Multi-class next optimal step prediction |
| | Behavior Classification Accuracy | **`84.21%`** | Tactical phase classification (Exploration/Push/Goal) |
| **Tier 5: Generative AI** | Task-Generation Validity | **`100.00%`** (1.0) | Rate of valid physical subgoal generation |
| | Constraint Satisfaction Rate | **`100.00%`** (1.0) | Rate of obeying non-deadlock physics |
| | Hallucination Rate | **`0.00%`** (0.0) | Zero hallucination via deterministic grounding |
| | Goal Interpretation Accuracy | **`100.00%`** (1.0) | Accurate target win condition identification |
| **Tier 6: RAG Retrieval** | Retrieval Precision | **`80.00%`** | Fraction of relevant retrieved tactical rules |
| | Retrieval Recall | **`70.00%`** | Fraction of target domain rules retrieved |
| | Grounded-Answer Rate | **`100.00%`** | 100% of advice statements grounded in corpus |
| | Context Relevance Score | **`0.478`** | Mean TF-IDF cosine semantic similarity |
| **Tier 7: Overall Hybrid** | $\Delta \text{Win Rate}$ | **`+100.00%`** | $\text{WinRate}_{\text{Hybrid}} - \text{WinRate}_{\text{DRL-only}}$ |
| | $\Delta \text{Average Reward}$ | **`+22.43`** | $\text{Reward}_{\text{Hybrid}} - \text{Reward}_{\text{DRL-only}}$ |
| | $\Delta \text{Step Efficiency}$ | **`+42.50 steps`** | Steps saved relative to baseline timeout |

---

## 🕹️ 5. Quickstart & Usage

### 0. Installation (Windows / Linux / macOS, Python 3.9+)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

### 1. Launch Interactive Web UI (Playable Game + AI Co-Pilot Studio)
```bash
python main.py --mode web
```
Open your browser at **[http://localhost:8000](http://localhost:8000)** to:
- Play with **Arrow Keys** or **WASD** (or touch controls).
- Click **"💡 AI Hint"** (`H`) to view real-time tactical advice and glowing directional hints.
- Click **"🤖 AI Auto-Solve"** (`Space`) to watch the Hybrid AI Agent solve the puzzle with live playback controls (1x, 2x, 4x speed).
- Click **"🎲 Generate Random"** to create procedural solvable puzzles on the fly.

### 2. Run Complete 7-Tier Evaluation Benchmark
```bash
python run_experiments.py
```
Executes all models and outputs `experiment_results.json` (takes about 2-5 minutes on a CPU).

### 3. Watch Live CLI Step-by-Step Solver
```bash
python main.py --mode solve --level microban_1 --speed 0.1
```

### 4. Interactive Terminal Game with AI Co-Pilot
```bash
python main.py --mode interactive --level microban_2
```

### 5. Run Search Planner Comparison (A* vs. Backtracking)
```bash
python main.py --mode search --level microban_4
```

### 6. Run Unit Test Suite (23 Tests)
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 📁 6. Repository File Structure

```
drl_lab_task1/
├── sokoban/
│   ├── core/
│   │   ├── grid.py             # Coordinate system, tile enums, Zobrist hash, reachability flood fill
│   │   ├── environment.py      # Gymnasium-compliant environment, multi-channel tensor (C=6), rewards
│   │   ├── levels.py           # Microban suites, benchmark sets, and procedural reverse-pull generator
│   │   └── renderer.py         # Rich ANSI colorized visual renderer & ASCII displays
│   ├── perception/
│   │   ├── cnn_encoder.py      # ResNet-style CNN with attention pooling (C_in=6 -> D=128)
│   │   ├── lstm_model.py       # LSTM sequence model for temporal action & trajectory tracking
│   │   └── perception_eval.py  # Evaluates Accuracy, Precision, Recall, F1, MAE, RMSE, Behavior Acc
│   ├── reasoning/
│   │   ├── deadlock.py         # Explicit Deadlock Detector (Corner, Wall/Line, 2x2, Static precomputation)
│   │   ├── astar.py            # A* search with Hungarian algorithm (linear_sum_assignment) bipartite heuristic
│   │   ├── backtracking.py     # Depth-First Backtracking search with cycle detection & deadlock pruning
│   │   └── search_eval.py      # Search benchmarking (Path length, Expanded nodes, Search time, Optimality gap)
│   ├── rag_llm/
│   │   ├── knowledge_base.py   # Curated Sokoban tactical principles (corridors, packing, mobility, deadlocks)
│   │   ├── retriever.py        # Semantic TF-IDF & Cosine Similarity vector space retriever
│   │   ├── strategy_advisor.py # Strategic reasoning engine generating 16-dim context vector & explanations
│   │   └── rag_eval.py         # Evaluates Validity, Constraint Satisfaction, Hallucination, Precision, Recall
│   ├── drl/
│   │   ├── replay_buffer.py    # Prioritized Experience Replay (PER) & Standard Replay Buffers
│   │   ├── dqn_agent.py        # Dueling Double DQN (D3QN) with Deadlock Action Masking ($Q(s, a) \leftarrow -\infty$)
│   │   ├── ppo_agent.py        # PPO Actor-Critic agent with GAE and masked logits
│   │   └── drl_eval.py         # DRL training & evaluation metrics (Avg reward, Variance, Episode length, Win rate)
│   ├── hybrid/
│   │   ├── pipeline.py         # Strict End-to-End Hybrid Game AI Pipeline (Env -> Perception -> Search -> RAG -> DRL)
│   │   └── comparative_study.py# Comparative Ablation Study: Performance(Hybrid) - Performance(DRL-only)
│   └── visualizer/
│       └── dashboard.py        # Terminal visualizer and interactive CLI co-pilot
├── web_app/                    # Full Interactive Web Application & AI Studio
│   ├── index.html              # Modern dark-mode UI with game board & AI co-pilot telemetry
│   ├── style.css               # Ultra-premium styling, glassmorphism, glowing goal rings, animations
│   ├── app.js                  # Frontend game loop, Web Audio sound effects, AI playback
│   └── server.py               # REST API & static file backend (Python stdlib http.server)
├── tests/                      # Full unit test suite (23 passing tests)
├── main.py                     # Unified CLI for Web UI, solver, interactive play, search, and evaluations
├── run_experiments.py          # Complete 7-tier evaluation benchmark runner
├── experiment_results.json     # Saved JSON results of all evaluation metrics
├── requirements.txt            # Package dependencies
└── README.md                   # Master Documentation
```
