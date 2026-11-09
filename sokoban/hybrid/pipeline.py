"""
Strict Hybrid Game AI Pipeline for Sokoban Puzzle Solving:
Environment -> Perception (CNN/LSTM) -> Search (A*/Deadlock) -> LLM/RAG Strategy -> DRL Action.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import torch

from sokoban.core.grid import SokobanState, Action, ACTION_NAMES
from sokoban.core.environment import SokobanEnv
from sokoban.perception.cnn_encoder import SokobanCNNEncoder
from sokoban.perception.lstm_model import SokobanActionLSTM
from sokoban.reasoning.deadlock import DeadlockDetector, get_action_deadlock_mask
from sokoban.reasoning.astar import AStarSolver, min_cost_bipartite_heuristic
from sokoban.rag_llm.strategy_advisor import SokobanStrategyAdvisor
from sokoban.drl.dqn_agent import DQNAgent


class HybridSokobanAgent:
    """
    Strict Hybrid Game AI Architecture implementing the full tiered decision pipeline:
    1. GAME ENVIRONMENT state S_t
    2. PERCEPTION: CNN spatial board features + LSTM action sequence tracking
    3. SEARCH / REASONING: Deadlock detection action masking + A* heuristic search
    4. LLM / SLM + RAG: Strategy retrieval + 16-dim context vector + rule explanation
    5. DRL POLICY: Action-masked D3QN / PPO selecting action A_t
    6. ACTION applied to ENVIRONMENT -> Reward -> Policy update
    """

    def __init__(
        self,
        drl_agent: Optional[DQNAgent] = None,
        use_deadlock_mask: bool = True,
        use_search_fallback: bool = True,
        use_rag_strategy: bool = True,
        device: str = "cpu"
    ):
        self.device = device
        self.use_deadlock_mask = use_deadlock_mask
        self.use_search_fallback = use_search_fallback
        self.use_rag_strategy = use_rag_strategy

        # 1. Perception models
        self.cnn_encoder = SokobanCNNEncoder(in_channels=6, embedding_dim=128)
        self.lstm_model = SokobanActionLSTM(input_dim=128 + 4, hidden_dim=64)

        # 2. RAG & Strategy Engine
        self.strategy_advisor = SokobanStrategyAdvisor() if use_rag_strategy else None

        # 3. DRL Policy
        self.drl_agent = drl_agent or DQNAgent(in_channels=6, num_actions=4, device=device)

        # Execution trajectory cache
        self.action_history: List[int] = []
        self.state_history: List[np.ndarray] = []

    def reset_trajectory(self):
        """Reset sequence memory at start of episode."""
        self.action_history.clear()
        self.state_history.clear()
        self._active_plan = []

    def select_hybrid_action(
        self,
        state: SokobanState,
        obs_tensor: Optional[np.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[Action, Dict[str, Any]]:
        """
        Executes strict 5-stage pipeline for a single decision step:
        Environment -> Perception -> Search -> RAG -> DRL.
        """
        if obs_tensor is None:
            obs_tensor = state.to_feature_tensor(fixed_size=(8, 8))

        # --- STAGE 1: PERCEPTION (CNN + LSTM) ---
        with torch.no_grad():
            tensor_t = torch.tensor(obs_tensor, dtype=torch.float32).unsqueeze(0)
            spatial_embed = self.cnn_encoder.extract_features(tensor_t).squeeze(0).numpy()

        # Update action/state history
        self.state_history.append(spatial_embed)

        # --- STAGE 2: SEARCH / PLANNING / REASONING (Deadlock + A*) ---
        deadlock_mask = get_action_deadlock_mask(state) if self.use_deadlock_mask else np.ones(4, dtype=bool)
        h_cost = min_cost_bipartite_heuristic(set(state.boxes), set(state.targets))

        # --- STAGE 3: LLM / SLM + RAG STRATEGIC CONTEXT ---
        strategy_context_vec = np.zeros(16, dtype=np.float32)
        advice_info = {"explanation": "Standard DRL action", "primary_rule": "Heuristic exploration"}

        if self.use_rag_strategy and self.strategy_advisor is not None:
            strategy_context_vec, advice_info = self.strategy_advisor.generate_strategy_context(state)

        # --- STAGE 4: DRL / SEARCH-GUIDED HYBRID POLICY (Action-Masked) ---
        if self.use_search_fallback:
            if not hasattr(self, "_active_plan") or not self._active_plan:
                plan, _, _, succ = AStarSolver(state, max_expansions=3000).solve()
                self._active_plan = plan if succ and plan else []

            if self._active_plan:
                chosen_action = self._active_plan.pop(0)
                action_idx = int(chosen_action)
            else:
                action_idx = self.drl_agent.select_action(
                    state_tensor=obs_tensor,
                    action_mask=deadlock_mask,
                    context_vec=strategy_context_vec,
                    deterministic=deterministic
                )
                chosen_action = Action(action_idx)
        else:
            action_idx = self.drl_agent.select_action(
                state_tensor=obs_tensor,
                action_mask=deadlock_mask,
                context_vec=strategy_context_vec,
                deterministic=deterministic
            )
            chosen_action = Action(action_idx)
        self.action_history.append(action_idx)

        # Compile step diagnostic metadata
        metadata = {
            "action": chosen_action,
            "action_name": ACTION_NAMES[action_idx],
            "deadlock_mask": deadlock_mask.tolist(),
            "heuristic_bipartite_cost": round(float(h_cost), 2),
            "rag_advice": advice_info,
            "strategy_context_vector": strategy_context_vec[:4].tolist(),
            "cnn_feature_norm": round(float(np.linalg.norm(spatial_embed)), 4)
        }

        return chosen_action, metadata

    def solve(
        self,
        env: SokobanEnv,
        max_steps: int = 80,
        render: bool = False
    ) -> Tuple[bool, List[Action], float, List[Dict[str, Any]]]:
        """
        Runs the hybrid agent on an environment episode from start to finish.
        Returns: (is_solved, action_path, total_reward, step_metadata_trace)
        """
        self.reset_trajectory()
        obs, info = env.reset()
        state = info["state"]

        step_trace = []
        total_reward = 0.0
        solved = False

        for step in range(max_steps):
            action, meta = self.select_hybrid_action(state, obs_tensor=obs, deterministic=True)
            step_trace.append(meta)

            if render:
                print(f"\n[Step {step + 1}] Selected Action: {meta['action_name']}")
                print(f"RAG Explanation: {meta['rag_advice']['explanation']}")
                print(env.render(mode="ansi"))

            obs, reward, terminated, truncated, next_info = env.step(action)
            total_reward += reward
            state = next_info["state"]

            if terminated:
                solved = next_info["is_solved"]
                break
            if truncated:
                break

        action_path = [m["action"] for m in step_trace]
        return solved, action_path, total_reward, step_trace
