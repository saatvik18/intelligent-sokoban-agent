"""
DRL Training and Evaluation module for DQN & PPO with full metric logging.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from sokoban.core.environment import SokobanEnv
from sokoban.reasoning.deadlock import get_action_deadlock_mask
from sokoban.rag_llm.strategy_advisor import SokobanStrategyAdvisor
from sokoban.drl.dqn_agent import DQNAgent


def train_drl_agent(
    level_name: str = "microban_1",
    num_episodes: int = 100,
    max_steps_per_episode: int = 80,
    use_deadlock_mask: bool = True,
    use_rag_strategy: bool = True,
    seed: int = 42
) -> Tuple[DQNAgent, Dict[str, Any]]:
    """
    Train Dueling DQN agent on Sokoban environment with optional deadlock masking and RAG strategy context.
    """
    env = SokobanEnv(level_name=level_name, max_steps=max_steps_per_episode)
    agent = DQNAgent(in_channels=6, num_actions=4, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=num_episodes * 20)
    advisor = SokobanStrategyAdvisor() if use_rag_strategy else None

    episode_rewards = []
    episode_lengths = []
    win_flags = []
    convergence_episode = None

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        state = info["state"]
        ep_reward = 0.0
        steps = 0

        while True:
            # 1. Compute deadlock action mask if enabled
            mask = get_action_deadlock_mask(state) if use_deadlock_mask else np.ones(4, dtype=bool)

            # 2. Compute RAG strategic context vector if enabled
            context_vec = None
            if advisor is not None:
                context_vec, _ = advisor.generate_strategy_context(state)

            # 3. Sample action
            action = agent.select_action(obs, action_mask=mask, context_vec=context_vec)

            # 4. Step environment
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            next_state = next_info["state"]
            done = terminated or truncated

            next_mask = get_action_deadlock_mask(next_state) if use_deadlock_mask else np.ones(4, dtype=bool)
            next_context_vec = None
            if advisor is not None:
                next_context_vec, _ = advisor.generate_strategy_context(next_state)

            # 5. Store in replay buffer & train periodically
            agent.memory.push(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done,
                action_mask=next_mask,
                context_vec=context_vec
            )

            if steps % 4 == 0:
                agent.train_step(batch_size=32)

            obs = next_obs
            state = next_state
            ep_reward += reward
            steps += 1

            if done:
                is_win = 1.0 if next_info["is_solved"] else 0.0
                episode_rewards.append(ep_reward)
                episode_lengths.append(steps)
                win_flags.append(is_win)

                # Check convergence (win rate over last 10 episodes >= 80%)
                if convergence_episode is None and len(win_flags) >= 10:
                    if np.mean(win_flags[-10:]) >= 0.8:
                        convergence_episode = ep
                break

    metrics = {
        "average_reward": round(float(np.mean(episode_rewards)), 4),
        "reward_variance": round(float(np.var(episode_rewards)), 4),
        "episode_length": round(float(np.mean(episode_lengths)), 2),
        "success_win_rate": round(float(np.mean(win_flags)), 4),
        "convergence_episodes": convergence_episode if convergence_episode is not None else num_episodes,
        "sample_efficiency": round(float(np.sum(win_flags)) / max(1, float(np.sum(episode_lengths))), 4),
        "history": {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "wins": win_flags
        }
    }

    return agent, metrics


def evaluate_drl_agent(
    agent: DQNAgent,
    level_name: str = "microban_1",
    num_eval_episodes: int = 20,
    use_deadlock_mask: bool = True,
    use_rag_strategy: bool = True
) -> Dict[str, Any]:
    """Evaluate trained agent deterministically."""
    env = SokobanEnv(level_name=level_name, max_steps=100)
    advisor = SokobanStrategyAdvisor() if use_rag_strategy else None

    rewards = []
    lengths = []
    wins = []

    for ep in range(num_eval_episodes):
        obs, info = env.reset(seed=1000 + ep)
        state = info["state"]
        ep_reward = 0.0
        steps = 0

        while True:
            mask = get_action_deadlock_mask(state) if use_deadlock_mask else np.ones(4, dtype=bool)
            context_vec = None
            if advisor is not None:
                context_vec, _ = advisor.generate_strategy_context(state)

            action = agent.select_action(obs, action_mask=mask, context_vec=context_vec, deterministic=True)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            ep_reward += reward
            steps += 1
            obs = next_obs
            state = next_info["state"]

            if terminated or truncated:
                rewards.append(ep_reward)
                lengths.append(steps)
                wins.append(1.0 if next_info["is_solved"] else 0.0)
                break

    return {
        "average_reward": round(float(np.mean(rewards)), 4),
        "reward_variance": round(float(np.var(rewards)), 4),
        "episode_length": round(float(np.mean(lengths)), 2),
        "success_win_rate": round(float(np.mean(wins)), 4),
        "sample_efficiency": round(float(np.sum(wins)) / max(1, float(np.sum(lengths))), 4)
    }
