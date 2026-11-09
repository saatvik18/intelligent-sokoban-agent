"""
Sokoban Gymnasium/Gym style Environment.
"""

from typing import Tuple, Dict, Any, Optional, Union
import numpy as np

from sokoban.core.grid import (
    SokobanState, Action, Tile, ACTION_NAMES, parse_level_string
)
from sokoban.core.levels import get_level, generate_random_level


class SokobanEnv:
    """
    OpenAI Gym / Gymnasium compliant Sokoban Environment.
    Supports multi-channel grid tensor observation, reward shaping, and deadlock penalties.
    """

    def __init__(
        self,
        level_name: str = "microban_1",
        max_steps: int = 150,
        fixed_grid_shape: Tuple[int, int] = (8, 8),
        step_penalty: float = -0.02,
        box_target_reward: float = 2.0,
        box_off_target_penalty: float = -1.5,
        solved_reward: float = 15.0,
        deadlock_penalty: float = -5.0,
        invalid_move_penalty: float = -0.1,
    ):
        self.level_name = level_name
        self.max_steps = max_steps
        self.fixed_grid_shape = fixed_grid_shape

        # Reward hyperparameters
        self.step_penalty = step_penalty
        self.box_target_reward = box_target_reward
        self.box_off_target_penalty = box_off_target_penalty
        self.solved_reward = solved_reward
        self.deadlock_penalty = deadlock_penalty
        self.invalid_move_penalty = invalid_move_penalty

        self.state: Optional[SokobanState] = None
        self.initial_state: Optional[SokobanState] = None
        self.step_count: int = 0
        self.prev_boxes_on_target: int = 0
        self.total_reward: float = 0.0

        # Action space: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.action_space_n = 4
        # Channels: Walls, Unplaced Boxes, Boxes on target, Targets, Player, Reachable Area
        self.observation_channels = 6
        self.observation_shape = (self.observation_channels, fixed_grid_shape[0], fixed_grid_shape[1])

        self.reset(level_name=level_name)

    def reset(
        self,
        seed: Optional[int] = None,
        level_name: Optional[str] = None,
        custom_state: Optional[SokobanState] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial or designated level state."""
        if seed is not None:
            np.random.seed(seed)

        if custom_state is not None:
            self.state = custom_state
        elif level_name is not None:
            self.level_name = level_name
            self.state = get_level(level_name)
        else:
            self.state = get_level(self.level_name)

        self.initial_state = self.state
        self.step_count = 0
        self.prev_boxes_on_target = self.state.num_boxes_on_target()
        self.total_reward = 0.0

        obs = self.get_observation()
        info = self._get_info()
        return obs, info

    def get_observation(self, deadlock_map: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract multi-channel tensor representation of current state."""
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        return self.state.to_feature_tensor(
            deadlock_map=deadlock_map,
            fixed_size=self.fixed_grid_shape
        )

    def step(self, action: Union[int, Action]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute an action.
        Returns: (observation, reward, terminated, truncated, info)
        """
        if isinstance(action, int):
            action = Action(action)

        self.step_count += 1
        reward = self.step_penalty
        terminated = False
        truncated = False

        new_state, is_valid, is_push = self.state.apply_action(action)

        if not is_valid:
            reward += self.invalid_move_penalty
            # State remains unchanged
        else:
            self.state = new_state
            current_boxes_on_target = self.state.num_boxes_on_target()
            target_diff = current_boxes_on_target - self.prev_boxes_on_target

            if target_diff > 0:
                reward += self.box_target_reward * target_diff
            elif target_diff < 0:
                reward += self.box_off_target_penalty * abs(target_diff)

            self.prev_boxes_on_target = current_boxes_on_target

        # Check win condition
        if self.state.is_solved():
            reward += self.solved_reward
            terminated = True

        # Check step limit
        if self.step_count >= self.max_steps and not terminated:
            truncated = True

        self.total_reward += reward
        obs = self.get_observation()
        info = self._get_info(is_valid=is_valid, is_push=is_push)

        return obs, reward, terminated, truncated, info

    def _get_info(self, is_valid: bool = True, is_push: bool = False) -> Dict[str, Any]:
        return {
            "step": self.step_count,
            "is_solved": self.state.is_solved() if self.state else False,
            "boxes_on_target": self.state.num_boxes_on_target() if self.state else 0,
            "total_targets": len(self.state.targets) if self.state else 0,
            "is_valid": is_valid,
            "is_push": is_push,
            "player_pos": self.state.player if self.state else None,
            "boxes": set(self.state.boxes) if self.state else set(),
            "state": self.state
        }

    def render(self, mode: str = "ansi") -> Optional[str]:
        """Render the environment state."""
        if mode == "ansi":
            return self.state.to_ascii() if self.state else ""
        elif mode == "human":
            print(self.state.to_ascii() if self.state else "")
            print(f"Step: {self.step_count} | Targets: {self.state.num_boxes_on_target()}/{len(self.state.targets)}")
        return None
