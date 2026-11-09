"""
Replay Buffer implementations: Standard Uniform Experience Replay and Prioritized Experience Replay (PER).
"""

from typing import Tuple, List, Dict, Any, Optional
import random
import numpy as np
import torch


class ReplayBuffer:
    """Standard Uniform Experience Replay Buffer."""

    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        context_vec: Optional[np.ndarray] = None
    ):
        if context_vec is None:
            context_vec = np.zeros(16, dtype=np.float32)

        item = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "action_mask": action_mask,
            "context_vec": context_vec
        }

        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)

        states = torch.tensor(np.array([b["state"] for b in batch]), dtype=torch.float32)
        actions = torch.tensor([b["action"] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32)
        next_states = torch.tensor(np.array([b["next_state"] for b in batch]), dtype=torch.float32)
        dones = torch.tensor([b["done"] for b in batch], dtype=torch.float32)
        action_masks = torch.tensor(np.array([b["action_mask"] for b in batch]), dtype=torch.bool)
        context_vecs = torch.tensor(np.array([b["context_vec"] for b in batch]), dtype=torch.float32)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "action_masks": action_masks,
            "context_vecs": context_vecs
        }

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) with proportional prioritization."""

    def __init__(self, capacity: int = 20000, alpha: float = 0.6, beta_start: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.buffer: List[Dict[str, Any]] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        context_vec: Optional[np.ndarray] = None
    ):
        if context_vec is None:
            context_vec = np.zeros(16, dtype=np.float32)

        item = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "action_mask": action_mask,
            "context_vec": context_vec
        }

        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, torch.Tensor]:
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights_t = torch.tensor(weights, dtype=torch.float32)

        states = torch.tensor(np.array([s["state"] for s in samples]), dtype=torch.float32)
        actions = torch.tensor([s["action"] for s in samples], dtype=torch.long)
        rewards = torch.tensor([s["reward"] for s in samples], dtype=torch.float32)
        next_states = torch.tensor(np.array([s["next_state"] for s in samples]), dtype=torch.float32)
        dones = torch.tensor([s["done"] for s in samples], dtype=torch.float32)
        action_masks = torch.tensor(np.array([s["action_mask"] for s in samples]), dtype=torch.bool)
        context_vecs = torch.tensor(np.array([s["context_vec"] for s in samples]), dtype=torch.float32)

        batch_dict = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "action_masks": action_masks,
            "context_vecs": context_vecs
        }

        return batch_dict, indices, weights_t

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, prio in zip(indices, priorities):
            p = max(float(prio), 1e-6)
            self.priorities[idx] = p
            if p > self.max_priority:
                self.max_priority = p

    def __len__(self) -> int:
        return len(self.buffer)
