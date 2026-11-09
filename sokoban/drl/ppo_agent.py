"""
Proximal Policy Optimization (PPO) Actor-Critic Agent with Action Masking and GAE.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from sokoban.perception.cnn_encoder import SokobanCNNEncoder


class ActorCriticNetwork(nn.Module):
    """Actor-Critic architecture for PPO with action masking support."""

    def __init__(
        self,
        in_channels: int = 6,
        num_actions: int = 4,
        cnn_embed_dim: int = 128,
        strategy_dim: int = 16
    ):
        super().__init__()
        self.cnn = SokobanCNNEncoder(in_channels=in_channels, embedding_dim=cnn_embed_dim)
        fused_dim = cnn_embed_dim + strategy_dim

        # Actor head (policy logits)
        self.actor = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, num_actions)
        )

        # Critic head (state value)
        self.critic = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        state_tensor: torch.Tensor,
        context_vec: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial = self.cnn.extract_features(state_tensor)
        if context_vec is None:
            context_vec = torch.zeros((spatial.size(0), 16), device=spatial.device)
        fused = torch.cat([spatial, context_vec], dim=1)

        logits = self.actor(fused)
        value = self.critic(fused).squeeze(-1)
        return logits, value


class PPOAgent:
    """PPO Agent with Generalized Advantage Estimation (GAE)."""

    def __init__(
        self,
        in_channels: int = 6,
        num_actions: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cpu"
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)

        self.network = ActorCriticNetwork(in_channels=in_channels, num_actions=num_actions).to(self.device)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr, weight_decay=1e-4)

    def select_action(
        self,
        state_tensor: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        context_vec: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Returns: (action, log_prob, value)
        """
        self.network.eval()
        with torch.no_grad():
            s_t = torch.tensor(state_tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
            c_t = torch.tensor(context_vec, dtype=torch.float32, device=self.device).unsqueeze(0) if context_vec is not None else None
            logits, value = self.network(s_t, c_t)

            if action_mask is not None:
                mask_t = torch.tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
                logits[~mask_t] = -1e9

            dist = Categorical(logits=logits)
            if deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
                log_prob = float(dist.log_prob(torch.tensor(action, device=self.device)).item())
            else:
                act_tensor = dist.sample()
                action = int(act_tensor.item())
                log_prob = float(dist.log_prob(act_tensor).item())

            val = float(value.item())

        return action, log_prob, val

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float
    ) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages and discounted returns."""
        advantages = []
        gae = 0.0
        vals = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * vals[t + 1] * (1.0 - float(dones[t])) - vals[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)

        returns = [adv + v for adv, v in zip(advantages, values)]
        return advantages, returns
