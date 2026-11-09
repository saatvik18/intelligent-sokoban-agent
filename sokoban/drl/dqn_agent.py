"""
Dueling Double Deep Q-Network (D3QN) Agent with Action Masking and Multi-Modal Feature Fusion.
"""

from typing import Tuple, Dict, Any, Optional
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sokoban.perception.cnn_encoder import SokobanCNNEncoder
from sokoban.drl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN architecture fusing:
    - Spatial board features (via CNN Encoder)
    - Temporal context (via LSTM state)
    - High-level RAG strategic context vector (16-dim)
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_actions: int = 4,
        cnn_embed_dim: int = 128,
        strategy_dim: int = 16
    ):
        super().__init__()
        self.num_actions = num_actions
        self.cnn = SokobanCNNEncoder(in_channels=in_channels, embedding_dim=cnn_embed_dim)

        fused_dim = cnn_embed_dim + strategy_dim

        # Dueling Streams
        # 1. State Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )

        # 2. Advantage Stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, num_actions)
        )

    def forward(self, state_tensor: torch.Tensor, context_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        state_tensor: (B, C, H, W)
        context_vec: (B, 16)
        Returns: Q-values (B, 4)
        """
        spatial_feat = self.cnn.extract_features(state_tensor)

        if context_vec is None:
            context_vec = torch.zeros((spatial_feat.size(0), 16), device=spatial_feat.device)

        fused = torch.cat([spatial_feat, context_vec], dim=1)

        val = self.value_stream(fused)
        adv = self.advantage_stream(fused)

        # Dueling formula: Q(s, a) = V(s) + (A(s, a) - mean(A))
        q_vals = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q_vals


class DQNAgent:
    """
    Dueling Double DQN Agent with Deadlock Action Masking.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_actions: int = 4,
        lr: float = 5e-4,
        gamma: float = 0.98,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 2000,
        target_update_tau: float = 0.005,
        use_per: bool = True,
        device: str = "cpu"
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / max(1, epsilon_decay_steps)
        self.tau = target_update_tau
        self.device = torch.device(device)
        self.use_per = use_per

        # Networks
        self.policy_net = DuelingDQNNetwork(in_channels=in_channels, num_actions=num_actions).to(self.device)
        self.target_net = DuelingDQNNetwork(in_channels=in_channels, num_actions=num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)

        if use_per:
            self.memory = PrioritizedReplayBuffer(capacity=20000)
        else:
            self.memory = ReplayBuffer(capacity=20000)

        self.step_count = 0

    def select_action(
        self,
        state_tensor: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        context_vec: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> int:
        """
        Epsilon-greedy action selection with Deadlock Action Masking.
        """
        valid_actions = [i for i in range(self.num_actions) if action_mask is None or action_mask[i]]
        if not valid_actions:
            valid_actions = list(range(self.num_actions))

        if not deterministic and random.random() < self.epsilon:
            return random.choice(valid_actions)

        self.policy_net.eval()
        with torch.no_grad():
            s_t = torch.tensor(state_tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
            c_t = torch.tensor(context_vec, dtype=torch.float32, device=self.device).unsqueeze(0) if context_vec is not None else None
            q_values = self.policy_net(s_t, c_t).squeeze(0).cpu().numpy()

        if action_mask is not None:
            # Mask invalid or deadlocking actions
            for i in range(self.num_actions):
                if not action_mask[i]:
                    q_values[i] = -1e9

        return int(np.argmax(q_values))

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def train_step(self, batch_size: int = 64) -> Optional[float]:
        """Perform one Double DQN gradient update step."""
        if len(self.memory) < batch_size:
            return None

        self.step_count += 1
        self.policy_net.train()

        if self.use_per:
            batch, indices, is_weights = self.memory.sample(batch_size)
            is_weights = is_weights.to(self.device)
        else:
            batch = self.memory.sample(batch_size)
            indices, is_weights = None, torch.ones(batch_size, device=self.device)

        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)
        next_action_masks = batch["action_masks"].to(self.device)
        context_vecs = batch["context_vecs"].to(self.device)

        # Current Q-values: Q(s, a)
        curr_q = self.policy_net(states, context_vecs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN Target computation:
        with torch.no_grad():
            # 1. Best action in next state from policy network
            next_q_policy = self.policy_net(next_states, context_vecs)
            # Mask illegal moves in next state
            next_q_policy[~next_action_masks] = -1e9
            best_next_actions = next_q_policy.argmax(dim=1, keepdim=True)

            # 2. Evaluate Q-value using target network
            next_q_target = self.target_net(next_states, context_vecs).gather(1, best_next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q_target

        td_errors = torch.abs(curr_q - target_q).detach().cpu().numpy()
        loss = (is_weights * F.smooth_l1_loss(curr_q, target_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Update PER priorities
        if self.use_per and indices is not None:
            self.memory.update_priorities(indices, td_errors + 1e-5)

        # Soft update target network: theta_target = tau * theta_policy + (1 - tau) * theta_target
        for p_target, p_policy in zip(self.target_net.parameters(), self.policy_net.parameters()):
            p_target.data.copy_(self.tau * p_policy.data + (1.0 - self.tau) * p_target.data)

        self.update_epsilon()
        return float(loss.item())
