"""
LSTM-based Temporal Action Sequence and Trajectory Modeling for Sokoban.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SokobanActionLSTM(nn.Module):
    """
    Recurrent sequence model:
    Processes trajectory of spatial embeddings and actions over time.
    Predicts next action sequence, future state progress, and behavior classification.
    """

    def __init__(
        self,
        input_dim: int = 128 + 4,  # Spatial embedding + one-hot action
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_actions: int = 4,
        num_classes: int = 3  # Behavior/Phase: Exploration, Box Pushing, Goal Alignment
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional or multilayer LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # 1. Action prediction head (predicts next optimal action in sequence)
        self.action_pred_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )

        # 2. Distance/Reward regression head (predicts remaining distance to solution)
        self.distance_regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # 3. Behavior / state phase classification head
        self.behavior_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(
        self,
        seq_inputs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Input: (Batch, Seq_Len, Input_Dim)
        Returns: (action_logits, distance_pred, behavior_logits, (h_n, c_n))
        """
        lstm_out, (hn, cn) = self.lstm(seq_inputs, hidden_state)
        # Focus on the most recent step representation
        last_out = lstm_out[:, -1, :]

        action_logits = self.action_pred_head(last_out)
        dist_pred = self.distance_regression_head(last_out).squeeze(-1)
        behavior_logits = self.behavior_classifier(last_out)

        return action_logits, dist_pred, behavior_logits, (hn, cn)

    def get_sequence_embedding(
        self,
        seq_inputs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Extract hidden feature representation for DRL policy integration."""
        lstm_out, (hn, cn) = self.lstm(seq_inputs, hidden_state)
        return lstm_out[:, -1, :], (hn, cn)
