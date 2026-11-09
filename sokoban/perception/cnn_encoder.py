"""
CNN-based Visual State Representation & Feature Extraction for Sokoban Grids.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual convolutional block with LayerNorm and LeakyReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = F.leaky_relu(self.gn1(self.conv1(x)), negative_slope=0.1)
        out = self.gn2(self.conv2(out))
        return F.leaky_relu(out + res, negative_slope=0.1)


class SokobanCNNEncoder(nn.Module):
    """
    CNN State Representation model:
    Extracts spatial board features from multi-channel grid tensors (C_in x H x W).
    Provides spatial embedding vector and auxiliary perception classification head.
    """

    def __init__(
        self,
        in_channels: int = 6,
        embedding_dim: int = 128,
        num_classes: int = 2  # e.g., Solvable vs Deadlocked / Target Match
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        # Initial projection
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.1)
        )

        # Residual backbone
        self.res1 = ResidualBlock(32)
        self.conv_down = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1)
        )
        self.res2 = ResidualBlock(64)

        # Global Attention Pooling
        self.attn_weights = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Feature projection
        self.fc_embed = nn.Sequential(
            nn.Linear(64 + 64, embedding_dim),  # Attentive pool + Adaptive Avg pool
            nn.LayerNorm(embedding_dim),
            nn.LeakyReLU(0.1)
        )

        # Auxiliary Perception Classifier Head (Solvability / Deadlock risk prediction)
        self.classifier_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning the dense spatial embedding z in R^embedding_dim.
        Input: (B, C, H, W)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        out = self.init_conv(x)
        out = self.res1(out)
        out = self.conv_down(out)
        out = self.res2(out)

        # Attention pooling
        weights = self.attn_weights(out)
        attn_pool = (out * weights).sum(dim=(2, 3)) / (weights.sum(dim=(2, 3)) + 1e-6)
        avg_pool = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)

        combined = torch.cat([attn_pool, avg_pool], dim=1)
        embed = self.fc_embed(combined)
        return embed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (embedding, classification_logits)
        """
        embed = self.extract_features(x)
        logits = self.classifier_head(embed)
        return embed, logits
