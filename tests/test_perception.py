"""
Unit tests for CNN Perception and LSTM Sequence models.
"""

import unittest
import torch
import numpy as np
from sokoban.perception.cnn_encoder import SokobanCNNEncoder
from sokoban.perception.lstm_model import SokobanActionLSTM


class TestPerceptionModels(unittest.TestCase):

    def test_cnn_encoder_forward(self):
        model = SokobanCNNEncoder(in_channels=6, embedding_dim=128, num_classes=2)
        dummy_input = torch.randn(4, 6, 8, 8)
        embed, logits = model(dummy_input)

        self.assertEqual(embed.shape, (4, 128))
        self.assertEqual(logits.shape, (4, 2))

    def test_lstm_model_forward(self):
        model = SokobanActionLSTM(input_dim=132, hidden_dim=64, num_layers=1, num_actions=4, num_classes=3)
        dummy_seq = torch.randn(4, 5, 132)  # Batch=4, Seq_len=5, Dim=132
        act_logits, dist_pred, beh_logits, (hn, cn) = model(dummy_seq)

        self.assertEqual(act_logits.shape, (4, 4))
        self.assertEqual(dist_pred.shape, (4,))
        self.assertEqual(beh_logits.shape, (4, 3))


if __name__ == "__main__":
    unittest.main()
