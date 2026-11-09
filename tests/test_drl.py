"""
Unit tests for Replay Buffers, Dueling DQN, and PPO agent architectures.
"""

import unittest
import numpy as np
import torch
from sokoban.drl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from sokoban.drl.dqn_agent import DQNAgent, DuelingDQNNetwork
from sokoban.drl.ppo_agent import PPOAgent


class TestDRLModules(unittest.TestCase):

    def test_replay_buffer(self):
        buf = ReplayBuffer(capacity=50)
        s = np.zeros((6, 8, 8), dtype=np.float32)
        m = np.ones(4, dtype=bool)
        for i in range(10):
            buf.push(s, 0, 1.0, s, False, m)
        self.assertEqual(len(buf), 10)
        batch = buf.sample(batch_size=4)
        self.assertEqual(batch["states"].shape, (4, 6, 8, 8))

    def test_prioritized_replay_buffer(self):
        per = PrioritizedReplayBuffer(capacity=50)
        s = np.zeros((6, 8, 8), dtype=np.float32)
        m = np.ones(4, dtype=bool)
        for i in range(10):
            per.push(s, 1, 0.5, s, False, m)
        self.assertEqual(len(per), 10)
        batch, idxs, weights = per.sample(batch_size=4)
        self.assertEqual(len(idxs), 4)

    def test_dqn_action_selection(self):
        agent = DQNAgent(in_channels=6, num_actions=4, epsilon_start=0.0)  # Pure greedy
        s = np.zeros((6, 8, 8), dtype=np.float32)
        mask = np.array([True, False, False, False])
        act = agent.select_action(s, action_mask=mask, deterministic=True)
        self.assertEqual(act, 0)  # Only action 0 is unmasked

    def test_ppo_select_action(self):
        agent = PPOAgent(in_channels=6, num_actions=4)
        s = np.zeros((6, 8, 8), dtype=np.float32)
        mask = np.array([False, True, False, False])
        act, log_prob, val = agent.select_action(s, action_mask=mask, deterministic=True)
        self.assertEqual(act, 1)


if __name__ == "__main__":
    unittest.main()
