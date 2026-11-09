"""
Unit tests for Sokoban Grid and Environment.
"""

import unittest
import numpy as np
from sokoban.core.grid import (
    SokobanState, Action, Tile, parse_level_string, flood_fill_reachability, canonical_state_hash
)
from sokoban.core.environment import SokobanEnv
from sokoban.core.levels import get_level, generate_random_level


class TestSokobanCore(unittest.TestCase):

    def test_parse_and_render_level(self):
        level_str = """
#####
#@$.#
#####
"""
        state = parse_level_string(level_str)
        self.assertEqual(state.player, (1, 1))
        self.assertEqual(len(state.boxes), 1)
        self.assertEqual(len(state.targets), 1)
        self.assertFalse(state.is_solved())

    def test_reachability_flood_fill(self):
        level_str = """
#######
#@   .#
#  #  #
#  $  #
#######
"""
        state = parse_level_string(level_str)
        reachable = flood_fill_reachability(state.player, set(state.walls), set(state.boxes), state.height, state.width)
        self.assertIn((1, 1), reachable)
        self.assertIn((1, 2), reachable)
        self.assertNotIn((3, 3), reachable)  # Box location is not traversable floor

    def test_state_actions(self):
        level_str = """
######
#@ $.#
######
"""
        state = parse_level_string(level_str)
        # Move right towards box
        s1, valid, is_push = state.apply_action(Action.RIGHT)
        self.assertTrue(valid)
        self.assertFalse(is_push)
        self.assertEqual(s1.player, (1, 2))

        # Push box right
        s2, valid, is_push = s1.apply_action(Action.RIGHT)
        self.assertTrue(valid)
        self.assertTrue(is_push)
        self.assertEqual(s2.player, (1, 3))
        self.assertIn((1, 4), s2.boxes)

        # Illegal push into wall
        s3, valid, is_push = s2.apply_action(Action.RIGHT)
        self.assertFalse(valid)

    def test_environment_step_and_reward(self):
        env = SokobanEnv(level_name="microban_1")
        obs, info = env.reset()
        self.assertEqual(obs.shape, (6, 8, 8))
        self.assertFalse(info["is_solved"])

        # Execute actions
        obs, reward, term, trunc, info = env.step(Action.UP)
        self.assertIsInstance(reward, float)
        self.assertFalse(trunc)

    def test_procedural_generator(self):
        state = generate_random_level(height=7, width=7, num_boxes=1, seed=42)
        self.assertIsInstance(state, SokobanState)
        self.assertEqual(len(state.boxes), 1)
        self.assertEqual(len(state.targets), 1)


if __name__ == "__main__":
    unittest.main()
