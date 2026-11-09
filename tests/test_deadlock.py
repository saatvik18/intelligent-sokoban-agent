import unittest
import numpy as np
from sokoban.core.grid import parse_level_string, Action
from sokoban.reasoning.deadlock import DeadlockDetector, is_deadlocked_state, get_action_deadlock_mask


class TestDeadlockDetection(unittest.TestCase):

    def test_corner_deadlock(self):
        # Box pushed into top-left corner with no target
        level_str = """
#####
#$  #
# @ #
#  .#
#####
"""
        state = parse_level_string(level_str)
        is_dead, reason = is_deadlocked_state(state)
        self.assertTrue(is_dead)

    def test_corner_non_deadlock_on_target(self):
        # Box in corner but on a target is NOT a deadlock
        level_str = """
#####
#*  #
# @ #
#   #
#####
"""
        state = parse_level_string(level_str)
        is_dead, _ = is_deadlocked_state(state)
        self.assertFalse(is_dead)

    def test_2x2_square_deadlock(self):
        level_str = """
######
#    #
# $$ #
# ## #
# .@.#
######
"""
        state = parse_level_string(level_str)
        is_dead, reason = is_deadlocked_state(state)
        self.assertTrue(is_dead)

    def test_action_deadlock_mask(self):
        # Player is below a box against a top wall. Pushing UP puts box in corner!
        level_str = """
#####
#   #
# $ #
# @.#
#####
"""
        state = parse_level_string(level_str)
        mask = get_action_deadlock_mask(state)
        self.assertEqual(len(mask), 4)
        self.assertTrue(isinstance(mask[0], (bool, np.bool_)))


if __name__ == "__main__":
    unittest.main()
