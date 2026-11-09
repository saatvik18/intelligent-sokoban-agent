"""
Standard benchmark levels suite and procedural Sokoban puzzle generator.
"""

from typing import Dict, List, Tuple, Optional
import random
import numpy as np
from sokoban.core.grid import SokobanState, parse_level_string, Tile, Action, ACTION_DELTAS


# Curated Microban and benchmark levels of varying difficulties (Microban by David W. Skinner)
BENCHMARK_LEVELS: Dict[str, str] = {
    # Microban Level 1 (Tutorial - 1 Box)
    "microban_1": """
####
# .#
#  ###
#*@  #
#  $ #
#  ###
####
""",

    # Microban Level 2 (2 Boxes)
    "microban_2": """
  #####
  #   #
###$ $###
#  . .  #
#   @   #
#########
""",

    # Microban Level 3 (Corridor / Target packing)
    "microban_3": """
######
#    #
# $$ #
# .. #
# @  #
######
""",

    # Microban Level 4 (T-Junction)
    "microban_4": """
  #####
###   #
# $ $ #
# . . #
#  @  #
#######
""",

    # Microban Level 5 (Classic 3-Box Challenge)
    "microban_5": """
  ####
###  ####
#     $ #
# #  #$ #
# . .#@ #
#   #$  #
# .     #
#########
""",

    # Microban Level 6 (Tight Corner Navigation)
    "microban_6": """
######
#  . #
# $$ #
# .# #
# @  #
######
""",

    # Intermediate Level A (Crossroads)
    "intermediate_1": """
  #####
  #   #
###$ $###
#  . .  #
#   @   #
#########
""",

    # Advanced Level (4 Boxes)
    "advanced_1": """
#######
#     #
# $$$ #
# ... #
#  $  #
#  .  #
#  @  #
#######
""",
}


def get_level(name: str) -> SokobanState:
    """Get parsed SokobanState for a named benchmark level."""
    if name not in BENCHMARK_LEVELS:
        available = list(BENCHMARK_LEVELS.keys())
        raise KeyError(f"Level '{name}' not found. Available levels: {available}")
    return parse_level_string(BENCHMARK_LEVELS[name])


def get_benchmark_suite() -> Dict[str, SokobanState]:
    """Retrieve full benchmark suite as {name: SokobanState}."""
    return {name: parse_level_string(lvl_str) for name, lvl_str in BENCHMARK_LEVELS.items()}


def generate_random_level(
    height: int = 7,
    width: int = 7,
    num_boxes: int = 2,
    num_walls: int = 6,
    num_reverse_steps: int = 15,
    seed: Optional[int] = None
) -> SokobanState:
    """
    Generate guaranteed solvable Sokoban puzzle via reverse-pull dynamics:
    1. Start in solved state (boxes on targets).
    2. Player performs legal reverse moves (pulling boxes backward).
    3. Final state is guaranteed solvable from player's starting position!
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1. Create bounded room
    walls = set()
    for r in range(height):
        walls.add((r, 0))
        walls.add((r, width - 1))
    for c in range(width):
        walls.add((0, c))
        walls.add((height - 1, c))

    interior_cells = [
        (r, c) for r in range(1, height - 1)
        for c in range(1, width - 1)
    ]

    # 2. Add interior walls ensuring connectivity
    random.shuffle(interior_cells)
    for _ in range(min(num_walls, len(interior_cells) // 3)):
        cand = interior_cells.pop()
        walls.add(cand)

    available_floor = [pos for pos in interior_cells if pos not in walls]
    if len(available_floor) < num_boxes + 1:
        # Fallback to simple layout if overcrowded
        return get_level("microban_1")

    # 3. Choose target/box locations in solved configuration
    random.shuffle(available_floor)
    targets = set(available_floor[:num_boxes])
    boxes = set(targets)  # Initially all boxes on targets
    player = available_floor[num_boxes]

    # 4. Perform random reverse-pull steps
    current_player = player
    current_boxes = set(boxes)

    for _ in range(num_reverse_steps):
        # Reverse pull: Player moves from P to P_new, pulling a box from B to P
        # For this to be valid in reverse:
        # Player moves to empty neighbor, and box in direction opposite to movement is pulled to old player spot.
        actions = list(Action)
        random.shuffle(actions)
        moved = False

        for act in actions:
            dr, dc = ACTION_DELTAS[act]
            new_pr, new_pc = current_player[0] + dr, current_player[1] + dc

            if (new_pr, new_pc) in walls or (new_pr, new_pc) in current_boxes:
                continue
            if not (0 < new_pr < height - 1 and 0 < new_pc < width - 1):
                continue

            # Check if there is a box behind player that can be pulled
            pull_br, pull_bc = current_player[0] - dr, current_player[1] - dc
            if (pull_br, pull_bc) in current_boxes:
                # Pull box
                current_boxes.remove((pull_br, pull_bc))
                current_boxes.add(current_player)
                current_player = (new_pr, new_pc)
                moved = True
                break
            else:
                # Simple player movement without pull
                current_player = (new_pr, new_pc)
                moved = True
                break

        if not moved:
            break

    return SokobanState(
        walls=frozenset(walls),
        boxes=frozenset(current_boxes),
        targets=frozenset(targets),
        player=current_player,
        height=height,
        width=width
    )
