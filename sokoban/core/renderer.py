"""
Renderer utilities for terminal ASCII and colorized visual display.
"""

from typing import Optional, Dict
from sokoban.core.grid import SokobanState, Tile, TILE_TO_CHAR


# ANSI color codes for rich terminal visualization
COLOR_RESET = "\033[0m"
COLOR_WALL = "\033[47;30m"       # White background, black text
COLOR_PLAYER = "\033[1;36m"      # Bright cyan
COLOR_BOX = "\033[1;33m"         # Bright yellow
COLOR_TARGET = "\033[1;31m"      # Bright red
COLOR_BOX_TARGET = "\033[1;32m"  # Bright green
COLOR_DEADLOCK = "\033[41;37m"    # Red background


def render_rich_terminal(state: SokobanState, status_text: Optional[str] = None) -> str:
    """Render state using ANSI colored output."""
    lines = []
    if status_text:
        lines.append(f"\033[1;35m--- {status_text} ---\033[0m")

    for r in range(state.height):
        row = []
        for c in range(state.width):
            pos = (r, c)
            if pos in state.walls:
                row.append(f"{COLOR_WALL} # {COLOR_RESET}")
            elif pos in state.boxes and pos in state.targets:
                row.append(f"{COLOR_BOX_TARGET}[*]{COLOR_RESET}")
            elif pos in state.boxes:
                row.append(f"{COLOR_BOX}[$]{COLOR_RESET}")
            elif pos == state.player and pos in state.targets:
                row.append(f"{COLOR_PLAYER} + {COLOR_RESET}")
            elif pos == state.player:
                row.append(f"{COLOR_PLAYER} @ {COLOR_RESET}")
            elif pos in state.targets:
                row.append(f"{COLOR_TARGET} . {COLOR_RESET}")
            else:
                row.append("   ")
        lines.append("".join(row))

    info = f"\033[1;37mBoxes Placed: {state.num_boxes_on_target()}/{len(state.targets)} | Solved: {state.is_solved()}\033[0m"
    lines.append(info)
    return "\n".join(lines)
