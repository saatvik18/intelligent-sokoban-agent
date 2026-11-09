"""
Grid, coordinate system, state representation, and hashing for Sokoban.
"""

from enum import IntEnum
from typing import Tuple, Set, FrozenSet, Optional, List, Dict
import numpy as np


class Tile(IntEnum):
    EMPTY = 0
    WALL = 1
    BOX = 2
    TARGET = 3
    BOX_ON_TARGET = 4
    PLAYER = 5
    PLAYER_ON_TARGET = 6


CHAR_TO_TILE = {
    ' ': Tile.EMPTY,
    '#': Tile.WALL,
    '$': Tile.BOX,
    '.': Tile.TARGET,
    '*': Tile.BOX_ON_TARGET,
    '@': Tile.PLAYER,
    '+': Tile.PLAYER_ON_TARGET,
}

TILE_TO_CHAR = {
    Tile.EMPTY: ' ',
    Tile.WALL: '#',
    Tile.BOX: '$',
    Tile.TARGET: '.',
    Tile.BOX_ON_TARGET: '*',
    Tile.PLAYER: '@',
    Tile.PLAYER_ON_TARGET: '+',
}


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTION_DELTAS = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

# Global Zobrist table for fast O(1) state hashing
np.random.seed(42)
MAX_GRID_DIM = 32
ZOBRIST_BOX = np.random.randint(0, 2**63 - 1, size=(MAX_GRID_DIM, MAX_GRID_DIM), dtype=np.uint64)
ZOBRIST_PLAYER = np.random.randint(0, 2**63 - 1, size=(MAX_GRID_DIM, MAX_GRID_DIM), dtype=np.uint64)


def flood_fill_reachability(
    player_pos: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    boxes: Set[Tuple[int, int]],
    height: int,
    width: int
) -> Set[Tuple[int, int]]:
    """Compute all reachable floor positions for player without pushing any box."""
    if player_pos in walls or player_pos in boxes:
        return set()
    
    reachable = set()
    queue = [player_pos]
    reachable.add(player_pos)

    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if (nr, nc) not in walls and (nr, nc) not in boxes and (nr, nc) not in reachable:
                    reachable.add((nr, nc))
                    queue.append((nr, nc))
    return reachable


def canonical_state_hash(
    boxes: FrozenSet[Tuple[int, int]],
    player_pos: Tuple[int, int],
    walls: Set[Tuple[int, int]],
    height: int,
    width: int
) -> Tuple[int, Tuple[int, int]]:
    """
    Returns Zobrist hash and canonical player position (min reachable coordinate).
    Two states with identical box positions and player in the same reachable component
    are strategically identical.
    """
    reachable = flood_fill_reachability(player_pos, walls, set(boxes), height, width)
    canonical_player = min(reachable) if reachable else player_pos

    h = np.uint64(0)
    for r, c in boxes:
        if r < MAX_GRID_DIM and c < MAX_GRID_DIM:
            h ^= ZOBRIST_BOX[r, c]
    pr, pc = canonical_player
    if pr < MAX_GRID_DIM and pc < MAX_GRID_DIM:
        h ^= ZOBRIST_PLAYER[pr, pc]

    return int(h), canonical_player


class SokobanState:
    """Immutable, hashable representation of a Sokoban puzzle state."""

    def __init__(
        self,
        walls: FrozenSet[Tuple[int, int]],
        boxes: FrozenSet[Tuple[int, int]],
        targets: FrozenSet[Tuple[int, int]],
        player: Tuple[int, int],
        height: int,
        width: int
    ):
        self.walls: FrozenSet[Tuple[int, int]] = walls
        self.boxes: FrozenSet[Tuple[int, int]] = boxes
        self.targets: FrozenSet[Tuple[int, int]] = targets
        self.player: Tuple[int, int] = player
        self.height: int = height
        self.width: int = width

        self._hash: Optional[int] = None
        self._canonical_player: Optional[Tuple[int, int]] = None
        self._reachable_area: Optional[Set[Tuple[int, int]]] = None

    @property
    def reachable_area(self) -> Set[Tuple[int, int]]:
        if self._reachable_area is None:
            self._reachable_area = flood_fill_reachability(
                self.player, set(self.walls), set(self.boxes), self.height, self.width
            )
        return self._reachable_area

    @property
    def canonical_player(self) -> Tuple[int, int]:
        if self._canonical_player is None:
            self._canonical_player = min(self.reachable_area) if self.reachable_area else self.player
        return self._canonical_player

    def is_solved(self) -> bool:
        """Check if all targets are occupied by boxes."""
        return self.boxes == self.targets

    def num_boxes_on_target(self) -> int:
        """Count number of boxes currently on targets."""
        return len(self.boxes.intersection(self.targets))

    def apply_action(self, action: Action) -> Tuple[Optional['SokobanState'], bool, bool]:
        """
        Apply primitive move action.
        Returns: (new_state, is_valid_move, is_push)
        """
        dr, dc = ACTION_DELTAS[action]
        pr, pc = self.player
        nr, nc = pr + dr, pc + dc

        # Check boundary
        if not (0 <= nr < self.height and 0 <= nc < self.width):
            return None, False, False

        # Hit wall
        if (nr, nc) in self.walls:
            return None, False, False

        # Moving to empty space or target
        if (nr, nc) not in self.boxes:
            new_state = SokobanState(
                walls=self.walls,
                boxes=self.boxes,
                targets=self.targets,
                player=(nr, nc),
                height=self.height,
                width=self.width
            )
            return new_state, True, False

        # Attempting to push box
        nnr, nnc = nr + dr, nc + dc
        if not (0 <= nnr < self.height and 0 <= nnc < self.width):
            return None, False, False

        # Box blocked by wall or another box
        if (nnr, nnc) in self.walls or (nnr, nnc) in self.boxes:
            return None, False, False

        # Valid push
        new_boxes = set(self.boxes)
        new_boxes.remove((nr, nc))
        new_boxes.add((nnr, nnc))

        new_state = SokobanState(
            walls=self.walls,
            boxes=frozenset(new_boxes),
            targets=self.targets,
            player=(nr, nc),
            height=self.height,
            width=self.width
        )
        return new_state, True, True

    def to_feature_tensor(self, deadlock_map: Optional[np.ndarray] = None, fixed_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Extract multi-channel spatial tensor for CNN Perception (C x H x W).
        Channels:
          0: Walls
          1: Boxes (unplaced)
          2: Boxes on target
          3: Empty targets
          4: Player
          5: Reachable floor area
          6: Deadlock danger map (if provided)
        """
        H = fixed_size[0] if fixed_size else self.height
        W = fixed_size[1] if fixed_size else self.width
        C = 7 if deadlock_map is not None else 6

        tensor = np.zeros((C, H, W), dtype=np.float32)

        for r, c in self.walls:
            if r < H and c < W:
                tensor[0, r, c] = 1.0

        for r, c in self.boxes:
            if r < H and c < W:
                if (r, c) in self.targets:
                    tensor[2, r, c] = 1.0  # Box on target
                else:
                    tensor[1, r, c] = 1.0  # Unplaced box

        for r, c in self.targets:
            if r < H and c < W:
                if (r, c) not in self.boxes:
                    tensor[3, r, c] = 1.0  # Empty target

        pr, pc = self.player
        if pr < H and pc < W:
            tensor[4, pr, pc] = 1.0

        for r, c in self.reachable_area:
            if r < H and c < W:
                tensor[5, r, c] = 1.0

        if deadlock_map is not None:
            dh, dw = deadlock_map.shape
            tensor[6, :min(dh, H), :min(dw, W)] = deadlock_map[:min(dh, H), :min(dw, W)]

        return tensor

    def to_ascii(self) -> str:
        """Render board to ASCII string representation."""
        lines = []
        for r in range(self.height):
            row_chars = []
            for c in range(self.width):
                pos = (r, c)
                if pos in self.walls:
                    row_chars.append('#')
                elif pos in self.boxes and pos in self.targets:
                    row_chars.append('*')
                elif pos in self.boxes:
                    row_chars.append('$')
                elif pos == self.player and pos in self.targets:
                    row_chars.append('+')
                elif pos == self.player:
                    row_chars.append('@')
                elif pos in self.targets:
                    row_chars.append('.')
                else:
                    row_chars.append(' ')
            lines.append(''.join(row_chars))
        return '\n'.join(lines)

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self.boxes, self.player))
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SokobanState):
            return False
        return (
            self.boxes == other.boxes
            and self.player == other.player
            and self.targets == other.targets
            and self.walls == other.walls
        )

    def exact_eq(self, other: 'SokobanState') -> bool:
        """Exact position match including non-canonical player position."""
        return (
            self.boxes == other.boxes
            and self.player == other.player
            and self.targets == other.targets
            and self.walls == other.walls
        )


def parse_level_string(level_str: str) -> SokobanState:
    """Parse standard Sokoban ASCII level string into SokobanState."""
    lines = [line.rstrip('\r') for line in level_str.strip('\n').split('\n') if line]
    height = len(lines)
    width = max(len(line) for line in lines)

    walls = set()
    boxes = set()
    targets = set()
    player = None

    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            tile = CHAR_TO_TILE.get(ch, Tile.EMPTY)
            pos = (r, c)
            if tile == Tile.WALL:
                walls.add(pos)
            elif tile == Tile.BOX:
                boxes.add(pos)
            elif tile == Tile.TARGET:
                targets.add(pos)
            elif tile == Tile.BOX_ON_TARGET:
                boxes.add(pos)
                targets.add(pos)
            elif tile == Tile.PLAYER:
                player = pos
            elif tile == Tile.PLAYER_ON_TARGET:
                player = pos
                targets.add(pos)

    if player is None:
        raise ValueError("Sokoban level does not contain a player '@' or '+'!")
    if len(boxes) != len(targets):
        raise ValueError(f"Mismatched boxes ({len(boxes)}) and targets ({len(targets)})!")

    return SokobanState(
        walls=frozenset(walls),
        boxes=frozenset(boxes),
        targets=frozenset(targets),
        player=player,
        height=height,
        width=width
    )


def render_state_ascii(state: SokobanState, title: Optional[str] = None) -> str:
    out = []
    if title:
        out.append(f"--- {title} ---")
    out.append(state.to_ascii())
    out.append(f"Boxes on target: {state.num_boxes_on_target()}/{len(state.targets)} | Solved: {state.is_solved()}")
    return '\n'.join(out)
