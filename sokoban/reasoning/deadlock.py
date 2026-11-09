import functools
from typing import Set, Tuple, Dict, List, Optional, FrozenSet
import numpy as np
from sokoban.core.grid import SokobanState, Action, ACTIONS, ACTION_DELTAS, Tile


@functools.lru_cache(maxsize=256)
def _cached_static_deadlocks(walls: FrozenSet[Tuple[int, int]], targets: FrozenSet[Tuple[int, int]], height: int, width: int) -> FrozenSet[Tuple[int, int]]:
    reachable_pull = set()
    queue: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    for tr, tc in targets:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            pr, pc = tr - dr, tc - dc
            nbr, nbc = tr + dr, tc + dc
            if 0 <= pr < height and 0 <= pc < width and 0 <= nbr < height and 0 <= nbc < width:
                if (pr, pc) not in walls and (nbr, nbc) not in walls:
                    reachable_pull.add((tr, tc))
                    queue.append(((tr, tc), (dr, dc)))

    visited = set(queue)
    while queue:
        (br, bc), (dr, dc) = queue.pop(0)
        reachable_pull.add((br, bc))

        for pdr, pdc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nbr, nbc = br + pdr, bc + pdc
            npr, npc = br + 2 * pdr, bc + 2 * pdc

            if 0 <= nbr < height and 0 <= nbc < width and 0 <= npr < height and 0 <= npc < width:
                if (nbr, nbc) not in walls and (npr, npc) not in walls:
                    item = ((nbr, nbc), (pdr, pdc))
                    if item not in visited:
                        visited.add(item)
                        queue.append(item)

    static_deadlocks = set()
    for r in range(height):
        for c in range(width):
            pos = (r, c)
            if pos not in walls and pos not in reachable_pull and pos not in targets:
                static_deadlocks.add(pos)

    return frozenset(static_deadlocks)


def compute_static_deadlocks(walls: Set[Tuple[int, int]], targets: Set[Tuple[int, int]], height: int, width: int) -> Set[Tuple[int, int]]:
    return set(_cached_static_deadlocks(frozenset(walls), frozenset(targets), height, width))


class DeadlockDetector:
    """
    Comprehensive Deadlock Detector combining static deadlocks and dynamic patterns.
    """

    def __init__(self, walls: Set[Tuple[int, int]], targets: Set[Tuple[int, int]], height: int, width: int):
        self.walls = set(walls)
        self.targets = set(targets)
        self.height = height
        self.width = width
        self.static_deadlocks = compute_static_deadlocks(self.walls, self.targets, self.height, self.width)

    def check_corner_deadlock(self, box: Tuple[int, int]) -> bool:
        """Box is in a corner (two orthogonal adjacent walls) and is not on a target."""
        if box in self.targets:
            return False

        r, c = box
        up = (r - 1, c) in self.walls
        down = (r + 1, c) in self.walls
        left = (r, c - 1) in self.walls
        right = (r, c + 1) in self.walls

        # Corner configurations
        if (up and left) or (up and right) or (down and left) or (down and right):
            return True
        return False

    def check_line_deadlock(self, box: Tuple[int, int], boxes: Set[Tuple[int, int]]) -> bool:
        """Box is against a continuous wall segment with no targets along that segment."""
        if box in self.targets:
            return False

        r, c = box
        # Check horizontal wall above
        for dr in [-1, 1]:
            # Continuous horizontal wall
            if (r + dr, c) in self.walls:
                # Scan left along wall
                sc = c
                left_blocked = False
                while True:
                    if (r, sc) in self.walls:
                        left_blocked = True
                        break
                    if (r + dr, sc) not in self.walls:
                        break  # Wall ended
                    if (r, sc) in self.targets:
                        return False  # Target exists along wall!
                    sc -= 1

                # Scan right along wall
                sc = c
                right_blocked = False
                while True:
                    if (r, sc) in self.walls:
                        right_blocked = True
                        break
                    if (r + dr, sc) not in self.walls:
                        break  # Wall ended
                    if (r, sc) in self.targets:
                        return False  # Target exists along wall!
                    sc += 1

                if left_blocked and right_blocked:
                    return True

        # Check vertical wall left/right
        for dc in [-1, 1]:
            if (r, c + dc) in self.walls:
                # Scan up
                sr = r
                up_blocked = False
                while True:
                    if (sr, c) in self.walls:
                        up_blocked = True
                        break
                    if (sr, c + dc) not in self.walls:
                        break
                    if (sr, c) in self.targets:
                        return False
                    sr -= 1

                # Scan down
                sr = r
                down_blocked = False
                while True:
                    if (sr, c) in self.walls:
                        down_blocked = True
                        break
                    if (sr, c + dc) not in self.walls:
                        break
                    if (sr, c) in self.targets:
                        return False
                    sr += 1

                if up_blocked and down_blocked:
                    return True

        return False

    def check_2x2_square_deadlock(self, boxes: Set[Tuple[int, int]]) -> bool:
        """Any 2x2 square consisting entirely of boxes and walls is a deadlock if not all targets."""
        for br, bc in boxes:
            for dr, dc in [(0, 0), (-1, 0), (0, -1), (-1, -1)]:
                r, c = br + dr, bc + dc
                cells = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]

                all_blocked = True
                box_count = 0
                target_count = 0

                for cr, cc in cells:
                    pos = (cr, cc)
                    if pos in self.walls:
                        continue
                    elif pos in boxes:
                        box_count += 1
                        if pos in self.targets:
                            target_count += 1
                    else:
                        all_blocked = False
                        break

                if all_blocked and box_count > 0 and (box_count != target_count or target_count == 0):
                    return True

        return False

    def is_deadlock(self, state: SokobanState) -> Tuple[bool, Optional[str]]:
        """Evaluate if current state contains any irreversible deadlock."""
        boxes = set(state.boxes)

        # 1. Static deadlocks
        for box in boxes:
            if box in self.static_deadlocks:
                return True, f"Static deadlock at {box}"

        # 2. Corner deadlocks
        for box in boxes:
            if self.check_corner_deadlock(box):
                return True, f"Corner deadlock at {box}"

        # 3. Line/Wall deadlocks
        for box in boxes:
            if self.check_line_deadlock(box, boxes):
                return True, f"Line deadlock at {box}"

        # 4. 2x2 Square deadlocks
        if self.check_2x2_square_deadlock(boxes):
            return True, "2x2 Square deadlock"

        return False, None

    def get_deadlock_map(self) -> np.ndarray:
        """Returns 2D grid array with static deadlock locations marked as 1.0."""
        grid = np.zeros((self.height, self.width), dtype=np.float32)
        for r, c in self.static_deadlocks:
            if r < self.height and c < self.width:
                grid[r, c] = 1.0
        return grid

    def get_valid_action_mask(self, state: SokobanState) -> np.ndarray:
        """
        Compute boolean action mask for DRL & Search:
        mask[a] = True if action is legal and does NOT cause an immediate deadlock.
        """
        mask = np.zeros(4, dtype=bool)

        for act_idx, act in enumerate(ACTIONS):
            next_state, is_valid, is_push = state.apply_action(act)
            if not is_valid or next_state is None:
                mask[act_idx] = False
                continue

            if is_push:
                deadlocked, _ = self.is_deadlock(next_state)
                if deadlocked and not next_state.is_solved():
                    mask[act_idx] = False
                    continue

            mask[act_idx] = True

        # If all actions are deadlocked, keep legal moves to prevent crash
        if not np.any(mask):
            for act_idx, act in enumerate(ACTIONS):
                next_state, is_valid, _ = state.apply_action(act)
                if is_valid:
                    mask[act_idx] = True

        return mask


def is_deadlocked_state(state: SokobanState) -> Tuple[bool, Optional[str]]:
    detector = DeadlockDetector(set(state.walls), set(state.targets), state.height, state.width)
    return detector.is_deadlock(state)


def get_action_deadlock_mask(state: SokobanState) -> np.ndarray:
    detector = DeadlockDetector(set(state.walls), set(state.targets), state.height, state.width)
    return detector.get_valid_action_mask(state)
