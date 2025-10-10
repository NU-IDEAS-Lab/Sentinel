"""Reusable navigation utilities with both A* and Jump Point Search variants."""
from __future__ import annotations

import math
from itertools import count
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

PositionKey = Tuple[float, float]


STEP_SIZE = 0.25
PRECISION = 2
CARDINAL_DIRECTIONS: Mapping[int, Tuple[float, float]] = {
    0: (0.0, STEP_SIZE),
    90: (STEP_SIZE, 0.0),
    180: (0.0, -STEP_SIZE),
    270: (-STEP_SIZE, 0.0),
}
DIR_TO_YAW = {
    (0, 1): 0,
    (1, 0): 90,
    (0, -1): 180,
    (-1, 0): 270,
}


def make_key(x: float, z: float, precision: int = PRECISION) -> PositionKey:
    """Quantize world coordinates to a grid key."""
    return (round(x, precision), round(z, precision))


def normalize_yaw(yaw: float) -> int:
    """Snap an arbitrary yaw to the nearest 90 degree increment."""
    return int((math.floor((yaw + 45.0) / 90.0) * 90.0) % 360.0)


def build_node_lookup(
    positions: Sequence[Mapping[str, float]], precision: int = PRECISION
) -> Dict[PositionKey, Mapping[str, float]]:
    """Create a grid-aligned lookup table of positions keyed by quantized x/z."""
    return {make_key(pos["x"], pos["z"], precision): pos for pos in positions}


def nearest_key(
    nodes: Mapping[PositionKey, Mapping[str, float]],
    position: Mapping[str, float],
    precision: int = PRECISION,
) -> PositionKey:
    """Return an existing node key closest to the provided position."""
    key = make_key(position["x"], position["z"], precision)
    if key in nodes:
        return key
    return min(
        nodes,
        key=lambda k: (nodes[k]["x"] - position["x"]) ** 2
        + (nodes[k]["z"] - position["z"]) ** 2,
    )


def build_adjacency(
    nodes: Mapping[PositionKey, Mapping[str, float]],
    step_size: float = STEP_SIZE,
    precision: int = PRECISION,
) -> Dict[PositionKey, List[PositionKey]]:
    """Construct a cardinal-adjacency list for the reachable grid."""
    adjacency: Dict[PositionKey, List[PositionKey]] = {key: [] for key in nodes}
    for key, position in nodes.items():
        for dx, dz in CARDINAL_DIRECTIONS.values():
            neighbor_key = make_key(position["x"] + dx, position["z"] + dz, precision)
            if neighbor_key in nodes:
                adjacency[key].append(neighbor_key)
    return adjacency


def movement_cost(
    nodes: Mapping[PositionKey, Mapping[str, float]],
    a: PositionKey,
    b: PositionKey,
) -> float:
    """Return the number of grid steps between two cardinally-connected nodes."""
    dx = abs(nodes[a]["x"] - nodes[b]["x"])
    dz = abs(nodes[a]["z"] - nodes[b]["z"])
    return (dx + dz) / STEP_SIZE


def astar_actions(
    nodes: Mapping[PositionKey, Mapping[str, float]],
    adjacency: Mapping[PositionKey, Iterable[PositionKey]],
    start_key: PositionKey,
    start_yaw: int,
    goal_key: PositionKey,
    step_size: float = STEP_SIZE,
    stats: Optional[MutableMapping[str, int]] = None,
) -> List[str]:
    """Run A* over grid/yaw states and return navigation actions."""
    if start_key == goal_key:
        return []

    import heapq

    def heuristic(key: PositionKey) -> float:
        pos = nodes[key]
        goal = nodes[goal_key]
        return math.hypot(pos["x"] - goal["x"], pos["z"] - goal["z"]) / step_size

    counter = count()
    start_state = (start_key, start_yaw)
    open_heap: List[
        Tuple[float, float, int, Tuple[PositionKey, int]]
    ] = []
    heapq.heappush(open_heap, (heuristic(start_key), 0.0, next(counter), start_state))

    came_from: Dict[Tuple[PositionKey, int], Tuple[Tuple[PositionKey, int], str]] = {}
    g_score: Dict[Tuple[PositionKey, int], float] = {start_state: 0.0}

    if stats is not None:
        stats.setdefault("expanded", 0)
        stats.setdefault("generated", 1)  # starting state

    while open_heap:
        _, current_g, _, current_state = heapq.heappop(open_heap)
        current_pos, current_yaw = current_state

        if stats is not None:
            stats["expanded"] += 1

        if current_pos == goal_key:
            actions: List[str] = []
            while current_state in came_from:
                prev_state, action = came_from[current_state]
                actions.append(action)
                current_state = prev_state
            return list(reversed(actions))

        # Rotate left
        left_state = (current_pos, (current_yaw - 90) % 360)
        tentative_g = current_g + 1.0
        if tentative_g < g_score.get(left_state, float("inf")):
            g_score[left_state] = tentative_g
            came_from[left_state] = (current_state, "RotateLeft")
            heapq.heappush(
                open_heap,
                (tentative_g + heuristic(current_pos), tentative_g, next(counter), left_state),
            )
            if stats is not None:
                stats["generated"] += 1

        # Rotate right
        right_state = (current_pos, (current_yaw + 90) % 360)
        tentative_g = current_g + 1.0
        if tentative_g < g_score.get(right_state, float("inf")):
            g_score[right_state] = tentative_g
            came_from[right_state] = (current_state, "RotateRight")
            heapq.heappush(
                open_heap,
                (tentative_g + heuristic(current_pos), tentative_g, next(counter), right_state),
            )
            if stats is not None:
                stats["generated"] += 1

        # Move ahead
        forward_dx, forward_dz = CARDINAL_DIRECTIONS[current_yaw]
        neighbor_key = make_key(
            nodes[current_pos]["x"] + forward_dx,
            nodes[current_pos]["z"] + forward_dz,
        )
        if neighbor_key in adjacency[current_pos]:
            neighbor_state = (neighbor_key, current_yaw)
            tentative_g = current_g + 1.0
            if tentative_g < g_score.get(neighbor_state, float("inf")):
                g_score[neighbor_state] = tentative_g
                came_from[neighbor_state] = (current_state, "MoveAhead")
                heapq.heappush(
                    open_heap,
                    (
                        tentative_g + heuristic(neighbor_key),
                        tentative_g,
                        next(counter),
                        neighbor_state,
                    ),
                )
                if stats is not None:
                    stats["generated"] += 1

    raise RuntimeError("No path found to target")


def _rotation_actions(current_yaw: int, target_yaw: int) -> List[str]:
    diff = (target_yaw - current_yaw) % 360
    if diff == 0:
        return []
    if diff == 90:
        return ["RotateRight"]
    if diff == 180:
        return ["RotateRight", "RotateRight"]
    if diff == 270:
        return ["RotateLeft"]
    if diff < 180:
        quarter_turns = int(round(diff / 90.0))
        return ["RotateRight"] * quarter_turns
    quarter_turns = int(round((360.0 - diff) / 90.0))
    return ["RotateLeft"] * quarter_turns


def _path_to_actions(
    nodes: Mapping[PositionKey, Mapping[str, float]],
    path: Sequence[PositionKey],
    start_yaw: int,
) -> List[str]:
    if len(path) < 2:
        return []

    actions: List[str] = []
    yaw = start_yaw
    for prev_key, next_key in zip(path, path[1:]):
        prev_node = nodes[prev_key]
        next_node = nodes[next_key]
        dx = next_node["x"] - prev_node["x"]
        dz = next_node["z"] - prev_node["z"]
        if abs(dx) > 1e-6 and abs(dz) > 1e-6:
            raise ValueError("Diagonal movement is not supported")

        if abs(dx) > 1e-6:
            unit_dir = (1 if dx > 0 else -1, 0)
            steps = int(round(abs(dx) / STEP_SIZE))
        else:
            unit_dir = (0, 1 if dz > 0 else -1)
            steps = int(round(abs(dz) / STEP_SIZE))

        target_yaw = DIR_TO_YAW[unit_dir]
        actions.extend(_rotation_actions(yaw, target_yaw))
        yaw = target_yaw
        actions.extend(["MoveAhead"] * steps)
    return actions


def _is_walkable(
    nodes: Mapping[PositionKey, Mapping[str, float]],
    x: float,
    z: float,
    precision: int = PRECISION,
) -> bool:
    return make_key(x, z, precision) in nodes


def _forced_neighbor_exists(
    nodes: Mapping[PositionKey, Mapping[str, float]],
    key: PositionKey,
    direction: Tuple[int, int],
    precision: int = PRECISION,
) -> bool:
    dx, dz = direction
    node = nodes[key]
    x, z = node["x"], node["z"]
    if dx != 0:
        up_blocked = not _is_walkable(nodes, x, z + STEP_SIZE, precision)
        down_blocked = not _is_walkable(nodes, x, z - STEP_SIZE, precision)
        if up_blocked and _is_walkable(nodes, x + dx * STEP_SIZE, z + STEP_SIZE, precision):
            return True
        if down_blocked and _is_walkable(nodes, x + dx * STEP_SIZE, z - STEP_SIZE, precision):
            return True
    else:
        left_blocked = not _is_walkable(nodes, x + STEP_SIZE, z, precision)
        right_blocked = not _is_walkable(nodes, x - STEP_SIZE, z, precision)
        if left_blocked and _is_walkable(nodes, x + STEP_SIZE, z + dz * STEP_SIZE, precision):
            return True
        if right_blocked and _is_walkable(nodes, x - STEP_SIZE, z + dz * STEP_SIZE, precision):
            return True
    return False


def _jump(
    nodes: Mapping[PositionKey, Mapping[str, float]],
    current: PositionKey,
    direction: Tuple[int, int],
    goal: PositionKey,
    precision: int = PRECISION,
) -> Optional[PositionKey]:
    dx, dz = direction
    node = nodes[current]
    next_x = node["x"] + dx * STEP_SIZE
    next_z = node["z"] + dz * STEP_SIZE
    next_key = make_key(next_x, next_z, precision)
    if next_key not in nodes:
        return None
    if next_key == goal:
        return next_key
    if _forced_neighbor_exists(nodes, next_key, direction, precision):
        return next_key
    return _jump(nodes, next_key, direction, goal, precision)


def _pruned_directions(
    nodes: Mapping[PositionKey, Mapping[str, float]],
    current: PositionKey,
    parent: Optional[PositionKey],
    precision: int = PRECISION,
) -> List[Tuple[int, int]]:
    node = nodes[current]
    x, z = node["x"], node["z"]
    directions: List[Tuple[int, int]] = []

    if parent is None:
        for dx, dz in CARDINAL_DIRECTIONS.values():
            neighbor_key = make_key(x + dx, z + dz, precision)
            if neighbor_key in nodes:
                directions.append(
                    (
                        int(round(dx / STEP_SIZE)),
                        int(round(dz / STEP_SIZE)),
                    )
                )
        return directions

    parent_node = nodes[parent]
    diff_x = x - parent_node["x"]
    diff_z = z - parent_node["z"]
    dir_x = int(round(diff_x / STEP_SIZE)) if abs(diff_x) > 1e-6 else 0
    dir_z = int(round(diff_z / STEP_SIZE)) if abs(diff_z) > 1e-6 else 0

    if dir_x != 0:
        if _is_walkable(nodes, x + dir_x * STEP_SIZE, z, precision):
            directions.append((dir_x, 0))
        if not _is_walkable(nodes, x, z + STEP_SIZE, precision) and _is_walkable(
            nodes, x + dir_x * STEP_SIZE, z + STEP_SIZE, precision
        ):
            directions.append((0, 1))
        if not _is_walkable(nodes, x, z - STEP_SIZE, precision) and _is_walkable(
            nodes, x + dir_x * STEP_SIZE, z - STEP_SIZE, precision
        ):
            directions.append((0, -1))
    elif dir_z != 0:
        if _is_walkable(nodes, x, z + dir_z * STEP_SIZE, precision):
            directions.append((0, dir_z))
        if not _is_walkable(nodes, x + STEP_SIZE, z, precision) and _is_walkable(
            nodes, x + STEP_SIZE, z + dir_z * STEP_SIZE, precision
        ):
            directions.append((1, 0))
        if not _is_walkable(nodes, x - STEP_SIZE, z, precision) and _is_walkable(
            nodes, x - STEP_SIZE, z + dir_z * STEP_SIZE, precision
        ):
            directions.append((-1, 0))
    return directions


def _reconstruct_path(
    parents: Mapping[PositionKey, PositionKey],
    start: PositionKey,
    goal: PositionKey,
) -> List[PositionKey]:
    path: List[PositionKey] = [goal]
    current = goal
    while current != start:
        current = parents[current]
        path.append(current)
    path.reverse()
    return path


def jps_actions(
    nodes: Mapping[PositionKey, Mapping[str, float]],
    start_key: PositionKey,
    start_yaw: int,
    goal_key: PositionKey,
    precision: int = PRECISION,
    stats: Optional[MutableMapping[str, int]] = None,
) -> List[str]:
    """Jump Point Search variant returning rotation-aware navigation actions."""
    if start_key == goal_key:
        return []

    import heapq

    def heuristic(key: PositionKey) -> float:
        pos = nodes[key]
        goal = nodes[goal_key]
        return (abs(pos["x"] - goal["x"]) + abs(pos["z"] - goal["z"])) / STEP_SIZE

    counter = count()
    open_heap: List[Tuple[float, float, int, PositionKey]] = []
    heapq.heappush(open_heap, (heuristic(start_key), 0.0, next(counter), start_key))

    parents: Dict[PositionKey, PositionKey] = {}
    g_score: Dict[PositionKey, float] = {start_key: 0.0}
    closed: Set[PositionKey] = set()

    if stats is not None:
        stats.setdefault("expanded", 0)
        stats.setdefault("generated", 1)

    while open_heap:
        _, current_g, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue

        closed.add(current)
        if stats is not None:
            stats["expanded"] += 1

        if current == goal_key:
            path = _reconstruct_path(parents, start_key, goal_key)
            return _path_to_actions(nodes, path, start_yaw)

        parent = parents.get(current)
        directions = _pruned_directions(nodes, current, parent, precision)
        for dir_x, dir_z in directions:
            jump_key = _jump(nodes, current, (dir_x, dir_z), goal_key, precision)
            if jump_key is None:
                continue
            tentative_g = current_g + movement_cost(nodes, current, jump_key)
            if tentative_g + 1e-9 >= g_score.get(jump_key, float("inf")):
                continue
            parents[jump_key] = current
            g_score[jump_key] = tentative_g
            heapq.heappush(
                open_heap,
                (tentative_g + heuristic(jump_key), tentative_g, next(counter), jump_key),
            )
            if stats is not None:
                stats["generated"] += 1

    raise RuntimeError("No path found to target using JPS")


__all__ = [
    "STEP_SIZE",
    "PRECISION",
    "CARDINAL_DIRECTIONS",
    "make_key",
    "normalize_yaw",
    "build_node_lookup",
    "nearest_key",
    "build_adjacency",
    "movement_cost",
    "astar_actions",
    "jps_actions",
]
