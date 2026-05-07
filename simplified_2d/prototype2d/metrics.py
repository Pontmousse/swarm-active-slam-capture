from __future__ import annotations

from typing import Dict, List, Sequence, Set

import numpy as np

from .frontiers import compute_map_frontier


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points

    pts = sorted(points.tolist())
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(np.array(lower[-2]), np.array(lower[-1]), np.array(p)) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(np.array(upper[-2]), np.array(upper[-1]), np.array(p)) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.array(hull)


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def convex_hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    hull = convex_hull(points)
    return polygon_area(hull)


def convex_hull_perimeter(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    hull = convex_hull(points)
    if len(hull) < 2:
        return 0.0
    hull_c = np.vstack([hull, hull[0]])
    edges = hull_c[1:] - hull_c[:-1]
    return float(np.sum(np.linalg.norm(edges, axis=1)))


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray casting inclusion test for simple closed polygons."""
    if polygon.shape[0] < 3:
        return False
    x = float(point[0])
    y = float(point[1])
    n = int(polygon.shape[0])
    inside = False
    j = n - 1
    eps = 1e-14
    for i in range(n):
        yi = float(polygon[i, 1])
        yj = float(polygon[j, 1])
        if (yi > y) != (yj > y):
            xi = float(polygon[i, 0])
            xj = float(polygon[j, 0])
            cross_x = ((xj - xi) * (y - yi)) / (yj - yi + eps) + xi
            if cross_x > x:
                inside = not inside
        j = i
    return inside


def target_center_inside_agent_hull(agent_positions: np.ndarray, target_xy: np.ndarray) -> float:
    if agent_positions.shape[0] < 3:
        return 0.0
    hull = convex_hull(agent_positions)
    if hull.shape[0] < 3:
        return 0.0
    tgt = np.asarray(target_xy, dtype=float).reshape(2)
    return 1.0 if point_in_polygon(tgt, hull) else 0.0


def min_agent_to_boundary_distance(
    agent_positions: np.ndarray,
    boundary_xy: np.ndarray,
) -> float:
    """Minimum Euclidean distance between any agent and any boundary vertex."""
    if agent_positions.shape[0] == 0 or boundary_xy.shape[0] == 0:
        return float("inf")
    dmin = float("inf")
    for a in agent_positions:
        dv = boundary_xy - a.reshape(1, 2)
        d = float(np.min(np.linalg.norm(dv, axis=1)))
        if d < dmin:
            dmin = d
    return dmin


def oracle_frontier_denominator(
    dense_ordered_ids: Sequence[int],
    dense_adjacency: Dict[int, List[int]],
    dense_id_set: Set[int],
) -> int:
    """
    Stable upper-bound-style reference for frontier size: frontier when exactly one dense
    landmark is still unmapped from the viewpoint of the oracle (dense order minus head).
    """
    order = list(int(x) for x in dense_ordered_ids)
    if len(order) < 2:
        return 1
    partial_known = set(order[1:])
    return max(
        1,
        len(compute_map_frontier(partial_known, dense_adjacency, dense_id_set)),
    )


def compute_frontier_coverage_ratio(union_frontier_count: int, oracle_count: int) -> float:
    if oracle_count <= 0:
        return 0.0
    return float(min(1.0, float(union_frontier_count) / float(oracle_count)))


def distance_stats(points: np.ndarray, target_center: np.ndarray) -> List[float]:
    if len(points) == 0:
        return [0.0, 0.0, 0.0]
    distances = np.linalg.norm(points - target_center, axis=1)
    return [float(np.min(distances)), float(np.mean(distances)), float(np.max(distances))]


def min_inter_agent_distance(points: np.ndarray) -> float:
    """Minimum pairwise distance among agents; inf if fewer than two."""
    n = len(points)
    if n < 2:
        return float("inf")
    min_d = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(points[i] - points[j]))
            if d < min_d:
                min_d = d
    return min_d

