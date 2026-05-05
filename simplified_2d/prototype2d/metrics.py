from __future__ import annotations

from typing import List

import numpy as np


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


def distance_stats(points: np.ndarray, target_center: np.ndarray) -> List[float]:
    if len(points) == 0:
        return [0.0, 0.0, 0.0]
    distances = np.linalg.norm(points - target_center, axis=1)
    return [float(np.min(distances)), float(np.mean(distances)), float(np.max(distances))]

