from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def visible_points(
    agent_state: List[float],
    points: List[Dict],
    fov_radius: float,
    fov_angle: float,
    normal_threshold: float,
) -> List[Dict]:
    pos = np.array(agent_state[0:2])
    heading = agent_state[2]
    visible = []
    for point in points:
        point_pos = point["pos"]
        delta = point_pos - pos
        distance = np.linalg.norm(delta)
        if distance > fov_radius:
            continue
        angle_to_point = math.atan2(delta[1], delta[0])
        angle_diff = _wrap_angle(angle_to_point - heading)
        if abs(angle_diff) > fov_angle / 2.0:
            continue
        normal = point.get("normal")
        if normal is not None:
            point_to_agent = pos - point_pos
            point_to_agent_norm = np.linalg.norm(point_to_agent)
            if point_to_agent_norm > 1e-6:
                point_to_agent = point_to_agent / point_to_agent_norm
                if float(np.dot(np.array(normal), point_to_agent)) <= normal_threshold:
                    continue
        visible.append(point)
    return visible

