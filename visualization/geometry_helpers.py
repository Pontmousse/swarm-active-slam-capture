"""Numpy-only helpers for merged-map matplotlib animation."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

# Lidar boresight in the ray-local frame used by Ray_Cast_Lidar.py: center ray is +Y,
# then flipped (line 29) before body rotation — effective boresight is body -Y.
# Controllers align attitude to Spacecraft["LCD"] (centroid direction) when available.
BODY_LIDAR_BORESIGHT_AXIS = np.array([0.0, -1.0, 0.0], dtype=np.float64)

# Unit-cube corners centered at origin; scaled and rotated by agent pose.
_UNIT_CUBE_VERTICES = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=np.float64,
)

_CUBE_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def load_agents_history(path: str | Path) -> list:
    path = Path(path)
    with path.open("rb") as file:
        return pickle.load(file)


def find_valid_iterations(agents_history: list) -> int:
    """Stop at the first timestep where State_Estim is an empty list."""
    for i, frame in enumerate(agents_history):
        if not frame:
            continue
        state_estim = frame[0].get("State_Estim")
        if isinstance(state_estim, list) and len(state_estim) == 0:
            return i
    return len(agents_history)


def to_points_array(points: Any) -> np.ndarray:
    if points is None:
        return np.array([]).reshape(0, 3)
    arr = np.asarray(points)
    if arr.size == 0:
        return np.array([]).reshape(0, 3)
    return arr.reshape(-1, 3)


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Quaternion order: (qx, qy, qz, qw)."""
    qx, qy, qz, qw = [float(v) for v in quat]
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _pose3_to_position_rotation(state_estim: Any) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import gtsam
    except ImportError:
        return None
    if not isinstance(state_estim, gtsam.Pose3):
        return None
    pos = np.array(state_estim.translation(), dtype=np.float64).reshape(3)
    rot = np.array(state_estim.rotation().matrix(), dtype=np.float64).reshape(3, 3)
    return pos, rot


def agent_position_rotation(agent: dict) -> tuple[np.ndarray, np.ndarray]:
    pose3 = _pose3_to_position_rotation(agent.get("State_Estim"))
    if pose3 is not None:
        return pose3
    state = np.asarray(agent.get("State", []), dtype=np.float64).reshape(-1)
    if state.size < 10:
        raise ValueError("Agent has no usable State_Estim or State vector.")
    pos = state[:3]
    rot = quaternion_to_rotation_matrix(state[6:10])
    return pos, rot


def agent_pointing_direction(agent: dict) -> np.ndarray:
    """Unit vector for attitude arrow: LCD if stored, else lidar boresight (-body Y)."""
    lcd = agent.get("LCD")
    if lcd is not None:
        lcd_vec = np.asarray(lcd, dtype=np.float64).reshape(-1)
        if lcd_vec.size == 3:
            norm = np.linalg.norm(lcd_vec)
            if norm > 1e-12:
                return lcd_vec / norm

    _, rot = agent_position_rotation(agent)
    direction = rot @ BODY_LIDAR_BORESIGHT_AXIS
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return BODY_LIDAR_BORESIGHT_AXIS.copy()
    return direction / norm


def cube_edge_segments(position: np.ndarray, rotation: np.ndarray, cube_size: float) -> list[np.ndarray]:
    """Return list of (2, 3) edge segments for a wireframe cube."""
    verts = (_UNIT_CUBE_VERTICES * float(cube_size)) @ rotation.T + position.reshape(1, 3)
    return [verts[[i, j]] for i, j in _CUBE_EDGES]


def subsample_points(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    points = to_points_array(points)
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


# def get_merged_map_points(agent: dict, max_points: int, frame_idx: int) -> np.ndarray:
#     pts = to_points_array(agent.get("MergedMapSet"))
#     if len(pts) == 0:
#         shared = to_points_array(agent.get("MergedMapSharedSet"))
#         if len(shared) > 0:
#             print(
#                 f"[viz] frame {frame_idx}: MergedMapSet empty; "
#                 "falling back to MergedMapSharedSet"
#             )
#             pts = shared
#     return subsample_points(pts, max_points, seed=frame_idx)


def get_merged_map_points(agent: dict, max_points: int, frame_idx: int) -> np.ndarray:
    pts = to_points_array(agent.get("MergedMapSharedSet"))
    if len(pts) == 0:
        pts = to_points_array(agent.get("MergedMapSet"))
    return subsample_points(pts, max_points, seed=frame_idx)


def compute_axis_limits(
    agents_history: list,
    highlight_agent_1based: int,
    max_map_points_for_limits: int = 12000,
) -> tuple[float, float, float, float, float, float]:
    positions = []
    map_pts = []
    highlight_idx = highlight_agent_1based - 1
    for frame_idx, frame in enumerate(agents_history):
        for agent in frame:
            pos, _ = agent_position_rotation(agent)
            positions.append(pos)
        if 0 <= highlight_idx < len(frame):
            map_pts.append(
                subsample_points(
                    get_merged_map_points(frame[highlight_idx], max_map_points_for_limits, frame_idx),
                    max_map_points_for_limits,
                    seed=frame_idx + 17,
                )
            )

    if not positions:
        return -1.0, 1.0, -1.0, 1.0, -1.0, 1.0

    all_pts = [np.vstack(positions)]
    if map_pts:
        all_pts.append(np.vstack([p for p in map_pts if len(p) > 0]))
    stacked = np.vstack(all_pts)

    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    margin = 0.10 * span
    return (
        float(mins[0] - margin[0]),
        float(maxs[0] + margin[0]),
        float(mins[1] - margin[1]),
        float(maxs[1] + margin[1]),
        float(mins[2] - margin[2]),
        float(maxs[2] + margin[2]),
    )
