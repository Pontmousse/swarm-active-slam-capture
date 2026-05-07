from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model import Agent, ExperimentConfig


def _pd_to_target(
    agent_state: List[float],
    target_pos: np.ndarray,
    kp: float,
    kd: float,
) -> np.ndarray:
    pos = np.array(agent_state[0:2])
    vel = np.array(agent_state[3:5])
    error = target_pos - pos
    return kp * error - kd * vel


def search_controller(
    agent: Agent,
    target_center: np.ndarray,
    config: ExperimentConfig,
) -> Tuple[np.ndarray, float]:
    kp = float(getattr(config, "characterize_gain", 1.0))
    force = _pd_to_target(agent.state, target_center, kp, config.damping_gain)
    torque = -0.5 * agent.state[5]
    return force, torque


def encapsulate_controller(
    agent: Agent,
    target_center: np.ndarray,
    config: ExperimentConfig,
) -> Tuple[np.ndarray, float]:
    kp = float(getattr(config, "characterize_gain", 1.0))
    force = _pd_to_target(agent.state, target_center, kp, config.damping_gain)
    torque = -0.5 * agent.state[5]
    return force, torque


def capture_controller(
    agent: Agent,
    target_ap_pos: Optional[np.ndarray],
    config: ExperimentConfig,
) -> Tuple[np.ndarray, float]:
    if target_ap_pos is None:
        return np.zeros(2), 0.0
    force = _pd_to_target(agent.state, target_ap_pos, config.capture_gain, config.damping_gain)
    torque = -0.5 * agent.state[5]
    return force, torque


def capture_pid_controller(
    agent: Agent,
    ap_world: Optional[Dict[str, Any]],
    config: ExperimentConfig,
    aggressiveness: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """
    2D analogue of Capture_PID: F_p + F_v + optional F_n (normal alignment).
    ap_world: dict with keys pos (2,), vel (2,), optional normal (2,) in world frame.
    aggressiveness scales Kp/Kd/Kn within [0, 1].
    """
    if ap_world is None:
        return np.zeros(2), 0.0

    rp = np.asarray(ap_world["pos"], dtype=float).reshape(2)
    rpdot = np.asarray(ap_world["vel"], dtype=float).reshape(2)
    r = np.array(agent.state[0:2], dtype=float)
    rdot = np.array(agent.state[3:5], dtype=float)

    scale = max(0.05, min(1.0, float(aggressiveness)))
    kp = float(config.capture_gain) * scale
    kd = float(config.damping_gain) * scale

    # F_p = -Kp * (r - rp)  == Kp * (rp - r)
    f_p = -kp * (r - rp)
    # F_v = -Kd * (rdot - rpdot)
    f_v = -kd * (rdot - rpdot)

    f_n = np.zeros(2, dtype=float)
    kn = float(getattr(config, "capture_alignment_gain", 0.0)) * scale
    n_raw = ap_world.get("normal")
    if kn > 0.0 and n_raw is not None:
        n = np.asarray(n_raw, dtype=float).reshape(2)
        nn = float(np.linalg.norm(n))
        if nn >= 1e-9:
            n = n / nn
            diff = r - rp
            dist = float(np.linalg.norm(diff))
            if dist >= 1e-9:
                dhat = diff / dist
                mag = 1.0 - float(np.dot(n, dhat))
                # component of dhat orthogonal to n (3D Capture_PID construction)
                force_dir = dhat - float(np.dot(dhat, n)) * n
                f_n = -kn * mag * force_dir

    force = f_p + f_v + f_n
    omega = float(agent.state[5])
    torque = -float(config.hold_torque_gain) * 0.5 * omega
    return force, torque


def docked_controller() -> Tuple[np.ndarray, float]:
    return np.zeros(2), 0.0


# Phase 1 primitives and swarm shell live in swarm_forces (re-export for convenience).
from .swarm_forces import (  # noqa: E402
    clamp_total_force,
    compose_behavior_control,
    compose_phase1_control,
    hold_controller,
    resolve_follow_goal,
    search_bounce_controller,
    swarm_shell_for_legacy,
)

