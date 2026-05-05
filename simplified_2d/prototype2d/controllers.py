from __future__ import annotations

import math
from typing import List, Optional, Tuple

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
    force = _pd_to_target(agent.state, target_center, config.search_gain, config.damping_gain)
    torque = -0.5 * agent.state[5]
    return force, torque


def encapsulate_controller(
    agent: Agent,
    target_center: np.ndarray,
    config: ExperimentConfig,
) -> Tuple[np.ndarray, float]:
    force = _pd_to_target(agent.state, target_center, config.encapsulate_gain, config.damping_gain)
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


def docked_controller() -> Tuple[np.ndarray, float]:
    return np.zeros(2), 0.0

