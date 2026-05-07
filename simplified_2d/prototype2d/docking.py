"""Dock feasibility gates: simulator authority over transitioning to mode \"d\"."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .model import Agent, ExperimentConfig

_EPS = 1e-9


def _unit2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < _EPS:
        return np.array([1.0, 0.0])
    return v / n


def can_dock(
    agent: Agent,
    ap_record: Dict,
    config: ExperimentConfig,
) -> bool:
    """
    Gates for committing to dock (mode \"d\"):
    - distance(agent, AP) <= dock_distance
    - ||v_agent - v_ap|| <= dock_max_rel_speed
    - Approach velocity aligns with inward normal: dot(v_hat, -n_ap) >= dock_heading_dot_threshold
      where v_hat is unit relative velocity (agent - AP).
    """
    pos_a = np.array(agent.state[0:2], dtype=float)
    vel_a = np.array(agent.state[3:5], dtype=float)
    rp = np.asarray(ap_record["pos"], dtype=float).reshape(2)
    rpdot = np.asarray(ap_record["vel"], dtype=float).reshape(2)

    dist = float(np.linalg.norm(pos_a - rp))
    if dist > float(config.dock_distance):
        return False

    v_rel = vel_a - rpdot
    rel_speed = float(np.linalg.norm(v_rel))
    if rel_speed > float(config.dock_max_rel_speed):
        return False

    n = ap_record.get("normal")
    if n is None:
        return True

    n_ap = np.asarray(n, dtype=float).reshape(2)
    nn = float(np.linalg.norm(n_ap))
    if nn < _EPS:
        return True
    n_ap = n_ap / nn

    thr = float(config.dock_heading_dot_threshold)

    # Prefer velocity-based alignment: moving toward the AP along the inward normal (-n_ap).
    if rel_speed >= 0.08:
        v_hat = _unit2(v_rel)
        inward_dot = float(np.dot(v_hat, -n_ap))
        return inward_dot >= thr

    # Nearly static: require straight-line approach from agent to AP to align with inward normal
    to_ap = rp - pos_a
    if float(np.linalg.norm(to_ap)) < _EPS:
        return True
    approach_hat = _unit2(to_ap)
    inward_dot = float(np.dot(approach_hat, -n_ap))
    return inward_dot >= thr
