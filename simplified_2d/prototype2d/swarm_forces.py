from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model import Agent, ExperimentConfig

_EPS = 1e-8


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_pd_torque(agent: Agent, theta_des: Optional[float], config: ExperimentConfig) -> float:
    omega = float(agent.state[5])
    if theta_des is None:
        return -float(config.hold_torque_gain) * 0.5 * omega
    theta = float(agent.state[2])
    e = _wrap_angle(theta_des - theta)
    tau = float(config.pointing_kp) * e - float(config.pointing_kd) * omega
    cap = float(getattr(config, "pointing_tau_cap", 0.0))
    if cap > 0.0:
        tau = max(-cap, min(cap, tau))
    return tau


def _angle_to_world_point(agent: Agent, world_xy: np.ndarray) -> Optional[float]:
    pos = np.array(agent.state[0:2], dtype=float)
    d = world_xy.reshape(2) - pos
    if float(np.linalg.norm(d)) < _EPS:
        return None
    return float(np.arctan2(float(d[1]), float(d[0])))


def _clamp_force(force: np.ndarray, cap: float) -> np.ndarray:
    n = float(np.linalg.norm(force))
    if cap <= 0 or n <= cap or n < _EPS:
        return force
    return force * (cap / n)


def clamp_total_force(force: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    """Clamp combined force magnitude (legacy path after adding swarm shell)."""
    return _clamp_force(force, config.shell_force_cap)


def _dense_positions_by_id(dense_world: List[Dict]) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for pt in dense_world:
        out[int(pt["id"])] = np.asarray(pt["pos"], dtype=float).reshape(2)
    return out


def flocking_force(
    agent: Agent,
    dense_world: List[Dict],
    config: ExperimentConfig,
) -> np.ndarray:
    """Attraction toward visible landmarks (LandSet), 2D analogue of Encapsulate flocking."""
    nu = config.flk_potential
    rflk = config.flk_radius
    pos = np.array(agent.state[0:2], dtype=float)
    x, y = float(pos[0]), float(pos[1])
    id_to_pos = _dense_positions_by_id(dense_world)
    fx = 0.0
    fy = 0.0
    for pid in agent.land_set:
        rp = id_to_pos.get(int(pid))
        if rp is None:
            continue
        xp, yp = float(rp[0]), float(rp[1])
        nr = float(np.linalg.norm(pos - rp))
        denom = max(nr, _EPS)
        numer = nu * (rflk - nr)
        fx += (numer / denom) * (x - xp)
        fy += (numer / denom) * (y - yp)
    return np.array([fx, fy], dtype=float)


def antiflocking_force(
    agent: Agent,
    agents: List[Agent],
    config: ExperimentConfig,
) -> np.ndarray:
    """Repulsion from neighbors in comm_set; 2D analogue of Encapsulate antiflocking."""
    ga = config.antflk_potential
    rant = config.antflk_radius
    pos = np.array(agent.state[0:2], dtype=float)
    x, y = float(pos[0]), float(pos[1])
    fx = 0.0
    fy = 0.0
    for nid in agent.comm_set:
        if nid == agent.id:
            continue
        other = agents[nid]
        ox = float(other.state[0])
        oy = float(other.state[1])
        nr = float(np.linalg.norm(pos - np.array([ox, oy])))
        denom = max(nr ** 3, _EPS)
        numer = ga * (1.0 / max(rant, _EPS) - 1.0 / max(nr, _EPS))
        fx += (numer / denom) * (ox - x)
        fy += (numer / denom) * (oy - y)
    return np.array([fx, fy], dtype=float)


def velocity_damping_force(agent: Agent, config: ExperimentConfig) -> np.ndarray:
    vx = float(agent.state[3])
    vy = float(agent.state[4])
    kd = config.swarm_damping_gain
    return np.array([-kd * vx, -kd * vy], dtype=float)


def target_standoff_force(
    agent: Agent,
    dense_world: List[Dict],
    config: ExperimentConfig,
) -> np.ndarray:
    """Push agent outward if closer than target_standoff_distance to any dense point."""
    d_min = config.target_standoff_distance
    gain = config.target_standoff_gain
    if d_min <= 0 or gain <= 0 or not dense_world:
        return np.zeros(2, dtype=float)
    pos = np.array(agent.state[0:2], dtype=float)
    best_d = float("inf")
    best_delta = np.zeros(2, dtype=float)
    for pt in dense_world:
        p = np.asarray(pt["pos"], dtype=float).reshape(2)
        delta = pos - p
        dist = float(np.linalg.norm(delta))
        if dist < best_d:
            best_d = dist
            if dist > _EPS:
                best_delta = delta
            else:
                best_delta = np.array([_EPS, 0.0])
    if best_d >= d_min:
        return np.zeros(2, dtype=float)
    # Inside standoff shell: push along outward ray from nearest surface point
    n_out = best_delta / max(float(np.linalg.norm(best_delta)), _EPS)
    penetration = d_min - best_d
    return gain * penetration * n_out


def swarm_shell_for_legacy(
    agent: Agent,
    agents: List[Agent],
    dense_world: List[Dict],
    config: ExperimentConfig,
) -> np.ndarray:
    """Flocking + antiflocking + standoff (no extra swarm velocity damping)."""
    return (
        flocking_force(agent, dense_world, config)
        + antiflocking_force(agent, agents, config)
        + target_standoff_force(agent, dense_world, config)
    )


def hold_controller(agent: Agent, config: ExperimentConfig) -> Tuple[np.ndarray, float]:
    vx = float(agent.state[3])
    vy = float(agent.state[4])
    omega = float(agent.state[5])
    kh = config.hold_velocity_gain
    kt = config.hold_torque_gain
    force = np.array([-kh * vx, -kh * vy], dtype=float)
    torque = -kt * omega
    return force, torque


def search_bounce_controller(
    agent: Agent,
    config: ExperimentConfig,
    iteration: int,
) -> Tuple[np.ndarray, float]:
    """Bounded roam: bounce inside search_box with small lateral noise (3D Search analogy)."""
    box = config.search_box
    if len(box) != 4:
        xmin, xmax, ymin, ymax = -3.0, 3.0, -3.0, 3.0
    else:
        xmin, xmax, ymin, ymax = [float(v) for v in box]
    k = config.search_bounce_gain
    x = float(agent.state[0])
    y = float(agent.state[1])

    base_seed = getattr(config, "rng_seed", 0)
    rng = random.Random((base_seed ^ (iteration * 10007) ^ (agent.id * 131)) & ((1 << 63) - 1))

    def rnd_pm() -> float:
        return k / 4.0 * rng.uniform(-1.0, 1.0)

    fx = 0.0
    fy = 0.0
    if x > xmax:
        fx = -k
        fy += rnd_pm() * config.search_lateral_noise
    elif x < xmin:
        fx = k
        fy += rnd_pm() * config.search_lateral_noise
    if y > ymax:
        fy += -k
        fx += rnd_pm() * config.search_lateral_noise
    elif y < ymin:
        fy += k
        fx += rnd_pm() * config.search_lateral_noise

    torque = _yaw_pd_torque(agent, None, config)
    return np.array([fx, fy], dtype=float), torque


def _pd_to_goal(
    agent_state: List[float],
    goal_xy: np.ndarray,
    kp: float,
    kd: float,
) -> np.ndarray:
    pos = np.array(agent_state[0:2], dtype=float)
    vel = np.array(agent_state[3:5], dtype=float)
    error = goal_xy.reshape(2) - pos
    return kp * error - kd * vel


def explore_controller(
    agent: Agent,
    dense_world: List[Dict],
    target_center_xy: np.ndarray,
    config: ExperimentConfig,
) -> Tuple[np.ndarray, float]:
    """PD toward nearest frontier point with outward standoff; fallback CoM or hold."""
    id_to_pos = _dense_positions_by_id(dense_world)
    pos = np.array(agent.state[0:2], dtype=float)
    frontier_ids = agent.map_frontier or []

    if not frontier_ids:
        fb = (config.characterize_fallback or "com").lower()
        if fb == "com":
            goal = np.asarray(target_center_xy, dtype=float).reshape(2)
        else:
            return hold_controller(agent, config)
    else:
        best_pid = None
        best_d = float("inf")
        for pid in frontier_ids:
            p = id_to_pos.get(int(pid))
            if p is None:
                continue
            d = float(np.linalg.norm(pos - p))
            if d < best_d:
                best_d = d
                best_pid = int(pid)
        if best_pid is None:
            fb = (config.characterize_fallback or "com").lower()
            if fb == "com":
                goal = np.asarray(target_center_xy, dtype=float).reshape(2)
            else:
                return hold_controller(agent, config)
        else:
            p = id_to_pos[best_pid]
            delta_out = pos - p
            nd = float(np.linalg.norm(delta_out))
            if nd < _EPS:
                outward = np.array([1.0, 0.0], dtype=float)
            else:
                outward = delta_out / nd
            standoff = float(config.characterize_standoff)
            goal = p + outward * standoff

    kp = float(config.characterize_gain)
    kd = float(config.damping_gain)
    force = _pd_to_goal(agent.state, goal, kp, kd)
    tau_b = _yaw_pd_torque(agent, _angle_to_world_point(agent, np.asarray(goal, dtype=float)), config)
    return force, tau_b


def follow_controller(
    agent: Agent,
    goal_xy: np.ndarray,
    config: ExperimentConfig,
) -> Tuple[np.ndarray, float]:
    kp = float(config.follow_gain)
    kd = float(config.damping_gain)
    force = _pd_to_goal(agent.state, goal_xy, kp, kd)
    tau_b = _yaw_pd_torque(agent, _angle_to_world_point(agent, np.asarray(goal_xy, dtype=float)), config)
    return force, tau_b


def _resolve_search_theta_des(
    agent: Agent,
    agents: List[Agent],
    dense_world: List[Dict],
    config: ExperimentConfig,
    iteration: int,
) -> Optional[float]:
    params = getattr(agent, "behavior_params", {}) or {}
    orient_deg = params.get("target_orientation_deg")
    if orient_deg is not None:
        try:
            return float(np.deg2rad(float(orient_deg)))
        except (TypeError, ValueError):
            pass

    look_world = params.get("look_world_xy")
    if isinstance(look_world, (list, tuple)) and len(look_world) >= 2:
        try:
            world = np.array([float(look_world[0]), float(look_world[1])], dtype=float)
            th = _angle_to_world_point(agent, world)
            if th is not None:
                return th
        except (TypeError, ValueError):
            pass

    # Primary: point to centroid of currently visible points.
    id_to_pos = _dense_positions_by_id(dense_world)
    vis_pts = []
    for pid in agent.land_set:
        p = id_to_pos.get(int(pid))
        if p is not None:
            vis_pts.append(p)
    if vis_pts:
        centroid = np.mean(np.stack(vis_pts, axis=0), axis=0)
        th = _angle_to_world_point(agent, centroid)
        if th is not None:
            return th

    # Fallback: neighbors' navigation hints.
    for msg in sorted(
        agent.last_messages_by_sender.values(),
        key=lambda m: float(m.get("sent_time", -1.0)),
        reverse=True,
    ):
        payload = msg.get("payload") or {}
        hint = payload.get("navigation_hint") or {}
        kind = str(hint.get("kind", "none")).lower()
        if kind == "world_point":
            wx = hint.get("world_xy")
            if isinstance(wx, (list, tuple)) and len(wx) >= 2:
                try:
                    world = np.array([float(wx[0]), float(wx[1])], dtype=float)
                    th = _angle_to_world_point(agent, world)
                    if th is not None:
                        return th
                except (TypeError, ValueError):
                    pass
        elif kind == "point_ids":
            pts = []
            for pid in hint.get("point_ids") or []:
                p = id_to_pos.get(int(pid))
                if p is not None:
                    pts.append(p)
            if pts:
                centroid = np.mean(np.stack(pts, axis=0), axis=0)
                th = _angle_to_world_point(agent, centroid)
                if th is not None:
                    return th

    # Fallback: local consensus heading.
    headings = []
    for nid in agent.comm_set:
        if int(nid) == int(agent.id):
            continue
        if 0 <= int(nid) < len(agents):
            headings.append(float(agents[int(nid)].state[2]))
    if headings:
        s = float(np.sum(np.sin(headings)))
        c = float(np.sum(np.cos(headings)))
        if abs(s) > _EPS or abs(c) > _EPS:
            return float(np.arctan2(s, c))

    # Final deterministic scan fallback.
    base_seed = int(getattr(config, "rng_seed", 0))
    phase = 0.04 * float(iteration) + 0.3 * float(agent.id) + 0.01 * float(base_seed)
    return float(0.5 * np.sin(phase))


def resolve_follow_goal(
    agent: Agent,
    target_center_xy: np.ndarray,
    dense_world: List[Dict],
) -> Optional[np.ndarray]:
    """Resolve navigation_hint from neighbor messages; tie-break by latest sent_time then lowest sender id."""
    best_goal: Optional[np.ndarray] = None
    best_time = -1.0
    best_sender = 10**9
    id_to_pos = _dense_positions_by_id(dense_world)

    for sid, msg in agent.last_messages_by_sender.items():
        payload = msg.get("payload") or {}
        hint = payload.get("navigation_hint") or {}
        kind = (hint.get("kind") or "none").lower()
        if kind == "none":
            continue
        sent_time = float(msg.get("sent_time", -1.0))
        goal: Optional[np.ndarray] = None
        if kind == "target_com":
            goal = np.asarray(target_center_xy, dtype=float).reshape(2).copy()
        elif kind == "world_point":
            wx = hint.get("world_xy")
            if wx is not None and len(wx) >= 2:
                goal = np.array([float(wx[0]), float(wx[1])], dtype=float)
        elif kind == "point_ids":
            raw_ids = hint.get("point_ids") or []
            pts = []
            for pid in raw_ids:
                p = id_to_pos.get(int(pid))
                if p is not None:
                    pts.append(p)
            if pts:
                goal = np.mean(np.stack(pts, axis=0), axis=0)
        if goal is None:
            continue
        if sent_time > best_time or (sent_time == best_time and int(sid) < best_sender):
            best_time = sent_time
            best_sender = int(sid)
            best_goal = goal
    return best_goal


def _find_ap_record(
    attachment_world: Optional[List[Dict[str, Any]]],
    ap_id: Optional[int],
) -> Optional[Dict[str, Any]]:
    if attachment_world is None or ap_id is None:
        return None
    for ap in attachment_world:
        if int(ap["id"]) == int(ap_id):
            return ap
    return None


def compose_behavior_control(
    agent: Agent,
    agents: List[Agent],
    dense_world: List[Dict],
    config: ExperimentConfig,
    iteration: int,
    target_center_xy: np.ndarray,
    attachment_world: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, float]:
    """Motion primitives + always-on swarm shell."""
    raw = (agent.behavior or "search").lower()
    allowed = {"search", "hold", "explore", "characterize", "follow", "capture"}

    if raw not in allowed:
        beh = "search"
    else:
        beh = raw

    agg = float(getattr(agent, "behavior_aggressiveness", 0.5))

    if beh == "hold":
        u_b, tau_b = hold_controller(agent, config)
    elif beh == "explore":
        u_b, tau_b = explore_controller(agent, dense_world, target_center_xy, config)
    elif beh == "characterize":
        # Characterize is an idle/hover shell: no explicit PD goal.
        u_b = np.zeros(2, dtype=float)
        tau_b = _yaw_pd_torque(
            agent,
            _resolve_search_theta_des(agent, agents, dense_world, config, iteration),
            config,
        )
    elif beh == "follow":
        g = resolve_follow_goal(agent, target_center_xy, dense_world)
        if g is None:
            u_b, tau_b = search_bounce_controller(agent, config, iteration)
        else:
            u_b, tau_b = follow_controller(agent, g, config)
    elif beh == "capture":
        ap_rec = _find_ap_record(attachment_world, agent.target_ap)
        if ap_rec is None:
            u_b, tau_b = hold_controller(agent, config)
        else:
            from . import controllers

            u_b, tau_b = controllers.capture_pid_controller(agent, ap_rec, config, agg)
    else:
        u_b, tau_b = search_bounce_controller(agent, config, iteration)
        tau_b = _yaw_pd_torque(
            agent,
            _resolve_search_theta_des(agent, agents, dense_world, config, iteration),
            config,
        )

    # Keep antiflocking always-on (except physical dock mode handled in simulator).
    # Disable flocking while capturing to avoid biasing AP approach.
    if beh == "capture":
        u_flk = np.zeros(2, dtype=float)
    else:
        u_flk = flocking_force(agent, dense_world, config)
    u_ant = antiflocking_force(agent, agents, config)
    u_damp = velocity_damping_force(agent, config)
    u_so = target_standoff_force(agent, dense_world, config)

    force = u_b + u_flk + u_ant + u_damp + u_so
    force = _clamp_force(force, config.shell_force_cap)
    return force, tau_b


def compose_phase1_control(
    agent: Agent,
    agents: List[Agent],
    dense_world: List[Dict],
    config: ExperimentConfig,
    iteration: int,
) -> Tuple[np.ndarray, float]:
    """Behavior primitive + always-on flocking, antiflocking, damping, optional standoff."""
    beh = (agent.behavior or "search").lower()
    if beh == "hold":
        u_b, tau_b = hold_controller(agent, config)
    else:
        u_b, tau_b = search_bounce_controller(agent, config, iteration)

    u_flk = flocking_force(agent, dense_world, config)
    u_ant = antiflocking_force(agent, agents, config)
    u_damp = velocity_damping_force(agent, config)
    u_so = target_standoff_force(agent, dense_world, config)

    force = u_b + u_flk + u_ant + u_damp + u_so
    force = _clamp_force(force, config.shell_force_cap)
    return force, tau_b
