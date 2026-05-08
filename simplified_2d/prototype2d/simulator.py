from __future__ import annotations

import os
import pickle
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from . import docking
from .delays import DelayModel
from .io import load_config, load_target_definition, save_json, save_target_definition
from .metrics import (
    convex_hull_area,
    convex_hull_perimeter,
    compute_frontier_coverage_ratio,
    distance_stats,
    min_agent_to_boundary_distance,
    min_inter_agent_distance,
    oracle_frontier_denominator,
    target_center_inside_agent_hull,
)
from .env_loader import maybe_load_dotenv
from . import swarm_forces
from .frontiers import compute_map_frontier
from .behavior_command import BehaviorCommand, validate_and_clamp

_MAX_AGENT_BEHAVIOR_HISTORY = 256


def _append_behavior_history(
    agent: Agent,
    *,
    current_time: float,
    prev_behavior: str,
    cmd: BehaviorCommand,
) -> None:
    rationale = getattr(cmd, "outbound_message", None) or ""
    rationale_preview = rationale[:240] + ("…" if len(rationale) > 240 else "")
    hist = getattr(agent, "behavior_history", None)
    if hist is None:
        agent.behavior_history = []
        hist = agent.behavior_history
    hist.append(
        {
            "time": float(current_time),
            "previous_behavior": prev_behavior,
            "behavior": str(cmd.behavior),
            "target_ap_id": cmd.target_ap_id,
            "rationale_preview": rationale_preview or None,
        }
    )
    if len(hist) > _MAX_AGENT_BEHAVIOR_HISTORY:
        hist[:] = hist[-_MAX_AGENT_BEHAVIOR_HISTORY:]
from .decision import build_decision_snapshot, make_backend
from .messages import export_from_results_dir
from .model import Agent, MetricsSnapshot, TargetDefinition, TargetState
from .perception import visible_points


def _rotation_matrix(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _target_points_world(target_state: List[float], points: List[Dict]) -> List[Dict]:
    pos = np.array(target_state[0:2])
    theta = target_state[2]
    rot = _rotation_matrix(theta)
    omega = target_state[5]
    target_vel = np.array(target_state[3:5])
    world_points = []
    for point in points:
        body = np.array([point["x"], point["y"]])
        world_pos = pos + rot @ body
        vel = target_vel + omega * np.array([-body[1], body[0]])
        normal = None
        if point.get("normal") is not None:
            normal = (rot @ np.array(point["normal"])).tolist()
        world_points.append(
            {
                "id": point["id"],
                "pos": world_pos,
                "vel": vel,
                "normal": normal,
                "body": body,
                "point_id": point.get("point_id"),
            }
        )
    return world_points


def _agent_to_dict(agent: Agent) -> Dict:
    return agent.to_dict()


def _neighbor_ids(agents: List[Agent], agent_id: int, radius: float) -> List[int]:
    origin = np.array(agents[agent_id].state[0:2])
    neighbors = []
    for other in agents:
        if other.id == agent_id:
            continue
        if np.linalg.norm(np.array(other.state[0:2]) - origin) <= radius:
            neighbors.append(other.id)
    return neighbors


def _queue_ready(queue: List[Dict], current_time: float) -> Tuple[List[Dict], List[Dict]]:
    ready = [item for item in queue if item["deliver_time"] <= current_time]
    pending = [item for item in queue if item["deliver_time"] > current_time]
    return ready, pending


def _world_positions_for_point_ids(
    dense_world: List[Dict], point_ids: List[int]
) -> Dict[int, np.ndarray]:
    id_to_pos = {int(pt["id"]): pt["pos"] for pt in dense_world}
    out: Dict[int, np.ndarray] = {}
    for pid in point_ids:
        key = int(pid)
        if key in id_to_pos:
            out[key] = id_to_pos[key]
    return out


def _update_map(
    agent: Agent,
    point_ids: List[int],
    current_time: float,
    positions_by_id: Optional[Dict[int, np.ndarray]] = None,
) -> None:
    for pid in point_ids:
        pid_i = int(pid)
        entry = agent.map.get(pid_i)
        wp = None
        if positions_by_id is not None and pid_i in positions_by_id:
            p = positions_by_id[pid_i]
            wp = [float(p[0]), float(p[1])]
        if entry is None:
            agent.map[pid_i] = {
                "first_seen": current_time,
                "last_seen": current_time,
                "num_observations": 1,
                "last_world_position": wp,
            }
        else:
            entry["last_seen"] = current_time
            entry["num_observations"] += 1
            if wp is not None:
                entry["last_world_position"] = wp


def _build_status_payload(agent: Agent, content: Optional[str]) -> Dict[str, object]:
    return {
        "behavior": str(agent.behavior),
        "target_ap": agent.target_ap,
        "sender_xy": [float(agent.state[0]), float(agent.state[1])],
        "content": content or "",
    }


def _recompute_map_frontiers(
    agents: List[Agent],
    target_def: TargetDefinition,
) -> None:
    if not target_def.dense_point_ids_ordered:
        for agent in agents:
            agent.map_frontier = []
        return
    dense_set: Set[int] = set(target_def.dense_point_ids_ordered)
    adj = target_def.dense_adjacency
    for agent in agents:
        agent.map_frontier = compute_map_frontier(agent.map.keys(), adj, dense_set)


def _distance_to_ap(point_pos: np.ndarray, agent_state: List[float]) -> float:
    return float(np.linalg.norm(point_pos - np.array(agent_state[0:2])))


def _dock_pose(agent: Agent, target_state: List[float]) -> List[float]:
    pos = np.array(agent.state[0:2])
    theta = agent.state[2]
    target_pos = np.array(target_state[0:2])
    target_theta = target_state[2]
    rot = _rotation_matrix(target_theta)
    rel = rot.T @ (pos - target_pos)
    return [float(rel[0]), float(rel[1]), float(theta - target_theta)]


def _apply_docked_state(agent: Agent, target_state: List[float]) -> None:
    if agent.dock_pose is None:
        return
    rel = np.array(agent.dock_pose[0:2])
    rel_theta = agent.dock_pose[2]
    target_pos = np.array(target_state[0:2])
    target_theta = target_state[2]
    target_vel = np.array(target_state[3:5])
    omega = target_state[5]
    rot = _rotation_matrix(target_theta)
    world_pos = target_pos + rot @ rel
    world_vel = target_vel + omega * np.array([-rel[1], rel[0]])
    agent.state[0] = float(world_pos[0])
    agent.state[1] = float(world_pos[1])
    agent.state[2] = float(target_theta + rel_theta)
    agent.state[3] = float(world_vel[0])
    agent.state[4] = float(world_vel[1])
    agent.state[5] = float(omega)


def _advance_target(target: TargetState, dt: float) -> None:
    state = target.state
    state[0] += state[3] * dt
    state[1] += state[4] * dt
    state[2] += state[5] * dt


def _advance_agent(agent: Agent, force: np.ndarray, torque: float, dt: float) -> None:
    ax = force[0] / agent.mass
    ay = force[1] / agent.mass
    alpha = torque / agent.inertia
    agent.state[3] += ax * dt
    agent.state[4] += ay * dt
    agent.state[5] += alpha * dt
    agent.state[0] += agent.state[3] * dt
    agent.state[1] += agent.state[4] * dt
    agent.state[2] += agent.state[5] * dt


def _write_history_snapshots(
    results_dir: str,
    agents_history: List[List[Dict]],
    target_history: List[Dict],
    attachment_history: List[List[Dict]],
    metrics_history: List[Dict],
    messages_history: List[Dict],
    prompt_traces: List[Dict],
    *,
    current_time: Optional[float] = None,
    step: Optional[int] = None,
    total_steps: Optional[int] = None,
    reason: str = "periodic",
) -> None:
    with open(os.path.join(results_dir, "agents_history.pkl"), "wb") as handle:
        pickle.dump(agents_history, handle)
    with open(os.path.join(results_dir, "target_history.pkl"), "wb") as handle:
        pickle.dump(target_history, handle)
    with open(os.path.join(results_dir, "attachment_points.pkl"), "wb") as handle:
        pickle.dump(attachment_history, handle)
    with open(os.path.join(results_dir, "metrics_history.pkl"), "wb") as handle:
        pickle.dump(metrics_history, handle)
    with open(os.path.join(results_dir, "messages_history.pkl"), "wb") as handle:
        pickle.dump(messages_history, handle)
    with open(os.path.join(results_dir, "prompt_traces.pkl"), "wb") as handle:
        pickle.dump(prompt_traces, handle)
    if current_time is not None and step is not None and total_steps is not None:
        print(
            f"[snap] {reason} t={current_time:5.2f}s step={step:4d}/{total_steps} "
            f"rows={len(metrics_history)}",
            flush=True,
        )
    else:
        print(
            f"[snap] {reason} wrote snapshot files rows={len(metrics_history)}",
            flush=True,
        )


def run_simulation(config_path: str) -> str:
    maybe_load_dotenv()
    config = load_config(config_path)
    target_def = load_target_definition(
        config.target_json_path,
        dense_boundary_closed=config.dense_boundary_closed,
    )

    target_state = TargetState(
        state=list(config.initial_target_state),
        mass=config.target_mass,
        inertia=config.target_inertia,
    )

    agents: List[Agent] = []
    for idx in range(config.num_agents):
        state = list(config.initial_agent_states[idx])
        agents.append(
            Agent(
                id=idx,
                state=state,
                mode="p",
                mass=config.agent_mass,
                inertia=config.agent_inertia,
                action_time=0.0,
            )
        )

    results_dir = os.path.join(config.output_root, config.name)
    os.makedirs(results_dir, exist_ok=True)
    save_json(os.path.join(results_dir, "config.json"), config.to_dict())
    save_target_definition(os.path.join(results_dir, "target.json"), target_def)

    rng_seed_int = int(getattr(config, "rng_seed", 0))
    perception_delay_model = DelayModel(
        config.perception_delay,
        rng=np.random.default_rng(rng_seed_int),
    )
    communication_delay_model = DelayModel(
        config.communication_delay,
        rng=np.random.default_rng(rng_seed_int + 137),
    )
    actuation_delay_model = DelayModel(
        config.actuation_delay,
        rng=np.random.default_rng(rng_seed_int + 239),
    )

    dense_id_set: Set[int] = {int(pt.id) for pt in target_def.dense_points}
    oracle_frontier_count = oracle_frontier_denominator(
        target_def.dense_point_ids_ordered,
        target_def.dense_adjacency,
        dense_id_set,
    )

    mode_sec_by_agent: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    behavior_sec_by_agent: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    integrated_control_effort_sum = 0.0
    total_llm_prompt_tokens = 0
    total_llm_completion_tokens = 0

    agents_history: List[List[Dict]] = []
    target_history: List[Dict] = []
    attachment_history: List[List[Dict]] = []
    metrics_history: List[Dict] = []
    messages_history: List[Dict] = []
    prompt_traces: List[Dict] = []
    time_to_all_docked: Optional[float] = None

    perception_queue: List[Dict] = []
    message_queue: List[Dict] = []
    actuation_queue: List[Dict] = []
    last_applied: Dict[int, Tuple[np.ndarray, float]] = {
        agent.id: (np.zeros(2), 0.0) for agent in agents
    }

    decision_backend_impl = make_backend(config)

    total_decision_calls = 0
    total_decision_invalid = 0
    total_decision_latency_sec = 0.0
    total_llm_calls = 0
    total_steps = int(np.floor(config.duration / config.dt))
    periodic_snapshot_writes = 10
    snapshot_interval_steps = max(1, total_steps // periodic_snapshot_writes)
    # log_interval_steps = max(1, int(round(1.0 / max(config.dt, 1e-6))))
    log_interval_steps = 10 * config.dt
    print(
        f"[sim] {config.name} | duration={config.duration}s dt={config.dt} steps={total_steps}",
        flush=True,
    )
    dense_points = [
        {"id": pt.id, "x": pt.x, "y": pt.y, "normal": pt.normal}
        for pt in target_def.dense_points
    ]
    for step in range(total_steps + 1):
        current_time = step * config.dt

        target_center = np.array(target_state.state[0:2])
        attachment_points = [
            {
                "id": ap.id,
                "point_id": ap.point_id,
                "x": ap.x,
                "y": ap.y,
                "normal": ap.normal,
            }
            for ap in target_def.attachment_points
        ]
        attachment_world = _target_points_world(target_state.state, attachment_points)
        dense_world = _target_points_world(target_state.state, dense_points)

        for agent in agents:
            agent.inbox = []
            agent.land_set = []

        message_ages = []
        ready_perception, perception_queue = _queue_ready(perception_queue, current_time)
        for item in ready_perception:
            agent = agents[item["agent_id"]]
            pos_map_a = _world_positions_for_point_ids(dense_world, item["point_ids"])
            _update_map(agent, item["point_ids"], current_time, pos_map_a)
            agent.land_set.extend(item["point_ids"])

        ready_messages, message_queue = _queue_ready(message_queue, current_time)
        for message in ready_messages:
            recipient = agents[message["recipient_id"]]
            recipient.inbox.append(message)
            recipient.last_messages_by_sender[message["sender_id"]] = message
            message_ages.append(current_time - message["sent_time"])
            messages_history.append(message)

        for agent in agents:
            if agent.mode != "d":
                agent.mode = "p"
            agent.comm_set = _neighbor_ids(agents, agent.id, config.communication_radius)

            visible = visible_points(
                agent.state,
                dense_world,
                config.fov_radius,
                config.fov_angle,
                config.normal_visibility_threshold,
            )
            visible_ids = [pt["id"] for pt in visible]
            perception_delay = perception_delay_model.sample(agent.id)
            perception_queue.append(
                {
                    "agent_id": agent.id,
                    "point_ids": visible_ids,
                    "sent_time": current_time,
                    "deliver_time": current_time + perception_delay,
                }
            )

        ready_perception, perception_queue = _queue_ready(perception_queue, current_time)
        for item in ready_perception:
            agent = agents[item["agent_id"]]
            pos_map = _world_positions_for_point_ids(dense_world, item["point_ids"])
            _update_map(agent, item["point_ids"], current_time, pos_map)
            agent.land_set.extend(item["point_ids"])

        _recompute_map_frontiers(agents, target_def)

        decision_calls_step = 0
        decision_invalid_step = 0
        decision_latency_sum_step = 0.0
        llm_calls_step = 0
        llm_prompt_tokens_step = 0
        llm_completion_tokens_step = 0

        period = float(getattr(config, "decision_period", 1.0))
        for agent in agents:
            if (
                agent.last_decision_time >= 0.0
                and current_time - agent.last_decision_time < period
            ):
                continue
            vis_ap_ids: List[int] = []
            for ap in attachment_world:
                d = float(np.linalg.norm(ap["pos"] - np.array(agent.state[0:2])))
                if d <= config.ap_detection_radius:
                    vis_ap_ids.append(int(ap["id"]))
            snap = build_decision_snapshot(
                agent,
                agents,
                config,
                current_time,
                vis_ap_ids,
                target_center,
            )
            t0 = time.perf_counter()
            raw_cmd = decision_backend_impl.decide(agent.id, snap)
            dt_decide = time.perf_counter() - t0
            decision_latency_sum_step += dt_decide
            backend_name = (getattr(config, "decision_backend", "fsm") or "fsm").lower()
            if backend_name == "openai":
                llm_calls_step += 1
                agent.llm_call_count += 1
                usage = getattr(decision_backend_impl, "last_usage", None) or {}
                llm_prompt_tokens_step += int(usage.get("prompt_tokens", 0))
                llm_completion_tokens_step += int(usage.get("completion_tokens", 0))
                last_trace = getattr(decision_backend_impl, "last_trace", None)
                if isinstance(last_trace, dict):
                    trace_row = dict(last_trace)
                    trace_row["time"] = float(current_time)
                    prompt_traces.append(trace_row)
            decision_calls_step += 1
            cmd = validate_and_clamp(raw_cmd, config)
            if cmd is None:
                decision_invalid_step += 1
                cmd = BehaviorCommand(
                    behavior="hold",
                    params={"aggressiveness": 0.5},
                )
            prev_behavior = str(getattr(agent, "behavior", "search"))
            agent.behavior = cmd.behavior
            agent.behavior_aggressiveness = float(
                cmd.params.get("aggressiveness", 0.5)
            )
            agent.behavior_params = dict(cmd.params)
            agent.decision_target_ap_id = cmd.target_ap_id
            agent.outbound_message = cmd.outbound_message
            agent.outbound_channel = str(getattr(cmd, "message_channel", "broadcast"))
            agent.outbound_recipient_id = getattr(cmd, "message_recipient_id", None)
            agent.outbound_reason = getattr(cmd, "ap_reason", None)
            agent.decision_backend_last = str(
                getattr(config, "decision_backend", "fsm")
            )
            agent.last_decision_time = current_time
            _append_behavior_history(
                agent,
                current_time=float(current_time),
                prev_behavior=prev_behavior,
                cmd=cmd,
            )
            agg = float(cmd.params.get("aggressiveness", 0.5))
            print(
                f"[dec] t={current_time:5.2f} a={agent.id} b={backend_name} "
                f"beh={cmd.behavior} ap={cmd.target_ap_id} agg={agg:.2f}",
                flush=True,
            )
            if cmd.behavior != prev_behavior:
                print(
                    f"[chg] t={current_time:5.2f} a={agent.id} {prev_behavior}->{cmd.behavior}",
                    flush=True,
                )

        total_decision_calls += decision_calls_step
        total_decision_invalid += decision_invalid_step
        total_decision_latency_sec += decision_latency_sum_step
        total_llm_calls += llm_calls_step
        total_llm_prompt_tokens += llm_prompt_tokens_step
        total_llm_completion_tokens += llm_completion_tokens_step

        for agent in agents:
            content = (agent.outbound_message or "").strip()
            if not content:
                continue
            payload = _build_status_payload(agent, content)
            ch = str(getattr(agent, "outbound_channel", "broadcast") or "broadcast").lower()
            if ch == "direct":
                rid = getattr(agent, "outbound_recipient_id", None)
                if rid is None or int(rid) not in set(agent.comm_set):
                    continue
                recipients = [int(rid)]
                channel = "direct"
                mtype = "agent_direct"
            else:
                recipients = list(agent.comm_set)
                channel = "broadcast"
                mtype = "agent_broadcast"
            for recipient_id in recipients:
                comm_delay = communication_delay_model.sample(recipient_id)
                message_queue.append(
                    {
                        "message_id": f"{agent.id}-{recipient_id}-{step}",
                        "sent_time": current_time,
                        "deliver_time": current_time + comm_delay,
                        "sender_id": agent.id,
                        "recipient_id": recipient_id,
                        "channel": channel,
                        "message_type": mtype,
                        "payload": payload,
                        "metadata": {},
                    }
                )
            # One-shot send semantics: decision loop explicitly issues each message.
            agent.outbound_message = None
            agent.outbound_recipient_id = None

        ready_messages, message_queue = _queue_ready(message_queue, current_time)
        for message in ready_messages:
            recipient = agents[message["recipient_id"]]
            recipient.inbox.append(message)
            recipient.last_messages_by_sender[message["sender_id"]] = message
            message_ages.append(current_time - message["sent_time"])
            messages_history.append(message)

        for agent in agents:
            bids = []
            aps: List[int] = []
            for ap in attachment_world:
                distance = np.linalg.norm(ap["pos"] - np.array(agent.state[0:2]))
                if distance <= config.ap_detection_radius:
                    aps.append(ap["id"])
                    bids.append(1.0 / max(distance, 1e-3))
            agent.aps = aps
            agent.aps_bids = bids

            if agent.behavior == "capture":
                pref = getattr(agent, "decision_target_ap_id", None)
                if pref is not None and pref in aps:
                    agent.target_ap = int(pref)
                elif pref is None:
                    agent.target_ap = None
                else:
                    # Agent-chosen AP is not currently available/visible.
                    agent.target_ap = None

        ap_conflicts = 0

        for agent in agents:
            # Docking is physical, not a policy behavior:
            # once contact constraints pass, commit to mode "d".
            if agent.mode != "d" and agent.target_ap is not None:
                ap = next((ap for ap in attachment_world if ap["id"] == agent.target_ap), None)
                if ap is not None and docking.can_dock(agent, ap, config):
                    agent.mode = "d"
                    agent.dock_pose = _dock_pose(agent, target_state.state)
                    agent.dock_time = current_time

            if agent.mode == "d":
                agent.control_force = [0.0, 0.0]
                agent.control_torque = 0.0
                continue

            force, torque = swarm_forces.compose_behavior_control(
                agent,
                agents,
                dense_world,
                config,
                step,
                target_center,
                attachment_world=attachment_world,
            )

            actuation_delay = actuation_delay_model.sample(agent.id)
            actuation_queue.append(
                {
                    "agent_id": agent.id,
                    "deliver_time": current_time + actuation_delay,
                    "force": force,
                    "torque": torque,
                }
            )

        ready_actuation, actuation_queue = _queue_ready(actuation_queue, current_time)
        for item in ready_actuation:
            last_applied[item["agent_id"]] = (item["force"], item["torque"])

        for agent in agents:
            if agent.mode == "d":
                continue
            applied_force, applied_torque = last_applied[agent.id]
            agent.control_force = [float(applied_force[0]), float(applied_force[1])]
            agent.control_torque = float(applied_torque)
            fuel_step = config.fuel_coeff * float(np.linalg.norm(applied_force)) * config.dt
            agent.fuel_consumed += fuel_step
            _advance_agent(agent, applied_force, applied_torque, config.dt)

        _advance_target(target_state, config.dt)
        for agent in agents:
            if agent.mode == "d":
                _apply_docked_state(agent, target_state.state)

        agent_positions = np.array([agent.state[0:2] for agent in agents])
        hull_area = convex_hull_area(agent_positions)
        min_d, mean_d, max_d = distance_stats(agent_positions, target_center)
        mode_counts = {"s": 0, "e": 0, "c": 0, "d": 0, "p": 0}
        total_control_effort = 0.0
        total_fuel = 0.0
        behavior_counts: Dict[str, int] = {}
        for agent in agents:
            mode_counts[agent.mode] = mode_counts.get(agent.mode, 0) + 1
            beh = getattr(agent, "behavior", "search")
            behavior_counts[beh] = behavior_counts.get(beh, 0) + 1
            total_control_effort += float(np.linalg.norm(agent.control_force))
            total_fuel += agent.fuel_consumed

        if time_to_all_docked is None and mode_counts.get("d", 0) == len(agents):
            time_to_all_docked = current_time

        map_sizes = [len(agent.map) for agent in agents]
        map_size_mean = float(np.mean(map_sizes)) if map_sizes else 0.0
        global_map = set()
        for agent in agents:
            global_map.update(agent.map.keys())
        total_points = max(len(dense_points), 1)
        map_coverage_ratio = float(len(global_map)) / float(total_points)

        assigned_aps = [agent.target_ap for agent in agents if agent.target_ap is not None]
        unique_assigned = len(set(assigned_aps))
        ap_coverage_ratio = float(unique_assigned) / float(max(len(attachment_world), 1))

        capture_errors = []
        for agent in agents:
            if agent.behavior == "capture" and agent.target_ap is not None:
                ap = next((ap for ap in attachment_world if ap["id"] == agent.target_ap), None)
                if ap is not None:
                    capture_errors.append(_distance_to_ap(ap["pos"], agent.state))
        capture_error_mean = float(np.mean(capture_errors)) if capture_errors else 0.0

        message_age_mean = float(np.mean(message_ages)) if message_ages else 0.0
        message_age_max = float(np.max(message_ages)) if message_ages else 0.0

        pair_d = min_inter_agent_distance(agent_positions)
        if pair_d == float("inf"):
            pair_d = 0.0

        frontier_sizes = [len(agent.map_frontier) for agent in agents]
        frontier_size_mean = float(np.mean(frontier_sizes)) if frontier_sizes else 0.0
        global_frontier_union: Set[int] = set()
        for agent in agents:
            global_frontier_union.update(agent.map_frontier)
        global_frontier_union_count = len(global_frontier_union)

        hull_perimeter = convex_hull_perimeter(agent_positions)
        target_inside_flag = target_center_inside_agent_hull(agent_positions, target_center)
        boundary_xy = (
            np.array([[float(p["pos"][0]), float(p["pos"][1])] for p in dense_world])
            if dense_world
            else np.zeros((0, 2))
        )
        min_clear_target = (
            min_agent_to_boundary_distance(agent_positions, boundary_xy)
            if boundary_xy.shape[0] > 0
            else 0.0
        )
        if min_clear_target == float("inf"):
            min_clear_target = 0.0

        frontier_cov = compute_frontier_coverage_ratio(global_frontier_union_count, oracle_frontier_count)

        dt_step = float(config.dt)
        integrated_control_effort_sum += total_control_effort * dt_step
        for agent in agents:
            mode_sec_by_agent[agent.id][agent.mode] += dt_step
            behavior_sec_by_agent[agent.id][agent.behavior] += dt_step

        metrics_history.append(
            MetricsSnapshot(
                time=current_time,
                convex_hull_area=hull_area,
                min_distance=min_d,
                mean_distance=mean_d,
                max_distance=max_d,
                control_effort=total_control_effort,
                fuel_consumed_total=total_fuel,
                mode_counts=mode_counts,
                message_age_mean=message_age_mean,
                message_age_max=message_age_max,
                map_size_mean=map_size_mean,
                map_coverage_ratio=map_coverage_ratio,
                ap_conflicts=ap_conflicts,
                ap_coverage_ratio=ap_coverage_ratio,
                capture_error_mean=capture_error_mean,
                behavior_counts=behavior_counts,
                min_inter_agent_distance=pair_d,
                frontier_size_mean=frontier_size_mean,
                global_frontier_union_count=global_frontier_union_count,
                decision_calls_step=decision_calls_step,
                decision_invalid_step=decision_invalid_step,
                decision_latency_sum_sec_step=decision_latency_sum_step,
                llm_calls_step=llm_calls_step,
                convex_hull_perimeter=hull_perimeter,
                target_center_inside_hull=target_inside_flag,
                min_agent_boundary_distance=min_clear_target,
                frontier_coverage_ratio=frontier_cov,
                llm_prompt_tokens_step=llm_prompt_tokens_step,
                llm_completion_tokens_step=llm_completion_tokens_step,
            ).to_dict()
        )

        agents_history.append([_agent_to_dict(agent) for agent in agents])
        target_history.append(target_state.to_dict())
        attachment_history.append(
            [
                {
                    "id": ap["id"],
                    "point_id": ap["point_id"],
                    "pos": ap["pos"].tolist(),
                    "normal": ap["normal"],
                }
                for ap in attachment_world
            ]
        )

        for agent in agents:
            agent.time_step = current_time
            agent.iteration = step

        modes_compact = "".join(agent.mode for agent in agents)
        if step == 0 or step == total_steps or (step % log_interval_steps == 0):
            print(
                f"[sim] t={current_time:5.2f}s step={step:4d}/{total_steps} "
                f"m={modes_compact} dec={decision_calls_step} llm={llm_calls_step}",
                flush=True,
            )
        if step > 0 and (
            step == total_steps
            or step % snapshot_interval_steps == 0
        ):
            _write_history_snapshots(
                results_dir,
                agents_history,
                target_history,
                attachment_history,
                metrics_history,
                messages_history,
                prompt_traces,
                current_time=float(current_time),
                step=step,
                total_steps=total_steps,
                reason="periodic",
            )

    _write_history_snapshots(
        results_dir,
        agents_history,
        target_history,
        attachment_history,
        metrics_history,
        messages_history,
        prompt_traces,
        current_time=float(total_steps * config.dt),
        step=total_steps,
        total_steps=total_steps,
        reason="final",
    )

    selected_agent = int(np.random.default_rng(rng_seed_int + 991).choice([a.id for a in agents]))
    generated_docs = export_from_results_dir(results_dir, agent_id=selected_agent)
    for _, out_path in generated_docs.items():
        print(f"[sim] wrote {out_path}", flush=True)

    final_row = metrics_history[-1] if metrics_history else {}
    dwell_modes = {
        str(aid): {m: round(t, 6) for m, t in modes.items()}
        for aid, modes in sorted(mode_sec_by_agent.items())
    }
    dwell_behaviors = {
        str(aid): {b: round(t, 6) for b, t in bh.items()}
        for aid, bh in sorted(behavior_sec_by_agent.items())
    }

    performance = {
        "experiment": config.name,
        "rng_seed": rng_seed_int,
        "time_to_all_docked": time_to_all_docked,
        "final_convex_hull_area": float(final_row.get("convex_hull_area", 0.0)),
        "final_convex_hull_perimeter": float(final_row.get("convex_hull_perimeter", 0.0)),
        "final_target_center_inside_hull": float(final_row.get("target_center_inside_hull", 0.0)),
        "final_map_coverage_ratio": float(final_row.get("map_coverage_ratio", 0.0)),
        "final_frontier_coverage_ratio": float(final_row.get("frontier_coverage_ratio", 0.0)),
        "final_min_agent_boundary_distance": float(final_row.get("min_agent_boundary_distance", 0.0)),
        "total_fuel_consumed": metrics_history[-1]["fuel_consumed_total"] if metrics_history else 0.0,
        "integrated_control_effort": integrated_control_effort_sum,
        "mean_decision_latency_sec": (
            float(total_decision_latency_sec / max(total_decision_calls, 1))
            if total_decision_calls > 0
            else 0.0
        ),
        "time_in_mode_per_agent_sec": dwell_modes,
        "time_in_behavior_per_agent_sec": dwell_behaviors,
        "decision_calls_total": total_decision_calls,
        "decision_invalid_total": total_decision_invalid,
        "decision_latency_sum_sec": total_decision_latency_sec,
        "llm_calls_total": total_llm_calls,
        "llm_prompt_tokens_total": total_llm_prompt_tokens,
        "llm_completion_tokens_total": total_llm_completion_tokens,
    }
    save_json(os.path.join(results_dir, "performance.json"), performance)

    print(f"[sim] done -> {results_dir}", flush=True)
    return results_dir


if __name__ == "__main__":
    default_config = os.path.join(os.path.dirname(__file__), "config.json")
    run_simulation(default_config)

