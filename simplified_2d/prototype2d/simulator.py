from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import controllers
from .delays import DelayModel
from .io import load_config, load_target_definition, save_json
from .metrics import convex_hull_area, distance_stats
from .model import Agent, MetricsSnapshot, TargetState
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


def _update_map(agent: Agent, point_ids: List[int], current_time: float) -> None:
    for pid in point_ids:
        entry = agent.map.get(pid)
        if entry is None:
            agent.map[pid] = {
                "first_seen": current_time,
                "last_seen": current_time,
                "num_observations": 1,
            }
        else:
            entry["last_seen"] = current_time
            entry["num_observations"] += 1


def _distance_to_ap(point_pos: np.ndarray, agent_state: List[float]) -> float:
    return float(np.linalg.norm(point_pos - np.array(agent_state[0:2])))


def _resolve_ap_conflicts(
    agents: List[Agent],
    attachment_world: List[Dict],
) -> int:
    ap_positions = {ap["id"]: ap["pos"] for ap in attachment_world}
    conflicts = 0
    for agent in agents:
        target_ap = agent.target_ap
        if target_ap is None:
            continue
        for sender_id, msg in agent.last_messages_by_sender.items():
            payload = msg.get("payload", {})
            sender_ap = payload.get("target_ap")
            sender_state = payload.get("state")
            if sender_ap is None or sender_ap != target_ap or sender_state is None:
                continue
            ap_pos = ap_positions.get(target_ap)
            if ap_pos is None:
                continue
            self_dist = _distance_to_ap(ap_pos, agent.state)
            sender_dist = _distance_to_ap(ap_pos, sender_state)
            if sender_dist < self_dist:
                agent.target_ap = None
                conflicts += 1
                break
    return conflicts

def _select_nearest_ap(agent: Agent, aps: List[Dict]) -> Optional[int]:
    if not aps:
        return None
    pos = np.array(agent.state[0:2])
    distances = [np.linalg.norm(ap["pos"] - pos) for ap in aps]
    return aps[int(np.argmin(distances))]["id"]


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


def run_simulation(config_path: str) -> str:
    config = load_config(config_path)
    target_def = load_target_definition(config.target_json_path)

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
                mode="s",
                mass=config.agent_mass,
                inertia=config.agent_inertia,
                action_time=config.search_duration + config.encapsulate_duration,
            )
        )

    results_dir = os.path.join(config.output_root, config.name)
    os.makedirs(results_dir, exist_ok=True)

    agents_history: List[List[Dict]] = []
    target_history: List[Dict] = []
    attachment_history: List[List[Dict]] = []
    metrics_history: List[Dict] = []
    messages_history: List[Dict] = []
    time_to_all_docked: Optional[float] = None

    perception_delay_model = DelayModel(config.perception_delay)
    communication_delay_model = DelayModel(config.communication_delay)
    actuation_delay_model = DelayModel(config.actuation_delay)
    perception_queue: List[Dict] = []
    message_queue: List[Dict] = []
    actuation_queue: List[Dict] = []
    last_applied: Dict[int, Tuple[np.ndarray, float]] = {
        agent.id: (np.zeros(2), 0.0) for agent in agents
    }

    total_steps = int(np.floor(config.duration / config.dt))
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
            _update_map(agent, item["point_ids"], current_time)
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
                if current_time < config.search_duration:
                    agent.mode = "s"
                elif current_time < config.search_duration + config.encapsulate_duration:
                    agent.mode = "e"
                else:
                    agent.mode = "c"

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
            _update_map(agent, item["point_ids"], current_time)
            agent.land_set.extend(item["point_ids"])

        for agent in agents:
            payload = {
                "state": list(agent.state),
                "mode": agent.mode,
                "target_ap": agent.target_ap,
                "action_time": agent.action_time,
                "visible_point_ids": list(agent.land_set),
                "known_point_ids": list(agent.map.keys()),
            }
            for neighbor_id in agent.comm_set:
                comm_delay = communication_delay_model.sample(neighbor_id)
                message_queue.append(
                    {
                        "message_id": f"{agent.id}-{neighbor_id}-{step}",
                        "sent_time": current_time,
                        "deliver_time": current_time + comm_delay,
                        "sender_id": agent.id,
                        "recipient_id": neighbor_id,
                        "channel": "broadcast",
                        "message_type": "agent_status",
                        "payload": payload,
                        "metadata": {},
                    }
                )

        ready_messages, message_queue = _queue_ready(message_queue, current_time)
        for message in ready_messages:
            recipient = agents[message["recipient_id"]]
            recipient.inbox.append(message)
            recipient.last_messages_by_sender[message["sender_id"]] = message
            message_ages.append(current_time - message["sent_time"])
            messages_history.append(message)

        for agent in agents:
            bids = []
            aps = []
            for ap in attachment_world:
                distance = np.linalg.norm(ap["pos"] - np.array(agent.state[0:2]))
                if distance <= config.ap_detection_radius:
                    aps.append(ap["id"])
                    bids.append(1.0 / max(distance, 1e-3))
            agent.aps = aps
            agent.aps_bids = bids
            if agent.mode == "c":
                if aps:
                    best_idx = int(np.argmax(bids))
                    agent.target_ap = aps[best_idx]
                else:
                    agent.target_ap = None

        ap_conflicts = _resolve_ap_conflicts(agents, attachment_world)

        for agent in agents:
            if agent.mode == "c" and agent.target_ap is not None:
                ap = next((ap for ap in attachment_world if ap["id"] == agent.target_ap), None)
                if ap is not None:
                    distance = np.linalg.norm(ap["pos"] - np.array(agent.state[0:2]))
                    if distance <= config.dock_distance:
                        agent.mode = "d"
                        agent.dock_pose = _dock_pose(agent, target_state.state)
                        agent.dock_time = current_time

            if agent.mode == "d":
                agent.control_force = [0.0, 0.0]
                agent.control_torque = 0.0
                continue

            if agent.mode == "s":
                force, torque = controllers.search_controller(agent, target_center, config)
            elif agent.mode == "e":
                force, torque = controllers.encapsulate_controller(agent, target_center, config)
            else:
                ap_pos = None
                if agent.target_ap is not None:
                    ap = next((ap for ap in attachment_world if ap["id"] == agent.target_ap), None)
                    if ap is not None:
                        ap_pos = ap["pos"]
                force, torque = controllers.capture_controller(agent, ap_pos, config)

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
        mode_counts = {"s": 0, "e": 0, "c": 0, "d": 0}
        total_control_effort = 0.0
        total_fuel = 0.0
        for agent in agents:
            mode_counts[agent.mode] = mode_counts.get(agent.mode, 0) + 1
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
            if agent.mode == "c" and agent.target_ap is not None:
                ap = next((ap for ap in attachment_world if ap["id"] == agent.target_ap), None)
                if ap is not None:
                    capture_errors.append(_distance_to_ap(ap["pos"], agent.state))
        capture_error_mean = float(np.mean(capture_errors)) if capture_errors else 0.0

        message_age_mean = float(np.mean(message_ages)) if message_ages else 0.0
        message_age_max = float(np.max(message_ages)) if message_ages else 0.0

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
        print(
            f"[sim] step {step}/{total_steps} t={current_time:.3f}s modes={modes_compact}",
            flush=True,
        )

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

    performance = {
        "experiment": config.name,
        "time_to_all_docked": time_to_all_docked,
        "final_convex_hull_area": metrics_history[-1]["convex_hull_area"] if metrics_history else 0.0,
        "total_fuel_consumed": metrics_history[-1]["fuel_consumed_total"] if metrics_history else 0.0,
    }
    save_json(os.path.join(results_dir, "performance.json"), performance)

    print(f"[sim] done -> {results_dir}", flush=True)
    return results_dir


if __name__ == "__main__":
    default_config = os.path.join(os.path.dirname(__file__), "config.json")
    run_simulation(default_config)

