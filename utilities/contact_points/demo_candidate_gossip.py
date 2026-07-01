"""
demo_candidate_gossip.py

Mini-simulation for decentralized contact candidate memory and gossip.

Pipeline:
    mock target cloud
        -> partial per-agent observations
        -> plane RANSAC
        -> local contact candidates
        -> per-agent candidate memory
        -> all-to-all gossip consensus

Assumption:
    All agents already share the same target/map frame.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

UTILITIES_DIR = Path(__file__).resolve().parents[1]
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

try:
    from ..data.mock_data import (
        generate_mock_satellite_point_cloud,
        generate_mock_agent_trajectories,
        generate_mock_partial_observations,
    )
    from .plane_ransac import segment_planes_ransac
    from .contact_points import generate_contact_points_from_segments
    from .candidate_gossip import (
        CandidateGossipMap,
        CandidateMatchThresholds,
        make_candidate_messages,
    )
except ImportError:
    from data.mock_data import (
        generate_mock_satellite_point_cloud,
        generate_mock_agent_trajectories,
        generate_mock_partial_observations,
    )
    from plane_ransac import segment_planes_ransac
    from contact_points import generate_contact_points_from_segments
    from candidate_gossip import (
        CandidateGossipMap,
        CandidateMatchThresholds,
        make_candidate_messages,
    )


def select_visible_points_for_agent(
    full_points: np.ndarray,
    full_normals: np.ndarray,
    agent_position: np.ndarray,
    target_center: np.ndarray,
    fov_deg: float,
    max_surface_angle_deg: float,
    max_points_per_agent: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Select a deterministic partial map from one agent viewpoint.
    """
    observed_points, _ = generate_mock_partial_observations(
        target_points=full_points,
        target_normals=full_normals,
        agent_positions=agent_position[None, :],
        target_center=target_center,
        fov_deg=fov_deg,
        max_surface_angle_deg=max_surface_angle_deg,
        max_points_per_agent=max_points_per_agent,
        seed=seed,
    )

    return observed_points


def run_local_perception(
    points: np.ndarray,
    agent_id: int,
    step: int,
    contact_spacing: float,
):
    plane_segments, _ = segment_planes_ransac(
        points,
        max_planes=10,
        distance_threshold=0.02,
        ransac_n=3,
        num_iterations=2500,
        min_inliers=80,
        min_remaining_points=80,
    )

    contact_points = generate_contact_points_from_segments(
        plane_segments,
        contact_spacing=contact_spacing,
        min_points_per_candidate=16,
        boundary_margin=0.20,
    )

    messages = make_candidate_messages(
        agent_id=agent_id,
        contact_points=contact_points,
        plane_segments=plane_segments,
        step=step,
    )

    return plane_segments, contact_points, messages


def perturb_candidate_messages_for_demo(
    messages,
    agent_id: int,
    offset_scale: float,
    noise_scale: float,
    seed: int,
):
    if offset_scale == 0.0 and noise_scale == 0.0:
        return messages

    angle = 2.0 * np.pi * agent_id / 3.0
    offset = float(offset_scale) * np.array([
        np.cos(angle),
        np.sin(angle),
        0.35 * np.sin(2.0 * angle),
    ])
    rng = np.random.default_rng(seed)

    for message in messages:
        message.position = np.asarray(message.position, dtype=float).copy()
        message.position += offset
        if noise_scale > 0.0:
            message.position += rng.normal(scale=float(noise_scale), size=3)

    return messages


def gossip_all_to_all(
    maps: list[CandidateGossipMap],
    thresholds: CandidateMatchThresholds,
    step: int,
    gossip_rounds: int,
    snapshot_callback=None,
) -> None:
    for gossip_round in range(gossip_rounds):
        round_start = time.time()
        round_trace_events = []
        print(f"[gossip] step={step} round={gossip_round + 1}/{gossip_rounds}: exporting messages...")
        outgoing_by_agent = [
            gossip_map.export_messages(agent_id=agent_id, step=step)
            for agent_id, gossip_map in enumerate(maps)
        ]
        outgoing_counts = [len(messages) for messages in outgoing_by_agent]
        print(f"[gossip] step={step} round={gossip_round + 1}/{gossip_rounds}: outgoing counts={outgoing_counts}")

        for receiver_id, gossip_map in enumerate(maps):
            receiver_start = time.time()
            received_messages = []

            for sender_id, messages in enumerate(outgoing_by_agent):
                if sender_id == receiver_id:
                    continue

                received_messages.extend(messages)

            before_count = len(gossip_map.candidates)
            print(
                f"[gossip] step={step} round={gossip_round + 1}/{gossip_rounds} "
                f"receiver={receiver_id}: received={len(received_messages)} "
                f"candidates_before={before_count}"
            )
            gossip_map.update_with_messages(
                messages=received_messages,
                thresholds=thresholds,
                current_step=step,
                trace_events=round_trace_events,
                receiver_id=receiver_id,
                gossip_round=gossip_round + 1,
                phase="gossip",
            )
            print(
                f"[gossip] step={step} round={gossip_round + 1}/{gossip_rounds} "
                f"receiver={receiver_id}: candidates_after={len(gossip_map.candidates)} "
                f"dt={time.time() - receiver_start:.2f}s"
            )

        if snapshot_callback is not None:
            snapshot_callback(gossip_round + 1, round_trace_events)
        print(
            f"[gossip] step={step} round={gossip_round + 1}/{gossip_rounds}: "
            f"done events={len(round_trace_events)} dt={time.time() - round_start:.2f}s"
        )


def pairwise_map_overlap(
    map_a: CandidateGossipMap,
    map_b: CandidateGossipMap,
    position_threshold: float,
) -> int:
    """
    Count how many candidates in map_a have a nearby candidate in map_b.
    """
    positions_a, _, _ = map_a.as_arrays()
    positions_b, _, _ = map_b.as_arrays()

    if len(positions_a) == 0 or len(positions_b) == 0:
        return 0

    overlap_count = 0

    for position in positions_a:
        distances = np.linalg.norm(positions_b - position, axis=1)

        if np.min(distances) <= position_threshold:
            overlap_count += 1

    return overlap_count


def collect_gossip_metrics(
    step: int,
    maps: list[CandidateGossipMap],
    thresholds: CandidateMatchThresholds,
) -> dict:
    shared_counts = []
    multi_agent_counts = []
    mean_confidences = []
    mean_observation_counts = []

    for gossip_map in maps:
        shared_counts.append(len(gossip_map.candidates))
        multi_agent_counts.append(sum(
            len(candidate.supporting_agents) > 1
            for candidate in gossip_map.candidates
        ))

        if len(gossip_map.candidates) == 0:
            mean_confidences.append(0.0)
            mean_observation_counts.append(0.0)
        else:
            mean_confidences.append(float(np.mean([
                candidate.confidence
                for candidate in gossip_map.candidates
            ])))
            mean_observation_counts.append(float(np.mean([
                candidate.observation_count
                for candidate in gossip_map.candidates
            ])))

    pairwise_overlaps = {}

    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            overlap_ij = pairwise_map_overlap(
                maps[i],
                maps[j],
                thresholds.position_threshold,
            )
            overlap_ji = pairwise_map_overlap(
                maps[j],
                maps[i],
                thresholds.position_threshold,
            )
            pairwise_overlaps[(i, j)] = 0.5 * (overlap_ij + overlap_ji)

    return {
        "step": step,
        "shared_counts": shared_counts,
        "multi_agent_counts": multi_agent_counts,
        "mean_confidences": mean_confidences,
        "mean_observation_counts": mean_observation_counts,
        "pairwise_overlaps": pairwise_overlaps,
    }


def print_timestep_summary(
    step: int,
    agent_positions: np.ndarray,
    visible_counts: list[int],
    local_counts: list[int],
    maps: list[CandidateGossipMap],
    verbose: bool = False,
    max_print_candidates: int = 8,
) -> None:
    print(f"\nTimestep {step}")
    print("-" * 80)

    for agent_id, gossip_map in enumerate(maps):
        multi_agent_count = sum(
            len(candidate.supporting_agents) > 1
            for candidate in gossip_map.candidates
        )
        print(
            f"Agent {agent_id}: "
            f"pos={np.round(agent_positions[agent_id], 2)} "
            f"visible points={visible_counts[agent_id]} "
            f"local observed candidates={local_counts[agent_id]} "
            f"shared candidates={len(gossip_map.candidates)} "
            f"multi-agent={multi_agent_count}"
        )

        if not verbose:
            continue

        for candidate in gossip_map.candidates[:max_print_candidates]:
            print(
                f"  shared {candidate.shared_cp_id:02d} "
                f"conf={candidate.confidence:.3f} "
                f"agents={sorted(candidate.supporting_agents)} "
                f"obs={candidate.observation_count}"
            )

        remaining = len(gossip_map.candidates) - max_print_candidates

        if remaining > 0:
            print(f"  ... {remaining} more shared candidates")


def print_final_maps(
    maps: list[CandidateGossipMap],
    verbose: bool = False,
    max_print_candidates: int = 12,
) -> None:
    print("\nFinal shared candidate maps")
    print("=" * 80)

    for agent_id, gossip_map in enumerate(maps):
        multi_agent_count = sum(
            len(candidate.supporting_agents) > 1
            for candidate in gossip_map.candidates
        )
        print(
            f"\nAgent {agent_id}: "
            f"shared candidates={len(gossip_map.candidates)} "
            f"multi-agent={multi_agent_count}"
        )

        candidates_to_print = (
            gossip_map.candidates
            if verbose
            else gossip_map.candidates[:max_print_candidates]
        )

        for candidate in candidates_to_print:
            print(
                f"  shared {candidate.shared_cp_id:02d} "
                f"pos={np.round(candidate.position, 3)} "
                f"normal={np.round(candidate.normal, 3)} "
                f"conf={candidate.confidence:.3f} "
                f"agents={sorted(candidate.supporting_agents)} "
                f"obs={candidate.observation_count}"
            )

        remaining = len(gossip_map.candidates) - len(candidates_to_print)

        if remaining > 0:
            print(f"  ... {remaining} more shared candidates")


def set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_final_maps(
    full_points: np.ndarray,
    trajectories: np.ndarray,
    maps: list[CandidateGossipMap],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        full_points[:, 0],
        full_points[:, 1],
        full_points[:, 2],
        s=1,
        alpha=0.05,
        c="gray",
        label="mock target",
    )

    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    for agent_id in range(trajectories.shape[1]):
        color = cmap(agent_id)
        marker = markers[agent_id % len(markers)]
        trajectory = trajectories[:, agent_id, :]

        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            color=color,
            linewidth=1.5,
            alpha=0.65,
            label=f"agent {agent_id} trajectory",
        )

        ax.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            trajectory[-1, 2],
            s=120,
            color=color,
            marker=marker,
            edgecolors="black",
        )

    for agent_id, gossip_map in enumerate(maps):
        positions, normals, ids = gossip_map.as_arrays()

        if len(ids) == 0:
            continue

        color = cmap(agent_id)
        marker = markers[agent_id % len(markers)]

        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            s=90,
            color=color,
            edgecolors="black",
            marker=marker,
            label=f"agent {agent_id} shared candidates",
        )

        ax.quiver(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            normals[:, 0],
            normals[:, 1],
            normals[:, 2],
            length=0.25,
            color=color,
            linewidth=1.0,
        )

        for candidate in gossip_map.candidates:
            ax.text(
                candidate.position[0],
                candidate.position[1],
                candidate.position[2],
                f"A{agent_id}:{candidate.shared_cp_id}"
                f" ({len(candidate.supporting_agents)})",
                fontsize=7,
                color="black",
            )

    ax.set_title("Final Shared Contact Candidate Maps After Gossip")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=24, azim=-42)
    all_plot_points = np.vstack([
        full_points,
        trajectories.reshape(-1, 3),
    ])
    set_axes_equal(ax, all_plot_points)
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved preview to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_gossip_metrics(
    metric_history: list[dict],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    if len(metric_history) == 0:
        return

    steps = np.array([entry["step"] for entry in metric_history])
    num_agents = len(metric_history[0]["shared_counts"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    cmap = plt.get_cmap("tab10")

    for agent_id in range(num_agents):
        color = cmap(agent_id)
        axes[0, 0].plot(
            steps,
            [entry["shared_counts"][agent_id] for entry in metric_history],
            marker="o",
            color=color,
            label=f"agent {agent_id}",
        )
        axes[0, 1].plot(
            steps,
            [entry["multi_agent_counts"][agent_id] for entry in metric_history],
            marker="o",
            color=color,
            label=f"agent {agent_id}",
        )
        axes[1, 0].plot(
            steps,
            [entry["mean_confidences"][agent_id] for entry in metric_history],
            marker="o",
            color=color,
            label=f"agent {agent_id}",
        )

    pair_keys = sorted(metric_history[0]["pairwise_overlaps"].keys())

    for pair_id, pair_key in enumerate(pair_keys):
        axes[1, 1].plot(
            steps,
            [entry["pairwise_overlaps"][pair_key] for entry in metric_history],
            marker="o",
            color=cmap(pair_id),
            label=f"{pair_key[0]}-{pair_key[1]}",
        )

    axes[0, 0].set_title("Shared Candidate Count")
    axes[0, 1].set_title("Multi-Agent Supported Count")
    axes[1, 0].set_title("Mean Confidence")
    axes[1, 1].set_title("Pairwise Map Overlap")

    axes[0, 0].set_ylabel("count")
    axes[0, 1].set_ylabel("count")
    axes[1, 0].set_ylabel("confidence")
    axes[1, 1].set_ylabel("overlap count")

    for ax in axes.ravel():
        ax.set_xlabel("timestep")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle("Candidate Gossip Convergence Metrics")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved metrics plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def snapshot_gossip_state(
    step: int,
    gossip_round: int,
    agent_positions: np.ndarray,
    maps: list[CandidateGossipMap],
    thresholds: CandidateMatchThresholds,
    trace_events: list[dict] | None = None,
) -> dict:
    agent_snapshots = []

    for gossip_map in maps:
        positions, normals, ids = gossip_map.as_arrays()
        confidences = np.array(
            [candidate.confidence for candidate in gossip_map.candidates],
            dtype=float,
        )
        support_counts = np.array(
            [len(candidate.supporting_agents) for candidate in gossip_map.candidates],
            dtype=int,
        )
        agent_snapshots.append(
            {
                "positions": positions.copy(),
                "normals": normals.copy(),
                "ids": ids.copy(),
                "confidences": confidences,
                "support_counts": support_counts,
            }
        )

    return {
        "step": int(step),
        "gossip_round": int(gossip_round),
        "agent_positions": np.asarray(agent_positions, dtype=float).copy(),
        "agents": agent_snapshots,
        "metrics": collect_gossip_metrics(step, maps, thresholds),
        "trace_events": list(trace_events or []),
    }


def find_cross_agent_candidate_matches(snapshot: dict, position_threshold: float) -> list[tuple[np.ndarray, np.ndarray]]:
    matches = []
    agents = snapshot["agents"]

    for agent_a in range(len(agents)):
        positions_a = agents[agent_a]["positions"]
        if len(positions_a) == 0:
            continue
        for agent_b in range(agent_a + 1, len(agents)):
            positions_b = agents[agent_b]["positions"]
            if len(positions_b) == 0:
                continue
            for pos_a in positions_a:
                dists = np.linalg.norm(positions_b - pos_a, axis=1)
                if len(dists) == 0:
                    continue
                idx = int(np.argmin(dists))
                if dists[idx] <= position_threshold:
                    matches.append((pos_a.copy(), positions_b[idx].copy()))

    return matches


def event_position(event: dict, key: str):
    value = event.get(key)
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def save_gossip_animation(
    full_points: np.ndarray,
    trajectories: np.ndarray,
    snapshot_history: list[dict],
    save_path: str,
    thresholds: CandidateMatchThresholds,
    interval_ms: int = 1200,
) -> None:
    if len(snapshot_history) == 0:
        return

    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    frame_repeat = max(1, int(round(float(interval_ms) / 500.0)))
    animation_frame_indices = [
        frame_idx
        for frame_idx in range(len(snapshot_history))
        for _ in range(frame_repeat)
    ]
    print(
        f"[gif] Saving {len(snapshot_history)} logical frames "
        f"({len(animation_frame_indices)} encoded frames, repeat={frame_repeat}) -> {save_path}"
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    target_stride = max(1, len(full_points) // 1800)
    target_points_viz = full_points[::target_stride]
    all_plot_points = np.vstack([full_points, trajectories.reshape(-1, 3)])
    set_axes_equal(ax, all_plot_points)
    max_step = max(snapshot["step"] for snapshot in snapshot_history)

    def update(frame_idx: int):
        snapshot = snapshot_history[frame_idx]
        step = snapshot["step"]
        gossip_round = snapshot["gossip_round"]
        ax.cla()
        ax.scatter(
            target_points_viz[:, 0],
            target_points_viz[:, 1],
            target_points_viz[:, 2],
            s=1,
            alpha=0.018,
            c="gray",
            label="mock target",
        )

        total_candidates = 0
        multi_agent_candidates = 0
        match_lines = find_cross_agent_candidate_matches(snapshot, thresholds.position_threshold)
        trace_events = snapshot.get("trace_events", [])

        for agent_id, agent_snapshot in enumerate(snapshot["agents"]):
            color = cmap(agent_id)
            trajectory = trajectories[: step + 1, agent_id, :]
            agent_position = snapshot["agent_positions"][agent_id]
            marker = markers[agent_id % len(markers)]

            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color=color,
                linewidth=1.0,
                alpha=0.45,
            )
            ax.scatter(
                agent_position[0],
                agent_position[1],
                agent_position[2],
                s=110,
                color=color,
                marker=marker,
                edgecolors="black",
                label=f"agent {agent_id}",
            )

            positions = agent_snapshot["positions"]
            normals = agent_snapshot["normals"]
            support_counts = agent_snapshot["support_counts"]
            confidences = agent_snapshot["confidences"]

            if len(positions) == 0:
                continue

            total_candidates += len(positions)
            multi_agent_candidates += int(np.sum(support_counts > 1))
            visual_offsets = np.zeros_like(positions)
            if len(positions) > 0:
                angles = 2.0 * np.pi * agent_id / max(1, len(snapshot["agents"]))
                visual_offsets[:] = 0.035 * np.array([np.cos(angles), np.sin(angles), 0.25 * np.sin(2.0 * angles)])
            positions_viz = positions + visual_offsets
            sizes = 35.0 + 55.0 * np.clip(confidences, 0.0, 1.0) + 55.0 * np.clip(support_counts - 1, 0, None)
            alphas = 0.45 + 0.45 * np.clip(confidences, 0.0, 1.0)

            ax.scatter(
                positions_viz[:, 0],
                positions_viz[:, 1],
                positions_viz[:, 2],
                s=sizes,
                color=color,
                alpha=float(np.mean(alphas)) if len(alphas) else 0.85,
                edgecolors="black",
                linewidths=np.where(support_counts > 1, 1.6, 0.6),
                marker=marker,
            )
            ax.quiver(
                positions_viz[:, 0],
                positions_viz[:, 1],
                positions_viz[:, 2],
                normals[:, 0],
                normals[:, 1],
                normals[:, 2],
                length=0.12,
                color=color,
                linewidth=0.7,
                alpha=0.55,
            )

        for pos_a, pos_b in match_lines:
            xs, ys, zs = zip(pos_a, pos_b)
            ax.plot(xs, ys, zs, color="black", linewidth=0.7, alpha=0.22)

        for event in trace_events:
            receiver_id = event.get("receiver_id")
            sender_id = event.get("sender_id")
            receiver_color = cmap(int(receiver_id) % 10) if receiver_id is not None else "black"
            sender_color = cmap(int(sender_id) % 10) if sender_id is not None else receiver_color
            sender_marker = markers[int(sender_id) % len(markers)] if sender_id is not None else "o"
            message_pos = event_position(event, "message_position")
            pre_pos = event_position(event, "pre_merge_position")
            post_pos = event_position(event, "post_merge_position")

            if message_pos is not None:
                ax.scatter(
                    message_pos[0],
                    message_pos[1],
                    message_pos[2],
                    s=150,
                    marker=sender_marker,
                    facecolors="none",
                    edgecolors=sender_color,
                    linewidths=1.8,
                    alpha=0.95,
                )

            if pre_pos is not None:
                ax.scatter(
                    pre_pos[0],
                    pre_pos[1],
                    pre_pos[2],
                    s=105,
                    marker="x",
                    color=receiver_color,
                    linewidths=1.6,
                    alpha=0.9,
                )

            if pre_pos is not None and post_pos is not None:
                xs, ys, zs = zip(pre_pos, post_pos)
                ax.plot(xs, ys, zs, color=receiver_color, linewidth=2.0, alpha=0.8)
            elif message_pos is not None and post_pos is not None and not event.get("matched", False):
                xs, ys, zs = zip(message_pos, post_pos)
                ax.plot(xs, ys, zs, color=sender_color, linewidth=1.2, alpha=0.45, linestyle=":")

        set_axes_equal(ax, all_plot_points)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=24, azim=-42)
        ax.set_title(
            f"Candidate Gossip | step {step}/{max_step} | round {gossip_round} | "
            f"candidates={total_candidates} | multi-agent={multi_agent_candidates} | "
            f"events={len(trace_events)}"
        )
        ax.legend(loc="upper right")
        fig.tight_layout()
        return []

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=animation_frame_indices,
        interval=500,
        repeat=False,
        blit=False,
    )
    ani.save(
        save_path,
        writer=animation.PillowWriter(fps=2),
    )
    plt.close(fig)
    print(f"Saved gossip animation to: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo decentralized contact candidate memory and gossip."
    )

    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--gossip-rounds", type=int, default=2)
    parser.add_argument("--contact-spacing", type=float, default=1.25)
    parser.add_argument("--orbit-radius", type=float, default=5.2)
    parser.add_argument("--z-amplitude", type=float, default=1.2)
    parser.add_argument("--angular-rate", type=float, default=0.45)
    parser.add_argument("--fov-deg", type=float, default=75.0)
    parser.add_argument("--max-surface-angle-deg", type=float, default=85.0)
    parser.add_argument("--max-points-per-agent", type=int, default=650)
    parser.add_argument("--max-print-candidates", type=int, default=12)
    parser.add_argument("--demo-agent-offset", type=float, default=0.20)
    parser.add_argument("--candidate-position-noise", type=float, default=0.0)
    parser.add_argument("--position-threshold", type=float, default=0.55)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--save-gif", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    num_agents = 3
    target_center = np.array([0.0, 0.0, 0.0])
    full_points, full_normals, _ = generate_mock_satellite_point_cloud(seed=args.seed)
    trajectory_steps = max(args.num_steps, 2)

    trajectories, _ = generate_mock_agent_trajectories(
        num_agents=num_agents,
        num_steps=trajectory_steps,
        orbit_radius=args.orbit_radius,
        z_amplitude=args.z_amplitude,
        angular_rate=args.angular_rate,
        seed=args.seed,
    )

    thresholds = CandidateMatchThresholds(
        position_threshold=float(args.position_threshold),
        normal_angle_threshold_deg=20.0,
        parent_plane_angle_threshold_deg=20.0,
        parent_plane_distance_threshold=0.08,
    )

    maps = [
        CandidateGossipMap()
        for _ in range(num_agents)
    ]
    metric_history = []
    snapshot_history = []

    for step in range(args.num_steps):
        step_start = time.time()
        print(f"\n[demo] step={step}/{args.num_steps - 1}: starting local perception")
        agent_positions = trajectories[step]
        visible_counts = []
        local_counts = []
        local_trace_events = []

        for agent_id in range(num_agents):
            agent_start = time.time()
            print(f"[local] step={step} agent={agent_id}: selecting visible points...")
            partial_points = select_visible_points_for_agent(
                full_points=full_points,
                full_normals=full_normals,
                agent_position=agent_positions[agent_id],
                target_center=target_center,
                fov_deg=args.fov_deg,
                max_surface_angle_deg=args.max_surface_angle_deg,
                max_points_per_agent=args.max_points_per_agent,
                seed=args.seed + 100 * step + agent_id,
            )

            print(
                f"[local] step={step} agent={agent_id}: "
                f"visible_points={len(partial_points)} running RANSAC/contact extraction..."
            )
            _, contact_points, messages = run_local_perception(
                points=partial_points,
                agent_id=agent_id,
                step=step,
                contact_spacing=args.contact_spacing,
            )
            messages = perturb_candidate_messages_for_demo(
                messages=messages,
                agent_id=agent_id,
                offset_scale=float(args.demo_agent_offset),
                noise_scale=float(args.candidate_position_noise),
                seed=args.seed + 1000 * step + agent_id,
            )

            before_count = len(maps[agent_id].candidates)
            print(
                f"[local] step={step} agent={agent_id}: "
                f"local_contact_points={len(contact_points)} messages={len(messages)} "
                f"candidates_before={before_count} updating local map..."
            )
            maps[agent_id].update_with_messages(
                messages=messages,
                thresholds=thresholds,
                current_step=step,
                trace_events=local_trace_events,
                receiver_id=agent_id,
                gossip_round=0,
                phase="local",
            )
            print(
                f"[local] step={step} agent={agent_id}: "
                f"candidates_after={len(maps[agent_id].candidates)} "
                f"dt={time.time() - agent_start:.2f}s"
            )

            visible_counts.append(len(partial_points))
            local_counts.append(len(contact_points))

        agent_positions_for_snapshot = agent_positions.copy()

        def record_round_snapshot(gossip_round: int, trace_events: list[dict] | None = None) -> None:
            snapshot_history.append(
                snapshot_gossip_state(
                    step=step,
                    gossip_round=gossip_round,
                    agent_positions=agent_positions_for_snapshot,
                    maps=maps,
                    thresholds=thresholds,
                    trace_events=trace_events,
                )
            )

        record_round_snapshot(0, local_trace_events)
        print(
            f"[demo] step={step}: local phase done local_events={len(local_trace_events)} "
            f"candidate_counts={[len(gossip_map.candidates) for gossip_map in maps]}"
        )

        gossip_all_to_all(
            maps=maps,
            thresholds=thresholds,
            step=step,
            gossip_rounds=args.gossip_rounds,
            snapshot_callback=record_round_snapshot,
        )
        print(
            f"[demo] step={step}: gossip done candidate_counts="
            f"{[len(gossip_map.candidates) for gossip_map in maps]}"
        )

        for gossip_map in maps:
            gossip_map.decay_unseen(
                current_step=step,
                max_missed_steps=10,
                decay_rate=0.95,
                min_confidence=0.1,
            )
        print(
            f"[demo] step={step}: decay done candidate_counts="
            f"{[len(gossip_map.candidates) for gossip_map in maps]}"
        )

        print_timestep_summary(
            step=step,
            agent_positions=agent_positions,
            visible_counts=visible_counts,
            local_counts=local_counts,
            maps=maps,
            verbose=args.verbose,
            max_print_candidates=args.max_print_candidates,
        )

        metric_history.append(
            collect_gossip_metrics(
                step=step,
                maps=maps,
                thresholds=thresholds,
            )
        )
        print(
            f"[demo] step={step}/{args.num_steps - 1}: finished "
            f"dt={time.time() - step_start:.2f}s snapshots={len(snapshot_history)}"
        )

    print_final_maps(
        maps,
        verbose=args.verbose,
        max_print_candidates=args.max_print_candidates,
    )

    plot_final_maps(
        full_points=full_points,
        trajectories=trajectories[:args.num_steps],
        maps=maps,
        save_path=args.save,
        show=not args.no_show,
    )

    metrics_save_path = None

    if args.save is not None:
        stem, dot, suffix = args.save.rpartition(".")

        if dot:
            metrics_save_path = f"{stem}_metrics.{suffix}"
        else:
            metrics_save_path = f"{args.save}_metrics"

    plot_gossip_metrics(
        metric_history=metric_history,
        save_path=metrics_save_path,
        show=not args.no_show,
    )

    if args.save_gif is not None:
        save_gossip_animation(
            full_points=full_points,
            trajectories=trajectories[:args.num_steps],
            snapshot_history=snapshot_history,
            save_path=args.save_gif,
            thresholds=thresholds,
        )


if __name__ == "__main__":
    main()
