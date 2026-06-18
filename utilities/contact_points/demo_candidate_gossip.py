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
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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
        max_planes=20,
        distance_threshold=0.035,
        ransac_n=3,
        num_iterations=1200,
        min_inliers=50,
        min_remaining_points=50,
    )

    contact_points = generate_contact_points_from_segments(
        plane_segments,
        contact_spacing=contact_spacing,
        min_points_per_candidate=8,
        boundary_margin=0.15,
    )

    messages = make_candidate_messages(
        agent_id=agent_id,
        contact_points=contact_points,
        plane_segments=plane_segments,
        step=step,
    )

    return plane_segments, contact_points, messages


def gossip_all_to_all(
    maps: list[CandidateGossipMap],
    thresholds: CandidateMatchThresholds,
    step: int,
    gossip_rounds: int,
) -> None:
    for _ in range(gossip_rounds):
        outgoing_by_agent = [
            gossip_map.export_messages(agent_id=agent_id, step=step)
            for agent_id, gossip_map in enumerate(maps)
        ]

        for receiver_id, gossip_map in enumerate(maps):
            received_messages = []

            for sender_id, messages in enumerate(outgoing_by_agent):
                if sender_id == receiver_id:
                    continue

                received_messages.extend(messages)

            gossip_map.update_with_messages(
                messages=received_messages,
                thresholds=thresholds,
                current_step=step,
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

    for agent_id in range(trajectories.shape[1]):
        color = cmap(agent_id)
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
            marker="^",
            edgecolors="black",
        )

    for agent_id, gossip_map in enumerate(maps):
        positions, normals, ids = gossip_map.as_arrays()

        if len(ids) == 0:
            continue

        color = cmap(agent_id)

        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            s=90,
            color=color,
            edgecolors="black",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo decentralized contact candidate memory and gossip."
    )

    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--gossip-rounds", type=int, default=2)
    parser.add_argument("--contact-spacing", type=float, default=1.0)
    parser.add_argument("--orbit-radius", type=float, default=5.2)
    parser.add_argument("--z-amplitude", type=float, default=1.2)
    parser.add_argument("--angular-rate", type=float, default=0.45)
    parser.add_argument("--fov-deg", type=float, default=75.0)
    parser.add_argument("--max-surface-angle-deg", type=float, default=85.0)
    parser.add_argument("--max-points-per-agent", type=int, default=650)
    parser.add_argument("--max-print-candidates", type=int, default=12)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save", type=str, default=None)
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
        position_threshold=0.35,
        normal_angle_threshold_deg=20.0,
        parent_plane_angle_threshold_deg=20.0,
        parent_plane_distance_threshold=0.08,
    )

    maps = [
        CandidateGossipMap()
        for _ in range(num_agents)
    ]
    metric_history = []

    for step in range(args.num_steps):
        agent_positions = trajectories[step]
        visible_counts = []
        local_counts = []

        for agent_id in range(num_agents):
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

            _, contact_points, messages = run_local_perception(
                points=partial_points,
                agent_id=agent_id,
                step=step,
                contact_spacing=args.contact_spacing,
            )

            maps[agent_id].update_with_messages(
                messages=messages,
                thresholds=thresholds,
                current_step=step,
            )

            visible_counts.append(len(partial_points))
            local_counts.append(len(contact_points))

        gossip_all_to_all(
            maps=maps,
            thresholds=thresholds,
            step=step,
            gossip_rounds=args.gossip_rounds,
        )

        for gossip_map in maps:
            gossip_map.decay_unseen(
                current_step=step,
                max_missed_steps=10,
                decay_rate=0.95,
                min_confidence=0.1,
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


if __name__ == "__main__":
    main()
