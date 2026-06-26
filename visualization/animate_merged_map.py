#!/usr/bin/env python3
"""Animate growing MergedMapSet from DDFGO++ Agents_History pickles (matplotlib GIF)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shared_config
from visualization.geometry_helpers import (
    agent_pointing_direction,
    agent_position_rotation,
    compute_axis_limits,
    cube_edge_segments,
    find_valid_iterations,
    get_merged_map_points,
    load_agents_history,
)

#############################################################################################
# Run gate — set True to open interactive matplotlib animation (no GIF save).
#############################################################################################
INTERACTIVE_PREVIEW = True


def _default_agents_pickle() -> Path:
    ddfgo_config_dir = PROJECT_ROOT / "DDFGO++"
    if str(ddfgo_config_dir) not in sys.path:
        sys.path.insert(0, str(ddfgo_config_dir))
    import config as ddfgo_config

    return Path(ddfgo_config.get_results_paths()["agents"])


def _default_downsample() -> int:
    try:
        ddfgo_config_dir = PROJECT_ROOT / "DDFGO++"
        if str(ddfgo_config_dir) not in sys.path:
            sys.path.insert(0, str(ddfgo_config_dir))
        import config as ddfgo_config

        return max(1, int(getattr(ddfgo_config, "animation_downsample_step", 1)))
    except Exception:
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate merged dense map growth from DDFGO++ Agents_History pickle."
    )
    parser.add_argument(
        "--agents-pickle",
        type=Path,
        default=None,
        help="Path to Agents_History_*.pkl (default: DDFGO++/config get_results_paths).",
    )
    parser.add_argument(
        "--highlight-agent",
        type=int,
        default=1,
        help="1-based agent id to highlight (merged map + trail).",
    )
    parser.add_argument(
        "--cube-size",
        type=float,
        default=float(shared_config.VIS_CUBESAT_SIZE_M),
        help="Wireframe cube edge length in meters.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=80,
        help="Number of past positions to keep in motion trail.",
    )
    parser.add_argument(
        "--max-map-points",
        type=int,
        default=8000,
        help="Max merged-map points drawn per frame.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=None,
        help="Use every Nth SLAM timestep (default: DDFGO animation_downsample_step).",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=120,
        help="Delay between animation frames in milliseconds.",
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save GIF to --output (default: save enabled).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GIF path (default: visualization/output/merged_map_agent<id>.gif).",
    )
    return parser.parse_args()


def _load_history(path: Path, downsample: int) -> list:
    agents_history = load_agents_history(path)
    valid = find_valid_iterations(agents_history)
    if valid < len(agents_history):
        print(f"[viz] Truncating history {len(agents_history)} -> {valid} valid iterations")
        agents_history = agents_history[:valid]
    if downsample > 1:
        agents_history = agents_history[::downsample]
    if len(agents_history) == 0:
        raise ValueError("No valid timesteps in Agents_History pickle.")
    return agents_history


def run_animation(args: argparse.Namespace) -> Path | None:
    pickle_path = args.agents_pickle or _default_agents_pickle()
    if not pickle_path.is_file():
        raise FileNotFoundError(f"Agents history pickle not found: {pickle_path}")

    downsample = args.downsample if args.downsample is not None else _default_downsample()
    downsample = max(1, int(downsample))
    agents_history = _load_history(pickle_path, downsample)

    n_agents = len(agents_history[0])
    highlight = int(args.highlight_agent)
    if highlight < 1 or highlight > n_agents:
        raise ValueError(f"--highlight-agent must be in [1, {n_agents}], got {highlight}")

    highlight_idx = highlight - 1
    cube_size = float(args.cube_size)
    trail_len = max(1, int(args.trail_length))
    arrow_scale = 1.5 * cube_size

    limits = compute_axis_limits(agents_history, highlight, max_map_points_for_limits=args.max_map_points)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_zlim(limits[4], limits[5])

    map_scatter = ax.scatter([], [], [], s=1, c="purple", alpha=0.35, depthshade=True, label="MergedMapSet")
    trail_lines = []
    cube_lines = []
    arrow_quivers = []

    for ai in range(n_agents):
        color = "darkorange" if ai == highlight_idx else "0.55"
        lw = 2.0 if ai == highlight_idx else 1.0
        trail_lines.append(ax.plot([], [], [], color=color, linewidth=lw, alpha=0.85)[0])
        cube_lines.append([ax.plot([], [], [], color=color, linewidth=lw)[0] for _ in range(12)])
        arrow_quivers.append(
            ax.quiver(
                0,
                0,
                0,
                0,
                0,
                0,
                color=color,
                linewidth=lw,
                arrow_length_ratio=0.25,
                length=arrow_scale,
            )
        )

    title = ax.set_title("")

    def update(frame_idx: int):
        frame = agents_history[frame_idx]
        agent_highlight = frame[highlight_idx]
        map_pts = get_merged_map_points(agent_highlight, args.max_map_points, frame_idx)
        if len(map_pts) > 0:
            map_scatter._offsets3d = (map_pts[:, 0], map_pts[:, 1], map_pts[:, 2])
        else:
            map_scatter._offsets3d = ([], [], [])

        for ai, agent in enumerate(frame):
            pos, rot = agent_position_rotation(agent)
            direction = agent_pointing_direction(agent)
            start_k = max(0, frame_idx - trail_len + 1)
            trail = np.array(
                [
                    agent_position_rotation(agents_history[k][ai])[0]
                    for k in range(start_k, frame_idx + 1)
                ],
                dtype=np.float64,
            )
            if len(trail) >= 2:
                trail_lines[ai].set_data(trail[:, 0], trail[:, 1])
                trail_lines[ai].set_3d_properties(trail[:, 2])
            else:
                trail_lines[ai].set_data([], [])
                trail_lines[ai].set_3d_properties([])

            edges = cube_edge_segments(pos, rot, cube_size)
            for edge_art, seg in zip(cube_lines[ai], edges):
                edge_art.set_data(seg[:, 0], seg[:, 1])
                edge_art.set_3d_properties(seg[:, 2])

            arrow_quivers[ai].remove()
            color = "darkorange" if ai == highlight_idx else "0.55"
            lw = 2.0 if ai == highlight_idx else 1.0
            arrow_quivers[ai] = ax.quiver(
                pos[0],
                pos[1],
                pos[2],
                direction[0],
                direction[1],
                direction[2],
                color=color,
                linewidth=lw,
                arrow_length_ratio=0.25,
                length=arrow_scale,
            )

        title.set_text(
            f"Frame {frame_idx + 1}/{len(agents_history)} | "
            f"agent {highlight} map pts: {len(map_pts)}"
        )
        artists = [map_scatter, title, *trail_lines]
        for lines in cube_lines:
            artists.extend(lines)
        artists.extend(arrow_quivers)
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(agents_history),
        interval=int(args.interval_ms),
        repeat=False,
        blit=False,
    )

    output_path = args.output
    if output_path is None:
        output_path = (
            PROJECT_ROOT
            / "visualization"
            / "output"
            / f"merged_map_agent{highlight}.gif"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save:
        n_frames = len(agents_history)
        print(f"[viz] Saving GIF ({n_frames} frames) -> {output_path}")

        def _save_progress(current: int, total: int) -> None:
            # Log first, last, and ~10 progress steps while Pillow encodes frames.
            step = max(1, total // 10)
            if current == 0 or current == total - 1 or (current + 1) % step == 0:
                print(f"[viz] encoding frame {current + 1}/{total}")

        ani.save(
            str(output_path),
            writer=animation.PillowWriter(fps=max(1, 1000 // int(args.interval_ms))),
            progress_callback=_save_progress,
        )
        print(f"[viz] Done. Wrote {output_path}")
        plt.close(fig)
        return output_path

    plt.show()
    return None


def main() -> None:
    args = parse_args()
    if INTERACTIVE_PREVIEW:
        args.save = False
        print("[viz] INTERACTIVE_PREVIEW=True — opening matplotlib window (no GIF save)")
    run_animation(args)


if __name__ == "__main__":
    main()
