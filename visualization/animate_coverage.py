#!/usr/bin/env python3
"""Animate map coverage / explore direction from SwarmCapture+ sim Agents_History pickles."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import matplotlib
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


def _default_sim_agents_pickle() -> Path:
    tag = shared_config.get_tag()
    data_dir = shared_config.SWARMCAPTURE_DATA_DIR
    for candidate in (
        data_dir / f"Agents_History{tag}.pkl",
        data_dir / f"Agents_History_{tag}.pkl",
        Path(shared_config.get_sim_data_paths()["agents"]),
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Sim Agents_History pickle not found under SwarmCapture+/Data "
        f"(tag={tag}). Run run_active_slam.py first."
    )


def _ensure_mock_data_shim(utilities_dir: Path) -> None:
    """Let utilities/coverage import when mock_data.py is absent (reuse sim Utilities shim)."""
    if (utilities_dir / "data" / "mock_data.py").is_file() or "data.mock_data" in sys.modules:
        return

    swarm_dir = PROJECT_ROOT / "SwarmCapture+"
    if str(swarm_dir) not in sys.path:
        sys.path.insert(0, str(swarm_dir))
    import Utilities as swarm_util

    mod = types.ModuleType("data.mock_data")
    mod.EllipsoidModel = swarm_util.EllipsoidModel

    def _cli_only(*_args, **_kwargs):
        raise NotImplementedError("utilities/data/mock_data.py is required for coverage CLI demos")

    mod.generate_mock_satellite_point_cloud = _cli_only
    mod.ellipsoid_wireframe = _cli_only
    mod.set_axes_equal = _cli_only
    sys.modules["data.mock_data"] = mod


def _import_coverage_module(*, headless: bool) -> object:
    utilities_dir = PROJECT_ROOT / "utilities"
    coverage_dir = utilities_dir / "coverage"
    for path in (coverage_dir, utilities_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    if headless:
        matplotlib.use("Agg")
    elif matplotlib.get_backend().lower() == "agg":
        for backend in ("TkAgg", "QtAgg", "Qt5Agg", "WXAgg"):
            try:
                matplotlib.use(backend, force=True)
                break
            except ImportError:
                continue

    _ensure_mock_data_shim(utilities_dir)

    import coverage as coverage_mod

    return coverage_mod


def _ellipsoid_wireframe(coverage_mod, ellipsoid, n_u: int = 32, n_v: int = 16):
    theta = np.linspace(0.0, 2.0 * np.pi, n_u)
    phi = np.linspace(0.05, np.pi - 0.05, n_v)
    xw = np.zeros((len(phi), len(theta)))
    yw = np.zeros_like(xw)
    zw = np.zeros_like(xw)
    for i, p in enumerate(phi):
        for j, t in enumerate(theta):
            pt = coverage_mod.ellipsoid_point_from_angles(ellipsoid, t, p)
            xw[i, j], yw[i, j], zw[i, j] = pt
    return xw, yw, zw


def _unit_vector3(value) -> np.ndarray | None:
    if value is None:
        return None
    vec = np.asarray(value, dtype=np.float64).reshape(-1)
    if vec.size != 3:
        return None
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-12 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate ellipsoid coverage and explore direction from sim Agents_History pickle."
    )
    parser.add_argument(
        "--agents-pickle",
        type=Path,
        default=None,
        help="Path to SwarmCapture+ Agents_History*.pkl (default: shared_config sim data dir).",
    )
    parser.add_argument(
        "--highlight-agent",
        type=int,
        default=1,
        help="1-based agent id to highlight (coverage shell + explore arrow).",
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
        default=10,
        help="Use every Nth sim timestep (default: 10).",
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
        help="Output GIF path (default: visualization/output/coverage_agent<id>.gif).",
    )
    return parser.parse_args()


def _load_history(path: Path, downsample: int) -> list:
    agents_history = load_agents_history(path)
    valid = find_valid_iterations(agents_history)
    if valid < len(agents_history):
        print(f"[viz] Truncating history {len(agents_history)} -> {valid} valid iterations")
        agents_history = agents_history[:valid]
    downsample = max(1, int(downsample))
    if downsample > 1:
        agents_history = agents_history[::downsample]
    if len(agents_history) == 0:
        raise ValueError("No valid timesteps in sim Agents_History pickle.")
    return agents_history


def run_animation(args: argparse.Namespace):
    coverage_mod = _import_coverage_module(headless=bool(args.save))
    print(f"[viz] matplotlib backend: {matplotlib.get_backend()}")

    pickle_path = args.agents_pickle or _default_sim_agents_pickle()
    if not pickle_path.is_file():
        raise FileNotFoundError(f"Sim agents history pickle not found: {pickle_path}")
    print(f"[viz] Loading sim history: {pickle_path}")

    agents_history = _load_history(pickle_path, args.downsample)
    n_frames = len(agents_history)
    print(f"[viz] Animating {n_frames} frames (downsample={args.downsample})")

    n_agents = len(agents_history[0])
    highlight = int(args.highlight_agent)
    if highlight < 1 or highlight > n_agents:
        raise ValueError(f"--highlight-agent must be in [1, {n_agents}], got {highlight}")

    highlight_idx = highlight - 1
    cube_size = float(args.cube_size)
    trail_len = max(1, int(args.trail_length))
    arrow_scale = 1.5 * cube_size
    explore_arrow_scale = 2.0 * cube_size

    limits = compute_axis_limits(agents_history, highlight, max_map_points_for_limits=args.max_map_points)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_zlim(limits[4], limits[5])

    map_scatter = ax.scatter([], [], [], s=1, c="purple", alpha=0.35, depthshade=True)
    explore_target_scatter = ax.scatter([], [], [], s=40, c="cyan", marker="x", linewidths=2)

    patch_collections: list = []
    ellipsoid_artists: list = []
    trail_lines = []
    cube_lines = []
    lcd_quivers = []
    explore_quivers = []

    for ai in range(n_agents):
        color = "darkorange" if ai == highlight_idx else "0.55"
        lw = 2.0 if ai == highlight_idx else 1.0
        trail_lines.append(ax.plot([], [], [], color=color, linewidth=lw, alpha=0.85)[0])
        cube_lines.append([ax.plot([], [], [], color=color, linewidth=lw)[0] for _ in range(12)])
        lcd_quivers.append(
            ax.quiver(
                0, 0, 0, 0, 0, 0,
                color=color,
                linewidth=lw,
                arrow_length_ratio=0.25,
                length=arrow_scale,
            )
        )
        if ai == highlight_idx:
            explore_quivers.append(
                ax.quiver(
                    0, 0, 0, 0, 0, 0,
                    color="cyan",
                    linewidth=2.0,
                    arrow_length_ratio=0.25,
                    length=explore_arrow_scale,
                )
            )

    title = ax.set_title("")
    log_step = max(1, n_frames // 20)

    def _clear_ellipsoid_artists() -> None:
        for artist in ellipsoid_artists:
            artist.remove()
        ellipsoid_artists.clear()

    def _draw_ellipsoid(ellipsoid) -> None:
        _clear_ellipsoid_artists()
        if ellipsoid is None:
            return
        xw, yw, zw = _ellipsoid_wireframe(coverage_mod, ellipsoid)
        ellipsoid_artists.append(
            ax.plot_wireframe(xw, yw, zw, color="0.35", linewidth=0.4, alpha=0.3)
        )

    def _clear_patch_collections() -> None:
        for collection in patch_collections:
            collection.remove()
        patch_collections.clear()

    def _draw_patches(patches) -> None:
        _clear_patch_collections()
        if not patches:
            return
        before = len(ax.collections)
        coverage_mod.add_patch_collection(
            ax, coverage_mod.get_uncovered_patches(patches), alpha=0.12, label="uncovered"
        )
        coverage_mod.add_patch_collection(
            ax, coverage_mod.get_covered_patches(patches), alpha=0.55, label="covered"
        )
        patch_collections.extend(ax.collections[before:])

    def update(frame_idx: int):
        frame = agents_history[frame_idx]
        agent_highlight = frame[highlight_idx]

        map_pts = get_merged_map_points(agent_highlight, args.max_map_points, frame_idx)
        map_scatter._offsets3d = (
            (map_pts[:, 0], map_pts[:, 1], map_pts[:, 2]) if len(map_pts) > 0 else ([], [], [])
        )

        _draw_ellipsoid(agent_highlight.get("MapEllipsoid"))
        _draw_patches(agent_highlight.get("MapCoveragePatches") or [])

        explore_pt = agent_highlight.get("MapExploreTarget")
        if explore_pt is not None:
            pt = np.asarray(explore_pt, dtype=np.float64).reshape(-1)
            if pt.size == 3:
                explore_target_scatter._offsets3d = ([pt[0]], [pt[1]], [pt[2]])
            else:
                explore_target_scatter._offsets3d = ([], [], [])
        else:
            explore_target_scatter._offsets3d = ([], [], [])

        for ai, agent in enumerate(frame):
            pos, rot = agent_position_rotation(agent)
            lcd_dir = agent_pointing_direction(agent)

            start_k = max(0, frame_idx - trail_len + 1)
            trail = np.array(
                [agent_position_rotation(agents_history[k][ai])[0] for k in range(start_k, frame_idx + 1)],
                dtype=np.float64,
            )
            if len(trail) >= 2:
                trail_lines[ai].set_data(trail[:, 0], trail[:, 1])
                trail_lines[ai].set_3d_properties(trail[:, 2])
            else:
                trail_lines[ai].set_data([], [])
                trail_lines[ai].set_3d_properties([])

            for edge_art, seg in zip(cube_lines[ai], cube_edge_segments(pos, rot, cube_size)):
                edge_art.set_data(seg[:, 0], seg[:, 1])
                edge_art.set_3d_properties(seg[:, 2])

            color = "darkorange" if ai == highlight_idx else "0.55"
            lw = 2.0 if ai == highlight_idx else 1.0
            lcd_quivers[ai].remove()
            lcd_quivers[ai] = ax.quiver(
                pos[0], pos[1], pos[2],
                lcd_dir[0], lcd_dir[1], lcd_dir[2],
                color=color,
                linewidth=lw,
                arrow_length_ratio=0.25,
                length=arrow_scale,
            )

        explore_dir = _unit_vector3(agent_highlight.get("MapExploreDirection"))
        explore_quivers[0].remove()
        if explore_dir is not None:
            pos, _ = agent_position_rotation(agent_highlight)
            explore_quivers[0] = ax.quiver(
                pos[0], pos[1], pos[2],
                explore_dir[0], explore_dir[1], explore_dir[2],
                color="cyan",
                linewidth=2.0,
                arrow_length_ratio=0.25,
                length=explore_arrow_scale,
            )
        else:
            explore_quivers[0] = ax.quiver(0, 0, 0, 0, 0, 0, color="cyan", length=0.0)

        patches = agent_highlight.get("MapCoveragePatches") or []
        ratio = (
            float(coverage_mod.compute_coverage_ratio(patches))
            if patches
            else float(agent_highlight.get("MapCoverageRatio", 0.0) or 0.0)
        )
        ellipsoid = agent_highlight.get("MapEllipsoid")
        n_covered = sum(1 for p in patches if getattr(p, "covered", False))
        if (
            frame_idx == 0
            or frame_idx == n_frames - 1
            or (frame_idx + 1) % log_step == 0
        ):
            print(
                f"[viz] frame {frame_idx + 1}/{n_frames} | "
                f"map_pts={len(map_pts)} | coverage={100.0 * ratio:.1f}% | "
                f"patches={len(patches)} ({n_covered} covered) | "
                f"ellipsoid={'yes' if ellipsoid is not None else 'no'} | "
                f"explore_dir={'yes' if explore_dir is not None else 'no'}"
            )

        title.set_text(
            f"Frame {frame_idx + 1}/{len(agents_history)} | agent {highlight} | "
            f"coverage {100.0 * ratio:.1f}% | map pts {len(map_pts)}"
        )

        artists = [map_scatter, explore_target_scatter, title, *trail_lines]
        artists.extend(ellipsoid_artists)
        for lines in cube_lines:
            artists.extend(lines)
        artists.extend(lcd_quivers)
        artists.extend(explore_quivers)
        artists.extend(patch_collections)
        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(agents_history),
        interval=int(args.interval_ms),
        repeat=False,
        blit=False,
    )

    output_path = args.output or (
        PROJECT_ROOT / "visualization" / "output" / f"coverage_agent{highlight}.gif"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save:
        n_frames = len(agents_history)
        print(f"[viz] Saving GIF ({n_frames} frames) -> {output_path}")

        def _save_progress(current: int, total: int) -> None:
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

    if matplotlib.get_backend().lower() == "agg":
        print(
            "[viz] WARNING: backend is still Agg — interactive window unavailable. "
            "Re-run with --save or install Tk/PyQt for plt.show()."
        )
        args.save = True
        output_path = args.output or (
            PROJECT_ROOT / "visualization" / "output" / f"coverage_agent{highlight}.gif"
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[viz] Falling back to GIF save -> {output_path}")
        ani.save(
            str(output_path),
            writer=animation.PillowWriter(fps=max(1, 1000 // int(args.interval_ms))),
        )
        print(f"[viz] Done. Wrote {output_path}")
        plt.close(fig)
        return output_path

    coverage_mod.enable_scroll_zoom(fig, ax)
    print("[viz] Opening interactive window (close it to exit)")
    plt.show()
    return ani


def main() -> None:
    args = parse_args()
    if INTERACTIVE_PREVIEW:
        args.save = False
        print("[viz] INTERACTIVE_PREVIEW=True — opening matplotlib window (no GIF save)")
    _ = run_animation(args)


if __name__ == "__main__":
    main()
