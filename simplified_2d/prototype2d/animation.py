"""
Animation helper for saved runs.

Usage:
  python -m simplified_2d.prototype2d.animation --results path/to/results --target path/to/target.json
  python -m simplified_2d.prototype2d.animation --results path/to/results --target path/to/target.json --margin 0.15
  python -m simplified_2d.prototype2d.animation ... --highlight-agent 1   # 1 = first agent in history list

Requires ``target_history.pkl`` in ``results_dir`` so the moving target aligns with simulation (auto-saved by the simulator).
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .io import load_target_definition
from .simulator import _target_points_world


def _load_history(results_dir: str, name: str):
    path = os.path.join(results_dir, name)
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _body_points_from_target(definition) -> Tuple[List[Dict], List[Dict]]:
    contour = [{"id": pt.id, "x": pt.x, "y": pt.y, "normal": pt.normal} for pt in definition.contour_points]
    dense = [{"id": pt.id, "x": pt.x, "y": pt.y, "normal": pt.normal} for pt in definition.dense_points]
    return contour, dense


def _dense_id_to_xy_world(dense_world: List[Dict]) -> Dict[int, Tuple[float, float]]:
    """Map dense landmark id -> world (x, y) for this frame."""
    out: Dict[int, Tuple[float, float]] = {}
    for p in dense_world:
        pos = p["pos"]
        out[int(p["id"])] = (float(pos[0]), float(pos[1]))
    return out


def _dense_id_to_xy_body(dense_dicts: List[Dict]) -> Dict[int, Tuple[float, float]]:
    """Body-frame dense positions when target motion history is unavailable."""
    return {int(d["id"]): (float(d["x"]), float(d["y"])) for d in dense_dicts}


def _compute_fixed_bounds(
    agents_history: List[List[Dict]],
    attachment_history: List[List[Dict]],
    target_history: Optional[List[Dict]],
    target_def,
    contour_dicts: List[Dict],
    dense_dicts: List[Dict],
    margin_frac: float = 0.12,
    min_pad: float = 0.35,
) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []

    def add_xy(px: float, py: float) -> None:
        xs.append(px)
        ys.append(py)

    n_frames = min(len(agents_history), len(attachment_history))

    target_states: List[List[float]] = []
    if target_history is not None and len(target_history) >= n_frames:
        target_states = [row["state"] for row in target_history[:n_frames]]

    for frame in range(n_frames):
        for agent in agents_history[frame]:
            add_xy(agent["state"][0], agent["state"][1])
        for ap in attachment_history[frame]:
            add_xy(ap["pos"][0], ap["pos"][1])

        if target_def is not None and frame < len(target_states):
            ts = target_states[frame]
            if dense_dicts:
                for p in _target_points_world(ts, dense_dicts):
                    add_xy(float(p["pos"][0]), float(p["pos"][1]))
            if contour_dicts and len(contour_dicts) >= 3:
                for p in _target_points_world(ts, contour_dicts):
                    add_xy(float(p["pos"][0]), float(p["pos"][1]))

    if not xs:
        return -3.0, 3.0, -3.0, 3.0

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span_x = xmax - xmin
    span_y = ymax - ymin
    span = max(span_x, span_y, min_pad)
    pad = max(margin_frac * span, min_pad)

    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad
    # keep square-ish extents for nicer overview (equal aspect)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    half = max(xmax - xmin, ymax - ymin) / 2.0
    return cx - half, cx + half, cy - half, cy + half


def animate_run(
    results_dir: str,
    target_path: Optional[str],
    interval_ms: int,
    margin_frac: float = 0.12,
    highlight_agent: Optional[int] = None,
) -> None:
    agents_history = _load_history(results_dir, "agents_history.pkl")
    attachment_history = _load_history(results_dir, "attachment_points.pkl")
    target_history_path = os.path.join(results_dir, "target_history.pkl")
    target_history = None
    if os.path.isfile(target_history_path):
        target_history = _load_history(results_dir, "target_history.pkl")

    target_def = load_target_definition(target_path) if target_path else None
    contour_dicts: List[Dict] = []
    dense_dicts: List[Dict] = []
    if target_def is not None:
        contour_dicts, dense_dicts = _body_points_from_target(target_def)

    if target_history is None and target_def is not None:
        print(
            "[animation] warning: missing target_history.pkl — target shape will stay in body frame at origin.",
        )

    highlight_idx: Optional[int] = None
    if highlight_agent is not None:
        if highlight_agent < 1:
            print("[animation] warning: --highlight-agent must be >= 1 (1-based); ignoring.")
        else:
            highlight_idx = highlight_agent - 1
            n0 = len(agents_history[0]) if agents_history else 0
            if highlight_idx >= n0:
                print(
                    f"[animation] warning: --highlight-agent={highlight_agent} out of range "
                    f"(only {n0} agent(s)); ignoring.",
                )
                highlight_idx = None

    xmin, xmax, ymin, ymax = _compute_fixed_bounds(
        agents_history,
        attachment_history,
        target_history,
        target_def,
        contour_dicts,
        dense_dicts,
        margin_frac=margin_frac,
    )
    xlim = (xmin, xmax)
    ylim = (ymin, ymax)

    fig, ax = plt.subplots(figsize=(6, 6))

    target_states: List[List[float]] = []
    if target_history is not None:
        target_states = [row["state"] for row in target_history]

    def update(frame):
        ax.clear()

        agents = agents_history[frame]
        all_x = [float(a["state"][0]) for a in agents]
        all_y = [float(a["state"][1]) for a in agents]
        if highlight_idx is not None and highlight_idx < len(agents):
            ox = [all_x[i] for i in range(len(agents)) if i != highlight_idx]
            oy = [all_y[i] for i in range(len(agents)) if i != highlight_idx]
            if ox:
                ax.scatter(ox, oy, c="black", s=36, label="agents", zorder=4)
            else:
                ax.scatter([], [], c="black", s=36, label="agents", zorder=4)
            ax.scatter(
                [all_x[highlight_idx]],
                [all_y[highlight_idx]],
                c="tab:orange",
                s=72,
                marker="s",
                edgecolors="black",
                linewidths=0.8,
                label=f"agent {highlight_agent} (highlight)",
                zorder=6,
            )
        else:
            ax.scatter(all_x, all_y, c="black", s=36, label="agents", zorder=4)

        attachments = attachment_history[frame]
        if attachments:
            ax.scatter(
                [ap["pos"][0] for ap in attachments],
                [ap["pos"][1] for ap in attachments],
                c="tab:red",
                s=52,
                label="attachment pts",
                zorder=5,
            )

        if target_def is not None and frame < len(target_states):
            ts = target_states[frame]
            if dense_dicts:
                world_dense = _target_points_world(ts, dense_dicts)
                ax.scatter(
                    [float(p["pos"][0]) for p in world_dense],
                    [float(p["pos"][1]) for p in world_dense],
                    c="tab:cyan",
                    s=14,
                    alpha=0.75,
                    label="target dense",
                    zorder=2,
                )
                if highlight_idx is not None and highlight_idx < len(agents):
                    agent_h = agents[highlight_idx]
                    id_to_xy = _dense_id_to_xy_world(world_dense)
                    map_obj = agent_h.get("map") or {}
                    mx, my = [], []
                    for pid in map_obj.keys():
                        pt = id_to_xy.get(int(pid))
                        if pt is not None:
                            mx.append(pt[0])
                            my.append(pt[1])
                    if mx:
                        ax.scatter(
                            mx,
                            my,
                            c="darkcyan",
                            s=28,
                            alpha=0.95,
                            edgecolors="teal",
                            linewidths=0.35,
                            label=f"agent {highlight_agent} map",
                            zorder=3,
                        )
            if contour_dicts and len(contour_dicts) >= 3:
                world_contour = _target_points_world(ts, contour_dicts)
                contour_xy = np.array([[float(p["pos"][0]), float(p["pos"][1])] for p in world_contour])
                ax.plot(
                    np.append(contour_xy[:, 0], contour_xy[0, 0]),
                    np.append(contour_xy[:, 1], contour_xy[0, 1]),
                    color="dimgray",
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.9,
                    label="target hull",
                    zorder=3,
                )
        elif target_def is not None and contour_dicts and len(contour_dicts) >= 3:
            # Fallback: draw body-fixed outline at origin (no motion)
            contour = np.array([[pt["x"], pt["y"]] for pt in contour_dicts])
            ax.plot(
                np.append(contour[:, 0], contour[0, 0]),
                np.append(contour[:, 1], contour[0, 1]),
                "k--",
                alpha=0.4,
                label="target body (static)",
                zorder=3,
            )
            if dense_dicts:
                ax.scatter(
                    [pt["x"] for pt in dense_dicts],
                    [pt["y"] for pt in dense_dicts],
                    c="tab:cyan",
                    s=14,
                    alpha=0.75,
                    label="target dense (static)",
                    zorder=2,
                )
                if highlight_idx is not None and highlight_idx < len(agents):
                    agent_h = agents[highlight_idx]
                    id_to_xy = _dense_id_to_xy_body(dense_dicts)
                    map_obj = agent_h.get("map") or {}
                    mx, my = [], []
                    for pid in map_obj.keys():
                        pt = id_to_xy.get(int(pid))
                        if pt is not None:
                            mx.append(pt[0])
                            my.append(pt[1])
                    if mx:
                        ax.scatter(
                            mx,
                            my,
                            c="darkcyan",
                            s=28,
                            alpha=0.95,
                            edgecolors="teal",
                            linewidths=0.35,
                            label=f"agent {highlight_agent} map",
                            zorder=3,
                        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Frame {frame}/{len(agents_history) - 1}")
        ax.legend(loc="upper right")
        ax.grid(False)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(agents_history),
        interval=interval_ms,
        repeat=False,
    )
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate a saved run")
    parser.add_argument("--results", required=True, help="Results directory")
    parser.add_argument("--target", default=None, help="Optional target JSON path")
    parser.add_argument("--interval", type=int, default=100, help="Frame interval in ms")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.12,
        help="Extra padding as fraction of max span around all trajectories (default 0.12)",
    )
    parser.add_argument(
        "--highlight-agent",
        type=int,
        default=None,
        metavar="N",
        help="1-based index into the per-frame agent list: draw that agent as a square and overlay its known landmark map (darker cyan). Omit for no highlight.",
    )
    args = parser.parse_args()
    animate_run(
        args.results,
        args.target,
        args.interval,
        margin_frac=args.margin,
        highlight_agent=args.highlight_agent,
    )


if __name__ == "__main__":
    main()

