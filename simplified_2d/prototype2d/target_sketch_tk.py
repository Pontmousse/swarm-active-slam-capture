"""
Tkinter freehand target sketch.

- Saves an internal **lines draft** (world_bounds + sampled polylines).
- **Export target JSON** wires **traced samples** into `dense_points` (all sketch lines concatenated).
  Optional RDP/thinning; `contour_points` is a bbox placeholder for validators/animation hull;
  `attachment_points`: polyline edge midpoints along traces, thinned by `--attachment-edge-stride`.

Usage:
  python -m simplified_2d.prototype2d.target_sketch_tk --output target.json
  python -m simplified_2d.prototype2d.target_sketch_tk -o target.json --load sketch.json
  python -m simplified_2d.prototype2d.target_sketch_tk -o target.json --load rectangle_target.json
"""

from __future__ import annotations

import argparse
import json
import math
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .io import load_target_definition, save_target_definition, validate_target_definition
from .model import AttachmentPoint, TargetDefinition, TargetPoint

Point = Tuple[float, float]


def polygon_orientation(points: List[Point]) -> float:
    """Signed area proxy for polygon winding (used for outward dense normals)."""
    area = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        area += (x2 - x1) * (y2 + y1)
    return area


def generate_dense_points(
    contour: List[Point],
    spacing: float,
    start_id: int = 1000,
) -> List[TargetPoint]:
    """Sample ``TargetPoint``s along a closed polygon contour at roughly ``spacing`` edge length."""
    dense: List[TargetPoint] = []
    if len(contour) < 2:
        return dense

    orientation = polygon_orientation(contour)
    idx_counter = start_id
    for idx in range(len(contour)):
        x1, y1 = contour[idx]
        x2, y2 = contour[(idx + 1) % len(contour)]
        edge = np.array([x2 - x1, y2 - y1])
        length = float(np.linalg.norm(edge))
        if length < 1e-6:
            continue
        direction = edge / length
        steps = max(1, int(np.floor(length / spacing)))
        if orientation < 0:
            normal = np.array([direction[1], -direction[0]])
        else:
            normal = np.array([-direction[1], direction[0]])
        for step in range(steps):
            t = step / float(steps)
            px = x1 + t * edge[0]
            py = y1 + t * edge[1]
            dense.append(
                TargetPoint(
                    id=idx_counter,
                    x=float(px),
                    y=float(py),
                    normal=[float(normal[0]), float(normal[1])],
                )
            )
            idx_counter += 1
    return dense


@dataclass
class SketchLine:
    closed: bool
    points: List[Point]


LINE_PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def dedupe_consecutive(points: Sequence[Point], tol: float = 1e-9) -> List[Point]:
    out: List[Point] = []
    for q in points:
        qx, qy = float(q[0]), float(q[1])
        if out and math.hypot(out[-1][0] - qx, out[-1][1] - qy) < tol:
            continue
        out.append((qx, qy))
    return out


def _perpendicular_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    line = b - a
    line_len = float(np.linalg.norm(line))
    if line_len < 1e-12:
        return float(np.linalg.norm(p - a))
    cross_z = line[0] * (a[1] - p[1]) - line[1] * (a[0] - p[0])
    return float(abs(cross_z) / line_len)


def _rdp_indices(pts: np.ndarray, epsilon: float) -> List[int]:
    """Return indices of vertices kept by Ramer–Doug–Peucker (open polyline)."""
    n = len(pts)
    if n <= 2:
        return list(range(n))

    stack = [(0, n - 1)]
    keep = {0, n - 1}
    while stack:
        start, end = stack.pop()
        if end - start <= 1:
            continue
        seg_a, seg_b = pts[start], pts[end]
        dists = [_perpendicular_distance(pts[i], seg_a, seg_b) for i in range(start + 1, end)]
        if not dists:
            continue
        rel = int(np.argmax(dists)) + start + 1
        dmax = _perpendicular_distance(pts[rel], seg_a, seg_b)
        if dmax > epsilon:
            keep.add(rel)
            stack.append((start, rel))
            stack.append((rel, end))
    return sorted(keep)


def simplify_closed_ring(ring: List[Point], epsilon: float) -> List[Point]:
    """Reduce vertices on a closed polygon; epsilon in world units (0 = no simplification)."""
    r = dedupe_consecutive(ring)
    if len(r) > 1 and math.hypot(r[-1][0] - r[0][0], r[-1][1] - r[0][1]) < 1e-5:
        r = r[:-1]
    if len(r) < 3:
        return ring
    if epsilon <= 1e-12:
        return r
    open_chain = np.array(r + [r[0]], dtype=float)
    idx = _rdp_indices(open_chain, epsilon)
    simp = [tuple(open_chain[i]) for i in idx]
    if len(simp) > 1 and math.hypot(simp[0][0] - simp[-1][0], simp[0][1] - simp[-1][1]) < 1e-9:
        simp = simp[:-1]
    if len(simp) < 3:
        return r
    return simp


def simplify_open_polyline(pts: List[Point], epsilon: float) -> List[Point]:
    """RDP on an open chain (epsilon in world units; 0 = unchanged)."""
    if epsilon <= 1e-12 or len(pts) < 3:
        return pts
    arr = np.array(pts, dtype=float)
    idx = _rdp_indices(arr, epsilon)
    return [tuple(arr[i]) for i in idx]


def thin_polyline_points(pts: List[Point], min_dist: float) -> List[Point]:
    """Keep successive samples roughly ``min_dist`` apart along each polyline order; 0 = no thinning."""
    if min_dist <= 1e-12 or len(pts) <= 1:
        return pts
    out: List[Point] = [(float(pts[0][0]), float(pts[0][1]))]
    for p in pts[1:-1]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= min_dist:
            out.append((float(p[0]), float(p[1])))
    last = (float(pts[-1][0]), float(pts[-1][1]))
    if math.hypot(last[0] - out[-1][0], last[1] - out[-1][1]) >= 1e-9:
        if len(out) == 1 or math.hypot(last[0] - out[-1][0], last[1] - out[-1][1]) >= min_dist:
            out.append(last)
        else:
            out[-1] = last
    return dedupe_consecutive(out)


def prepare_polyline_landmarks(sk: SketchLine, simplify_epsilon: float, min_sample_spacing: float) -> List[Point]:
    """
    Canonical point sequence for perception landmarks: sketch samples optionally RDP-open/closed then thinned.

    Drops invalid closed strokes with fewer than 3 unique vertices; returns [] for degenerate strips.
    """
    pts = dedupe_consecutive(sk.points)
    if not pts:
        return []
    if sk.closed:
        if len(pts) > 1 and math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) < 1e-5:
            pts = pts[:-1]
        if len(pts) < 3:
            return []
        pts_list = simplify_closed_ring(pts, simplify_epsilon) if simplify_epsilon > 1e-12 else list(pts)
    else:
        if len(pts) < 2:
            return list(pts) if len(pts) == 1 else []
        pts_list = simplify_open_polyline(list(pts), simplify_epsilon) if simplify_epsilon > 1e-12 else list(pts)

    pts_list = thin_polyline_points(pts_list, min_sample_spacing)

    # Single-point open stroke is still a landmark
    return pts_list


def contour_corners_from_landmarks(dense: List[TargetPoint], *, pad_frac: float = 0.02) -> List[TargetPoint]:
    """Placeholder hull (axis-aligned bbox) so ``contour_points`` satisfies validators; viz only."""
    xs = [p.x for p in dense]
    ys = [p.y for p in dense]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = xmax - xmin
    dy = ymax - ymin
    pad = max(1e-4, pad_frac * max(dx, dy, 1e-6))
    corners = (
        (xmin - pad, ymin - pad),
        (xmax + pad, ymin - pad),
        (xmax + pad, ymax + pad),
        (xmin - pad, ymax + pad),
    )
    return [TargetPoint(id=i, x=float(x), y=float(y), normal=None) for i, (x, y) in enumerate(corners)]


def attachments_along_landmark_polylines(
    segments: List[Tuple[int, List[TargetPoint], bool]],
    *,
    edge_stride: int = 1,
    start_ap_id: int = 2000,
) -> List[AttachmentPoint]:
    """
    Attachment points at midpoints along consecutive traced landmarks (per sketch line).

    ``edge_stride``: every stride-th polyline segment gets an AP (`1` ≈ rectangular mid-edge coverage).
    ``closed`` strokes include the closing edge from last landmark back to the first.

    Landmark chosen is the nearest dense sample in **that sketch line chunk** only.
    Duplicate ``point_id`` values are skipped (same policy as contour-based export).
    """
    if edge_stride < 1:
        raise ValueError("attachment edge_stride must be >= 1.")

    attachments: List[AttachmentPoint] = []
    used_point_ids_set: set = set()
    aid = start_ap_id

    for sketch_idx, chunk, closed in segments:
        if len(chunk) < 2:
            continue
        n = len(chunk)
        if closed and n >= 3:
            edge_start_indices = list(range(0, n, edge_stride))
        elif not closed:
            num_edges = n - 1
            edge_start_indices = list(range(0, num_edges, edge_stride))
        else:
            continue

        for ei in edge_start_indices:
            if closed:
                idx_a = ei % n
                idx_b = (idx_a + 1) % n
            else:
                idx_a = ei
                idx_b = ei + 1
            pa, pb = chunk[idx_a], chunk[idx_b]
            edge = np.array([pb.x - pa.x, pb.y - pa.y])
            if float(np.linalg.norm(edge)) < 1e-12:
                continue
            mx = float((pa.x + pb.x) * 0.5)
            my = float((pa.y + pb.y) * 0.5)
            nearest = _nearest_dense_point(chunk, (mx, my))
            if nearest.id in used_point_ids_set:
                continue
            used_point_ids_set.add(nearest.id)
            attachments.append(
                AttachmentPoint(
                    id=aid,
                    point_id=nearest.id,
                    x=nearest.x,
                    y=nearest.y,
                    normal=list(nearest.normal) if nearest.normal is not None else None,
                    label=f"line_{sketch_idx}_edge_{idx_a}",
                )
            )
            aid += 1

    return attachments


def build_target_definition_from_sketched_lines(
    lines: List[SketchLine],
    *,
    name: str,
    simplify_epsilon: float,
    min_sample_spacing: float,
    attachment_edge_stride: int = 1,
) -> TargetDefinition:
    """
    Perception-facing export: concatenate traced vertices from each sketch segment into ``dense_points``.

    ``lines`` preserves canvas order (by default all sketch lines).
    ``contour_points`` is bbox corners derived from landmarks (thin preview hull in animation).
    """
    dense_points: List[TargetPoint] = []
    indexed_segments: List[Tuple[int, List[TargetPoint], bool]] = []
    id_counter = 1000

    for sketch_idx, sk in enumerate(lines):
        pts = prepare_polyline_landmarks(sk, simplify_epsilon, min_sample_spacing)
        if not pts:
            continue
        slice_pts: List[TargetPoint] = []
        for x, y in pts:
            tp = TargetPoint(id=id_counter, x=float(x), y=float(y), normal=None)
            dense_points.append(tp)
            slice_pts.append(tp)
            id_counter += 1
        if slice_pts:
            indexed_segments.append((sketch_idx, slice_pts, sk.closed))

    if not dense_points:
        raise ValueError("No landmarks to export (check lines after simplify/thin thresholds).")

    contour_points = contour_corners_from_landmarks(dense_points)
    attachment_points = attachments_along_landmark_polylines(
        indexed_segments,
        edge_stride=attachment_edge_stride,
    )
    return TargetDefinition(
        name=name,
        contour_points=contour_points,
        dense_points=dense_points,
        attachment_points=attachment_points,
    )


def _nearest_dense_point(dense: List[TargetPoint], pos: Point) -> TargetPoint:
    arr = np.array([[p.x, p.y] for p in dense], dtype=float)
    delta = arr - np.array(pos, dtype=float)
    i = int(np.argmin(np.linalg.norm(delta, axis=1)))
    return dense[i]


def build_attachments_for_contour(
    contour: List[Point],
    dense_points: List[TargetPoint],
    start_ap_id: int = 2000,
) -> List[AttachmentPoint]:
    """One attachment per contour edge, snapped to nearest dense sample (like rectangle mid-edges)."""
    if len(contour) < 3 or not dense_points:
        return []
    attachments: List[AttachmentPoint] = []
    used_point_ids: set = set()
    ap_id = start_ap_id
    n = len(contour)
    for i in range(n):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % n]
        edge = np.array([x2 - x1, y2 - y1])
        if float(np.linalg.norm(edge)) < 1e-9:
            continue
        mx = float((x1 + x2) * 0.5)
        my = float((y1 + y2) * 0.5)
        nearest = _nearest_dense_point(dense_points, (mx, my))
        if nearest.id in used_point_ids:
            continue
        used_point_ids.add(nearest.id)
        attachments.append(
            AttachmentPoint(
                id=ap_id,
                point_id=nearest.id,
                x=nearest.x,
                y=nearest.y,
                normal=list(nearest.normal) if nearest.normal is not None else None,
                label=f"edge_{i}",
            )
        )
        ap_id += 1
    return attachments


def build_target_definition_from_polygon(
    contour: List[Point],
    *,
    name: str,
    spacing: float,
    simplify_epsilon: float,
) -> TargetDefinition:
    contour_ring = simplify_closed_ring(contour, simplify_epsilon)
    if len(contour_ring) < 3:
        raise ValueError("Contour must have at least 3 vertices after simplification.")
    dense_points = generate_dense_points(contour_ring, spacing, start_id=1000)
    if not dense_points:
        raise ValueError("Dense boundary is empty; try a larger --dense-spacing.")
    contour_points = [
        TargetPoint(id=i, x=pt[0], y=pt[1], normal=None) for i, pt in enumerate(contour_ring)
    ]
    attachment_points = build_attachments_for_contour(contour_ring, dense_points)
    return TargetDefinition(
        name=name,
        contour_points=contour_points,
        dense_points=dense_points,
        attachment_points=attachment_points,
    )


def sketch_lines_from_target(target: TargetDefinition) -> List[SketchLine]:
    pts = [(float(p.x), float(p.y)) for p in target.contour_points]
    return [SketchLine(closed=len(pts) >= 3, points=pts)]


def load_sketch_or_target(path: str) -> Tuple[str, Optional[dict], List[SketchLine]]:
    """Load a lines-draft JSON or an existing simulator target JSON into sketch lines."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if raw.get("lines"):
        meta, lines = sketch_from_dict(raw)
        return meta.get("name", "sketch"), raw.get("world_bounds"), lines
    if raw.get("contour_points"):
        target = load_target_definition(path)
        return target.name, None, sketch_lines_from_target(target)
    raise ValueError(f"Unrecognized JSON in {path}: need 'lines' or 'contour_points'.")


def sketch_from_dict(data: dict) -> Tuple[dict, List[SketchLine]]:
    wb = data.get("world_bounds", {})
    bounds = {
        "xmin": float(wb.get("xmin", -1.0)),
        "xmax": float(wb.get("xmax", 1.0)),
        "ymin": float(wb.get("ymin", -1.0)),
        "ymax": float(wb.get("ymax", 1.0)),
    }
    meta = {"name": data.get("name", "sketch"), "units": data.get("units", "world")}
    lines_out: List[SketchLine] = []
    for entry in data.get("lines", []):
        pts = [(float(p[0]), float(p[1])) for p in entry.get("points", [])]
        lines_out.append(SketchLine(bool(entry.get("closed", False)), pts))
    return {**meta, "world_bounds": bounds}, lines_out


def save_sketch_json(path: str, doc: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def load_sketch_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class SketchApp:
    PAD = 12

    def __init__(
        self,
        root: tk.Tk,
        output_path: str,
        sketch_draft_path: str,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        canvas_w: int,
        canvas_h: int,
        min_step_world: float,
        snap_tol_world: float,
        dense_spacing: float,
        simplify_epsilon: float,
        export_line_index: int,
        attachment_edge_stride: int,
        initial_lines: Optional[List[SketchLine]] = None,
        sketch_name: str = "sketch",
    ) -> None:
        self.root = root
        self.output_path = output_path
        self.sketch_draft_path = sketch_draft_path
        self.sketch_name = sketch_name
        self.dense_spacing = dense_spacing
        self.simplify_epsilon = simplify_epsilon
        self.export_line_index = export_line_index
        self.attachment_edge_stride = attachment_edge_stride
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.min_step_world = min_step_world
        self.snap_tol_world = snap_tol_world
        self.lines: List[SketchLine] = list(initial_lines) if initial_lines else []

        self._drawing = False
        self._current_world: List[Point] = []
        self._last_c: Tuple[int, int] = (0, 0)

        root.title("Target sketch — drag to trace, release to finish line")

        frm = tk.Frame(root)
        frm.pack(side=tk.TOP, fill=tk.X)

        tk.Button(frm, text="Close last line", command=self.close_last_line).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(frm, text="Undo last line", command=self.undo_line).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(frm, text="Clear all", command=self.clear_all).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(frm, text="Export simulator target", command=self.export_simulator_target).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(frm, text="Save lines draft", command=self.save_lines_draft).pack(side=tk.LEFT, padx=2, pady=4)

        self.status = tk.StringVar(
            value="Export packs every sketch segment into dense_points (what agents perceive). "
            "Optional --dense-spacing thin & --export-line for a subset. Hull outline is bbox only."
        )
        tk.Label(root, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)

        self.canvas = tk.Canvas(root, width=canvas_w, height=canvas_h, bg="#f8f9fa", highlightthickness=1)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Configure>", self.on_configure)

        self._suppress_configure = False
        self._redraw_full()

    def _line_color(self, index: int) -> str:
        return LINE_PALETTE[index % len(LINE_PALETTE)]

    def canvas_to_world(self, cx: int, cy: int) -> Point:
        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)
        inner_w = max(w - 2 * self.PAD, 1e-6)
        inner_h = max(h - 2 * self.PAD, 1e-6)
        nx = (cx - self.PAD) / inner_w
        ny = (cy - self.PAD) / inner_h
        wx = self.xmin + nx * (self.xmax - self.xmin)
        wy = self.ymax - ny * (self.ymax - self.ymin)
        return (wx, wy)

    def world_to_canvas(self, wx: float, wy: float) -> Tuple[float, float]:
        w = max(self.canvas.winfo_width(), 1)
        h = max(self.canvas.winfo_height(), 1)
        inner_w = max(w - 2 * self.PAD, 1e-6)
        inner_h = max(h - 2 * self.PAD, 1e-6)
        nx = (wx - self.xmin) / max(self.xmax - self.xmin, 1e-9)
        ny = (self.ymax - wy) / max(self.ymax - self.ymin, 1e-9)
        return self.PAD + nx * inner_w, self.PAD + ny * inner_h

    def set_status(self, msg: str) -> None:
        self.status.set(msg)

    def on_configure(self, _event: tk.Event) -> None:
        if getattr(self, "_suppress_configure", False):
            return
        self._redraw_full()

    def on_press(self, event: tk.Event) -> None:
        widget = getattr(event, "widget", None)
        if widget is not self.canvas:
            return
        self._drawing = True
        self.canvas.delete("scratch")
        wx, wy = self.canvas_to_world(event.x, event.y)
        self._current_world = [(wx, wy)]
        self._last_c = (event.x, event.y)
        ix, iy = self.world_to_canvas(wx, wy)
        self.canvas.create_oval(ix - 2, iy - 2, ix + 2, iy + 2, fill=self._line_color(len(self.lines)), outline="", tags="scratch")

    def on_motion(self, event: tk.Event) -> None:
        if not self._drawing:
            return
        wx, wy = self.canvas_to_world(event.x, event.y)
        lx, ly = self._current_world[-1]
        if math.hypot(wx - lx, wy - ly) < self.min_step_world:
            return
        self._current_world.append((wx, wy))
        x0, y0 = self._last_c
        self.canvas.create_line(x0, y0, event.x, event.y, fill=self._line_color(len(self.lines)), width=2, capstyle=tk.ROUND, tags="scratch")
        self._last_c = (event.x, event.y)

    def on_release(self, event: tk.Event) -> None:
        if not self._drawing:
            return
        self._drawing = False
        self.canvas.delete("scratch")
        pts = list(self._current_world)
        self._current_world = []
        if len(pts) < 2:
            self.set_status("Ignored: need a short drag (at least 2 samples).")
            return
        closed = False
        if len(pts) >= 3:
            d = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
            if d <= self.snap_tol_world:
                closed = True
        self.lines.append(SketchLine(closed=closed, points=pts))
        self.set_status(f"Line {len(self.lines)} added ({'closed' if closed else 'open'}). Total lines: {len(self.lines)}")
        self._redraw_full()

    def close_last_line(self) -> None:
        if not self.lines:
            self.set_status("No line to close.")
            return
        last = self.lines[-1]
        if len(last.points) < 3:
            self.set_status("Need at least 3 points to close.")
            return
        last.closed = True
        self.set_status(f"Line {len(self.lines)} marked closed.")
        self._redraw_full()

    def undo_line(self) -> None:
        if not self.lines:
            self.set_status("Nothing to undo.")
            return
        self.lines.pop()
        self.set_status(f"Removed last line. Remaining: {len(self.lines)}")
        self._redraw_full()

    def clear_all(self) -> None:
        self.lines.clear()
        self.canvas.delete("scratch")
        self.set_status("Cleared all lines.")
        self._redraw_full()

    def _redraw_full(self) -> None:
        self._suppress_configure = True
        self.canvas.delete("sketch")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self.canvas.create_rectangle(self.PAD, self.PAD, w - self.PAD, h - self.PAD, outline="#cccccc", width=1, dash=(4, 4), tags="sketch")

        def draw_line(sk: SketchLine, color_idx: int) -> None:
            cc = [(self.world_to_canvas(p[0], p[1])) for p in sk.points]
            if len(cc) < 2:
                return
            for i in range(len(cc) - 1):
                x0, y0 = cc[i]
                x1, y1 = cc[i + 1]
                self.canvas.create_line(x0, y0, x1, y1, fill=self._line_color(color_idx), width=2, capstyle=tk.ROUND, tags="sketch")
            if sk.closed and len(cc) >= 3:
                x0, y0 = cc[-1]
                x1, y1 = cc[0]
                self.canvas.create_line(x0, y0, x1, y1, fill=self._line_color(color_idx), width=2, capstyle=tk.ROUND, tags="sketch")
            for x, y in cc:
                self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=self._line_color(color_idx), outline="", tags="sketch")

        for idx, ln in enumerate(self.lines):
            draw_line(ln, idx)
        self._suppress_configure = False

    def document(self) -> dict:
        wb = {"xmin": self.xmin, "xmax": self.xmax, "ymin": self.ymin, "ymax": self.ymax}
        return {
            "name": self.sketch_name,
            "units": "world",
            "world_bounds": wb,
            "lines": [{"closed": ln.closed, "points": [[p[0], p[1]] for p in ln.points]} for ln in self.lines],
        }

    def save_lines_draft(self) -> None:
        doc = self.document()
        save_sketch_json(self.sketch_draft_path, doc)
        self.set_status(f"Saved lines draft ({len(self.lines)} line(s)) -> {self.sketch_draft_path}")

    def _lines_for_export(self) -> Optional[List[SketchLine]]:
        if self.export_line_index >= 0:
            if self.export_line_index < len(self.lines):
                return [self.lines[self.export_line_index]]
            self.set_status(f"No line at index {self.export_line_index}.")
            return None
        if not self.lines:
            self.set_status("Nothing to export; draw at least one line.")
            return None
        return list(self.lines)

    def export_simulator_target(self) -> None:
        lines_subset = self._lines_for_export()
        if lines_subset is None:
            return
        try:
            target = build_target_definition_from_sketched_lines(
                lines_subset,
                name=self.sketch_name,
                simplify_epsilon=self.simplify_epsilon,
                min_sample_spacing=self.dense_spacing,
                attachment_edge_stride=self.attachment_edge_stride,
            )
        except ValueError as err:
            self.set_status(f"Export failed: {err}")
            return
        errors = validate_target_definition(target)
        if errors:
            self.set_status("Validation: " + "; ".join(errors))
            return
        save_target_definition(self.output_path, target)
        self.set_status(
            f"Exported {len(target.dense_points)} landmark(s), {len(target.attachment_points)} attachment(s) -> {self.output_path}"
        )


def run_app(args: argparse.Namespace) -> None:
    xmin, xmax = args.xmin, args.xmax
    ymin, ymax = args.ymin, args.ymax
    meta_name = "sketch"
    if args.load:
        try:
            meta_name, wb, lines_read = load_sketch_or_target(args.load)
        except (OSError, ValueError, KeyError) as err:
            raise SystemExit(f"Failed to load {args.load}: {err}") from err
        if wb:
            xmin = float(wb.get("xmin", xmin))
            xmax = float(wb.get("xmax", xmax))
            ymin = float(wb.get("ymin", ymin))
            ymax = float(wb.get("ymax", ymax))
    else:
        lines_read = None

    draft_path = args.lines_draft or str(Path(args.output).with_name(Path(args.output).stem + "_lines.json"))

    root = tk.Tk()
    SketchApp(
        root,
        output_path=args.output,
        sketch_draft_path=draft_path,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        canvas_w=args.canvas_width,
        canvas_h=args.canvas_height,
        min_step_world=args.min_step,
        snap_tol_world=args.snap_tol,
        dense_spacing=args.dense_spacing,
        simplify_epsilon=args.simplify_epsilon,
        export_line_index=args.export_line,
        attachment_edge_stride=args.attachment_edge_stride,
        initial_lines=lines_read or [],
        sketch_name=meta_name,
    )
    root.mainloop()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Tkinter freehand sketch; export simulator target JSON (contour/dense/attachments)"
    )
    p.add_argument("--output", "-o", required=True, help="Target JSON path for Export simulator target")
    p.add_argument("--load", default=None, help="Lines draft or existing target JSON to edit")
    p.add_argument(
        "--lines-draft",
        default=None,
        metavar="PATH",
        help="Lines draft JSON path (default: <output_stem>_lines.json)",
    )
    p.add_argument(
        "--dense-spacing",
        type=float,
        default=0.0,
        metavar="WORLD",
        help="Extra spacing between successive exported landmarks along each traced line (0 = keep sketch sampling)",
    )
    p.add_argument(
        "--simplify-epsilon",
        type=float,
        default=0.0,
        metavar="WORLD",
        help="RDP simplification per sketch line (0 = keep traced vertices)",
    )
    p.add_argument(
        "--export-line",
        type=int,
        default=-1,
        metavar="N",
        help="Export only sketch line index N; default -1 exports every line segment",
    )
    p.add_argument(
        "--attachment-edge-stride",
        type=int,
        default=1,
        metavar="K",
        help="Place attachments every K polyline segments (edge mid→nearest landmark); 1 = densest sampling",
    )
    p.add_argument("--xmin", type=float, default=-1.0)
    p.add_argument("--xmax", type=float, default=1.0)
    p.add_argument("--ymin", type=float, default=-1.0)
    p.add_argument("--ymax", type=float, default=1.0)
    p.add_argument("--canvas-width", type=int, default=720)
    p.add_argument("--canvas-height", type=int, default=720)
    p.add_argument(
        "--min-step",
        type=float,
        default=0.02,
        metavar="WORLD",
        help="Minimum spacing between samples along a drag (world units)",
    )
    p.add_argument(
        "--snap-tol",
        type=float,
        default=0.08,
        metavar="WORLD",
        help="Release within this distance of start marks the polyline closed (≥3 points)",
    )
    args = p.parse_args()
    if args.xmin >= args.xmax or args.ymin >= args.ymax:
        raise SystemExit("Invalid world bounds (min must be < max).")
    if args.attachment_edge_stride < 1:
        raise SystemExit("--attachment-edge-stride must be >= 1.")

    run_app(args)


if __name__ == "__main__":
    main()
