"""
Plane segmentation -> simple contact point candidate generation.

Pipeline:
    mock_data.py satellite point cloud
        -> plane_ransac.py repeated RANSAC plane extraction
        -> plane-frame 2D bounding box
        -> rectangular grid subdivision
        -> occupied cell centers lifted back to 3D
        -> contact point candidates

This is still simple:
    - no contact feasibility scoring yet
"""

from dataclasses import dataclass
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, QhullError, cKDTree


try:
    from .mock_data import generate_mock_satellite_point_cloud
except ImportError:
    from mock_data import generate_mock_satellite_point_cloud


try:
    from .plane_ransac import (
        PlaneSegment,
        segment_planes_ransac,
        make_plane_basis,
    )
except ImportError:
    from plane_ransac import (
        PlaneSegment,
        segment_planes_ransac,
        make_plane_basis,
    )


# ============================================================
# Data structures
# ============================================================

@dataclass
class ContactPointCandidate:
    cp_id: int
    position: np.ndarray
    normal: np.ndarray
    parent_segment_id: int
    confidence: float
    area_support: float


# ============================================================
# Geometry helpers
# ============================================================

def project_points_to_plane_frame(points, center, normal):
    """
    Project 3D points into the local 2D coordinate frame of a plane.

    Returns:
        xy:
            shape (N, 2), 2D coordinates in the plane frame

        u, v:
            3D basis vectors spanning the plane
    """
    u, v = make_plane_basis(normal)

    rel = points - center

    x = rel @ u
    y = rel @ v

    xy = np.column_stack([x, y])

    return xy, u, v


def project_point_to_plane(point, plane_equation):
    """
    Project a 3D point onto the exact plane equation:

        ax + by + cz + d = 0
    """
    normal = plane_equation[:3]
    d = plane_equation[3]

    signed_distance = np.dot(normal, point) + d

    projected = point - signed_distance * normal

    return projected


# ============================================================
# Contact point candidate generation
# ============================================================

def convex_hull_polygon_xy(xy):
    """
    Return the convex in-plane support boundary of a plane segment.
    """
    if len(xy) < 3:
        return None

    try:
        hull = ConvexHull(xy)
    except QhullError:
        return None

    return xy[hull.vertices]


def polygon_area_xy(polygon_xy):
    x = polygon_xy[:, 0]
    y = polygon_xy[:, 1]

    return 0.5 * float(abs(
        np.dot(x, np.roll(y, -1)) -
        np.dot(y, np.roll(x, -1))
    ))


def lift_plane_xy_to_world(xy, center, u, v, plane_equation):
    points = np.array([
        center + xy_i[0] * u + xy_i[1] * v
        for xy_i in xy
    ])

    return np.array([
        project_point_to_plane(point, plane_equation)
        for point in points
    ])


def sample_points_in_convex_polygon(polygon_xy, spacing, boundary_margin=0.0):
    """
    Sample candidate centers inside a convex support polygon.

    A staggered grid gives stable density control without assuming that the
    segment itself is rectangular.
    """
    xmin, ymin = polygon_xy.min(axis=0)
    xmax, ymax = polygon_xy.max(axis=0)

    if spacing <= 0:
        raise ValueError("spacing must be positive")

    row_step = spacing * np.sqrt(3.0) / 2.0
    path = Path(polygon_xy)
    samples = []

    y = ymin
    row = 0

    while y <= ymax:
        x_offset = 0.5 * spacing if row % 2 else 0.0
        x = xmin + x_offset

        while x <= xmax:
            candidate = np.array([x, y])

            if path.contains_point(candidate, radius=-boundary_margin):
                samples.append(candidate)

            x += spacing

        y += row_step
        row += 1

    if len(samples) == 0:
        samples.append(polygon_xy.mean(axis=0))

    return np.array(samples)


def generate_contact_points_from_plane(
    plane_segment: PlaneSegment,
    contact_spacing=0.5,
    min_points_per_candidate=8,
    support_radius=None,
    boundary_margin=0.0,
):
    """
    Generate contact point candidates from one finite plane segment.

    Steps:
        1. Project plane inlier points into local 2D plane frame.
        2. Compute the convex in-plane support boundary.
        3. Sample candidate centers inside that boundary.
        4. Keep only candidates with enough nearby inlier support.
        5. Lift cell centers back to 3D.

    Notes:
        - contact_spacing controls final contact-point density.
        - boundary_margin keeps candidates away from the convex support edge.
        - The finite plane boundary is the convex hull of the segment inliers.
    """
    pts = plane_segment.points
    center = plane_segment.center
    normal = plane_segment.normal

    xy, u, v = project_points_to_plane_frame(pts, center, normal)
    polygon_xy = convex_hull_polygon_xy(xy)

    if polygon_xy is None:
        return []

    sample_xy = sample_points_in_convex_polygon(
        polygon_xy=polygon_xy,
        spacing=contact_spacing,
        boundary_margin=boundary_margin,
    )

    if support_radius is None:
        support_radius = 0.75 * contact_spacing

    tree = cKDTree(xy)
    candidates = []
    raw_support_counts = []
    area_per_candidate = polygon_area_xy(polygon_xy) / max(len(sample_xy), 1)

    for sample in sample_xy:
        support_ids = tree.query_ball_point(sample, r=support_radius)
        support_count = len(support_ids)

        if support_count < min_points_per_candidate:
            continue

        position = lift_plane_xy_to_world(
            sample[None, :],
            center=center,
            u=u,
            v=v,
            plane_equation=plane_segment.plane_equation,
        )[0]

        candidates.append({
            "position": position,
            "normal": normal.copy(),
            "parent_segment_id": plane_segment.segment_id,
            "area_support": area_per_candidate,
            "support_count": support_count,
        })

        raw_support_counts.append(support_count)

    if len(candidates) == 0:
        return []

    max_count = max(raw_support_counts)

    contact_points = []

    for local_id, item in enumerate(candidates):
        confidence = item["support_count"] / max_count

        cp = ContactPointCandidate(
            cp_id=-1,  # assigned globally later
            position=item["position"],
            normal=item["normal"],
            parent_segment_id=item["parent_segment_id"],
            confidence=float(confidence),
            area_support=float(item["area_support"]),
        )

        contact_points.append(cp)

    return contact_points


def generate_contact_points_from_segments(
    plane_segments,
    contact_spacing=0.5,
    min_points_per_candidate=8,
    support_radius=None,
    boundary_margin=0.0,
):
    """
    Generate contact point candidates for all detected plane segments.
    """
    all_candidates = []

    cp_id = 0

    for segment in plane_segments:
        candidates = generate_contact_points_from_plane(
            segment,
            contact_spacing=contact_spacing,
            min_points_per_candidate=min_points_per_candidate,
            support_radius=support_radius,
            boundary_margin=boundary_margin,
        )

        for cp in candidates:
            cp.cp_id = cp_id
            all_candidates.append(cp)
            cp_id += 1

    return all_candidates


# ============================================================
# Visualization
# ============================================================

def select_demo_observed_points(
    full_points: np.ndarray,
    side: str = "x_pos",
) -> np.ndarray:
    """
    Select a partial point cloud for the demo.

    This emulates a partial map at one instant of active SLAM.
    """
    if side == "x_pos":
        mask = full_points[:, 0] > 0.0
    elif side == "x_neg":
        mask = full_points[:, 0] < 0.0
    elif side == "y_pos":
        mask = full_points[:, 1] > 0.0
    elif side == "y_neg":
        mask = full_points[:, 1] < 0.0
    elif side == "z_pos":
        mask = full_points[:, 2] > 0.0
    elif side == "z_neg":
        mask = full_points[:, 2] < 0.0
    else:
        raise ValueError(f"Unknown side: {side}")

    observed_points = full_points[mask]

    if observed_points.shape[0] < 10:
        raise ValueError("Too few observed points selected.")

    return observed_points


def set_axes_equal(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def compute_plane_support_polygon(plane_segment):
    """
    Return the finite convex support polygon for one plane segment.
    """
    xy, u, v = project_points_to_plane_frame(
        plane_segment.points,
        plane_segment.center,
        plane_segment.normal,
    )
    polygon_xy = convex_hull_polygon_xy(xy)

    if polygon_xy is None:
        return None

    return lift_plane_xy_to_world(
        polygon_xy,
        center=plane_segment.center,
        u=u,
        v=v,
        plane_equation=plane_segment.plane_equation,
    )


def compute_contact_cell_polygon(contact_point, plane_segment, contact_spacing):
    """
    Return a small in-plane hex cell around one contact candidate.
    """
    u, v = make_plane_basis(plane_segment.normal)
    radius = 0.45 * contact_spacing

    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)

    polygon = np.array([
        contact_point.position +
        radius * np.cos(angle) * u +
        radius * np.sin(angle) * v
        for angle in angles
    ])

    return np.array([
        project_point_to_plane(point, plane_segment.plane_equation)
        for point in polygon
    ])


def style_clean_3d_axes(ax):
    """
    Remove background grid and panes so finite plane subsegments are easier to see.
    """
    ax.grid(False)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("white")


def plot_segments_and_contact_points(
    points,
    plane_segments,
    contact_points,
    remaining_points=None,
    contact_spacing=0.5,
    show_remaining=False,
    save_path=None,
    show=True,
):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    if show_remaining and remaining_points is not None and len(remaining_points) > 0:
        ax.scatter(
            remaining_points[:, 0],
            remaining_points[:, 1],
            remaining_points[:, 2],
            s=2,
            c="lightgray",
            alpha=0.12,
            label="unsegmented partial-map points",
        )

    cmap = plt.cm.get_cmap("tab20", max(len(plane_segments), 1))

    # Plane segments
    for segment in plane_segments:
        color = cmap(segment.segment_id)
        support_polygon = compute_plane_support_polygon(segment)

        if support_polygon is None:
            continue

        face = Poly3DCollection(
            [support_polygon],
            facecolors=[color],
            alpha=0.16,
            edgecolors="black",
            linewidths=1.4,
        )
        face.set_label(f"segment {segment.segment_id} support")
        ax.add_collection3d(face)

        closed_polygon = np.vstack([support_polygon, support_polygon[0]])

        ax.plot(
            closed_polygon[:, 0],
            closed_polygon[:, 1],
            closed_polygon[:, 2],
            color="black",
            linewidth=1.6,
        )

        segment_contact_points = [
            cp for cp in contact_points
            if cp.parent_segment_id == segment.segment_id
        ]

        for cp in segment_contact_points:
            cell_polygon = compute_contact_cell_polygon(
                cp,
                segment,
                contact_spacing,
            )
            cell = Poly3DCollection(
                [cell_polygon],
                facecolors=[color],
                alpha=0.30,
                edgecolors="black",
                linewidths=0.5,
            )
            ax.add_collection3d(cell)

        # Plane normal
        ax.quiver(
            segment.center[0],
            segment.center[1],
            segment.center[2],
            segment.normal[0],
            segment.normal[1],
            segment.normal[2],
            length=0.35,
            color="black",
            linewidth=1.0,
        )

    # Contact points
    if len(contact_points) > 0:
        cp_positions = np.array([cp.position for cp in contact_points])
        cp_normals = np.array([cp.normal for cp in contact_points])

        ax.scatter(
            cp_positions[:, 0],
            cp_positions[:, 1],
            cp_positions[:, 2],
            s=150,
            c="gold",
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            label="contact candidates",
        )

        # Candidate normals
        normal_length = 0.25

        ax.quiver(
            cp_positions[:, 0],
            cp_positions[:, 1],
            cp_positions[:, 2],
            cp_normals[:, 0],
            cp_normals[:, 1],
            cp_normals[:, 2],
            length=normal_length,
            color="crimson",
            linewidth=1.2,
        )

        # Optional small labels
        for cp in contact_points:
            ax.text(
                cp.position[0],
                cp.position[1],
                cp.position[2],
                f"{cp.cp_id}",
                fontsize=8,
                color="black",
            )

    ax.set_title("Bounded Plane Subsegments + Contact Point Candidates")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    style_clean_3d_axes(ax)

    visible_points = []
    for segment in plane_segments:
        support_polygon = compute_plane_support_polygon(segment)
        if support_polygon is not None:
            visible_points.append(support_polygon)
    if len(contact_points) > 0:
        visible_points.append(np.array([cp.position for cp in contact_points]))
    if show_remaining and remaining_points is not None and len(remaining_points) > 0:
        visible_points.append(remaining_points)

    if len(visible_points) > 0:
        set_axes_equal(ax, np.vstack(visible_points))
    else:
        set_axes_equal(ax, points)

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) <= 15:
        ax.legend(loc="upper right")

    ax.view_init(elev=24, azim=-45)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved preview to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# Main demo
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate contact point candidates from partial mock satellite maps."
    )

    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--side", type=str, default="x_pos")
    parser.add_argument("--contact-spacing", "--grid-size", dest="contact_spacing", type=float, default=1.0)
    parser.add_argument(
        "--min-points-per-candidate",
        "--min-points-per-cell",
        dest="min_points_per_candidate",
        type=int,
        default=8,
    )
    parser.add_argument("--support-radius", type=float, default=None)
    parser.add_argument("--boundary-margin", type=float, default=0.15)
    parser.add_argument("--show-remaining", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    full_points, _, _ = generate_mock_satellite_point_cloud(seed=args.seed)

    points = select_demo_observed_points(
        full_points=full_points,
        side=args.side,
    )

    plane_segments, remaining = segment_planes_ransac(
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
        contact_spacing=args.contact_spacing,
        min_points_per_candidate=args.min_points_per_candidate,
        support_radius=args.support_radius,
        boundary_margin=args.boundary_margin,
    )

    print("\nDetected plane segments:")
    print("-" * 80)

    for seg in plane_segments:
        print(f"Plane {seg.segment_id}")
        print(f"  number of points : {len(seg.points)}")
        print(f"  center           : {np.round(seg.center, 3)}")
        print(f"  normal           : {np.round(seg.normal, 3)}")
        print(f"  bbox area approx : {seg.area_estimate:.3f}")
        print()

    print("\nGenerated contact point candidates:")
    print("-" * 80)

    for cp in contact_points:
        print(f"CP {cp.cp_id}")
        print(f"  position          : {np.round(cp.position, 3)}")
        print(f"  normal            : {np.round(cp.normal, 3)}")
        print(f"  parent segment    : {cp.parent_segment_id}")
        print(f"  confidence        : {cp.confidence:.3f}")
        print(f"  area support      : {cp.area_support:.3f}")
        print()

    print(f"Total planes detected: {len(plane_segments)}")
    print(f"Total contact points : {len(contact_points)}")
    print(f"Selected side        : {args.side}")
    print(f"Full point cloud     : {full_points.shape}")
    print(f"Partial point cloud  : {points.shape}")
    print(f"Remaining points     : {len(remaining)}")
    print("-" * 80)

    plot_segments_and_contact_points(
        points,
        plane_segments,
        contact_points,
        remaining_points=remaining,
        contact_spacing=args.contact_spacing,
        show_remaining=args.show_remaining,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
