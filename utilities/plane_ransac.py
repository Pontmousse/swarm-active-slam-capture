"""
Repeated RANSAC plane segmentation demo.

Goal:
    mock_data.py satellite point cloud
        -> repeated Open3D RANSAC plane extraction
        -> bounded plane segments using inlier points
        -> rough area estimate
        -> visualization
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


try:
    import open3d as o3d
except ImportError as exc:
    raise ImportError(
        "Open3D is required. Install it with:\n\n"
        "    pip install open3d\n"
    ) from exc


try:
    from .mock_data import generate_mock_satellite_point_cloud
except ImportError:
    from mock_data import generate_mock_satellite_point_cloud


# ============================================================
# Data structures
# ============================================================

@dataclass
class PlaneSegment:
    segment_id: int
    points: np.ndarray
    normal: np.ndarray
    center: np.ndarray
    plane_equation: np.ndarray  # [a, b, c, d]
    area_estimate: float


# ============================================================
# Geometry utilities
# ============================================================

def to_open3d_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def make_plane_basis(normal):
    """
    Given a plane normal, create two orthonormal basis vectors u, v
    spanning the plane.
    """
    normal = normalize(normal)

    # Pick a vector not parallel to the normal
    if abs(normal[2]) < 0.9:
        temp = np.array([0.0, 0.0, 1.0])
    else:
        temp = np.array([1.0, 0.0, 0.0])

    u = np.cross(normal, temp)
    u = normalize(u)

    v = np.cross(normal, u)
    v = normalize(v)

    return u, v


def estimate_plane_area_bbox(points, normal):
    """
    Estimate bounded plane area using a 2D bounding box in plane coordinates.

    This is intentionally simple.
    Later, this can be replaced by:
        - convex hull area
        - alpha shape
        - concave polygon boundary
    """
    center = points.mean(axis=0)
    u, v = make_plane_basis(normal)

    rel = points - center
    x = rel @ u
    y = rel @ v

    width = x.max() - x.min()
    height = y.max() - y.min()

    return float(width * height)


def compute_plane_bbox_corners(points, normal):
    """
    Return 3D corners of the 2D bounding box of the segment
    in the local plane frame.
    """
    center = points.mean(axis=0)
    u, v = make_plane_basis(normal)

    rel = points - center
    x = rel @ u
    y = rel @ v

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    corners_2d = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
        (xmin, ymin),
    ]

    corners_3d = np.array([
        center + xi * u + yi * v
        for xi, yi in corners_2d
    ])

    return corners_3d


def orient_plane_normal_outward(normal, d, segment_center, global_center):
    """
    RANSAC plane normals have arbitrary sign.

    For visualization, orient the normal roughly away from the global
    point-cloud center.
    """
    direction = segment_center - global_center

    if np.dot(normal, direction) < 0:
        normal = -normal
        d = -d

    return normal, d


# ============================================================
# Repeated RANSAC plane segmentation
# ============================================================

def segment_planes_ransac(
    points,
    max_planes=10,
    distance_threshold=0.03,
    ransac_n=3,
    num_iterations=1000,
    min_inliers=25,
    min_remaining_points=25,
):
    """
    Repeated RANSAC plane extraction.

    Steps:
        1. Find dominant plane.
        2. Store its inliers as a PlaneSegment.
        3. Remove inliers.
        4. Repeat.
    """
    remaining = points.copy()
    global_center = points.mean(axis=0)

    segments = []

    for segment_id in range(max_planes):
        if len(remaining) < min_remaining_points:
            break

        pcd = to_open3d_point_cloud(remaining)

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )

        inliers = np.asarray(inliers, dtype=int)

        if len(inliers) < min_inliers:
            break

        segment_points = remaining[inliers]
        segment_center = segment_points.mean(axis=0)

        plane_model = np.asarray(plane_model, dtype=float)
        normal = plane_model[:3]
        d = plane_model[3]

        # Normalize plane equation
        norm_n = np.linalg.norm(normal)
        normal = normal / norm_n
        d = d / norm_n

        # Orient normal for cleaner plotting
        normal, d = orient_plane_normal_outward(
            normal,
            d,
            segment_center,
            global_center,
        )

        area = estimate_plane_area_bbox(segment_points, normal)

        segment = PlaneSegment(
            segment_id=segment_id,
            points=segment_points,
            normal=normal,
            center=segment_center,
            plane_equation=np.array([normal[0], normal[1], normal[2], d]),
            area_estimate=area,
        )

        segments.append(segment)

        # Remove inliers
        mask = np.ones(len(remaining), dtype=bool)
        mask[inliers] = False
        remaining = remaining[mask]

    return segments, remaining


# ============================================================
# Visualization
# ============================================================

def set_axes_equal(ax, points):
    """
    Make 3D matplotlib axes have equal scale.
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_plane_segments(points, segments, remaining_points):
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Plot remaining non-planar / leftover points
    if len(remaining_points) > 0:
        ax.scatter(
            remaining_points[:, 0],
            remaining_points[:, 1],
            remaining_points[:, 2],
            s=4,
            c="lightgray",
            alpha=0.4,
            label="remaining",
        )

    cmap = plt.cm.get_cmap("tab20", max(len(segments), 1))

    for segment in segments:
        pts = segment.points
        color = cmap(segment.segment_id)

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=6,
            color=color,
            alpha=0.85,
            label=f"plane {segment.segment_id}",
        )

        # Plot plane center
        ax.scatter(
            segment.center[0],
            segment.center[1],
            segment.center[2],
            s=90,
            color="black",
            marker="x",
        )

        # Plot normal arrow
        arrow_length = 0.35
        ax.quiver(
            segment.center[0],
            segment.center[1],
            segment.center[2],
            segment.normal[0],
            segment.normal[1],
            segment.normal[2],
            length=arrow_length,
            color="black",
            linewidth=1.5,
        )

        # Plot simple bounded rectangular extent
        corners = compute_plane_bbox_corners(segment.points, segment.normal)
        ax.plot(
            corners[:, 0],
            corners[:, 1],
            corners[:, 2],
            color="black",
            linewidth=1.2,
        )

    ax.set_title("Repeated RANSAC Plane Segmentation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    set_axes_equal(ax, points)

    # Avoid huge duplicated legend
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) <= 12:
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# ============================================================
# Main demo
# ============================================================

def main():
    points, _, _ = generate_mock_satellite_point_cloud(seed=6)

    segments, remaining = segment_planes_ransac(
        points,
        max_planes=20,
        distance_threshold=0.035,
        ransac_n=3,
        num_iterations=1200,
        min_inliers=50,
        min_remaining_points=50,
    )

    print("\nDetected plane segments:")
    print("-" * 80)

    for seg in segments:
        a, b, c, d = seg.plane_equation

        print(f"Plane {seg.segment_id}")
        print(f"  number of points : {len(seg.points)}")
        print(f"  center           : {np.round(seg.center, 3)}")
        print(f"  normal           : {np.round(seg.normal, 3)}")
        print(f"  equation         : {a:.3f} x + {b:.3f} y + {c:.3f} z + {d:.3f} = 0")
        print(f"  bbox area approx : {seg.area_estimate:.3f}")
        print()

    print(f"Remaining unsegmented points: {len(remaining)}")
    print("-" * 80)

    plot_plane_segments(points, segments, remaining)


if __name__ == "__main__":
    main()
