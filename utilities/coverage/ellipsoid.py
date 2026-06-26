"""
ellipsoid.py

Simple ellipsoid utility.

This file uses mock_data.py for the test satellite point cloud.

Expected folder:

    utilities/
    ├── mock_data.py
    └── ellipsoid.py

Run from project root:

    python utilities/ellipsoid.py

Or from inside utilities:

    python ellipsoid.py
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
    # Works if utilities is imported as a package
    from ..data.mock_data import (
        EllipsoidModel,
        generate_mock_satellite_point_cloud,
        ellipsoid_wireframe,
        set_axes_equal,
    )
except ImportError:
    # Works when running directly from utilities/coverage.
    from data.mock_data import (
        EllipsoidModel,
        generate_mock_satellite_point_cloud,
        ellipsoid_wireframe,
        set_axes_equal,
    )


def world_to_ellipsoid_frame(points: np.ndarray, ellipsoid: EllipsoidModel) -> np.ndarray:
    """
    Transform world-frame points into the ellipsoid local frame.
    """
    return (points - ellipsoid.center) @ ellipsoid.rotation


def ellipsoid_to_world_frame(points_local: np.ndarray, ellipsoid: EllipsoidModel) -> np.ndarray:
    """
    Transform ellipsoid-local points into the world frame.
    """
    return points_local @ ellipsoid.rotation.T + ellipsoid.center


def fit_viewpoint_ellipsoid_pca(
    points: np.ndarray,
    center: np.ndarray | None = None,
    margin: float = 1.25,
    min_axis: float = 0.25,
) -> EllipsoidModel:
    """
    Fit a simple PCA-based viewpoint ellipsoid around a point cloud.

    This is not an exact ellipsoid fitting method.

    It uses:

        center   = provided target center estimate, or point-cloud centroid fallback
        rotation = PCA eigenvectors
        axes     = point-cloud extents in PCA frame, inflated by margin

    This is good enough for a first active-SLAM viewpoint shell.
    """
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    if center is None:
        center = points.mean(axis=0)
    else:
        center = np.asarray(center, dtype=float)

    centered = points - center

    covariance = np.cov(centered.T)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort PCA directions from largest spread to smallest spread
    order = np.argsort(eigenvalues)[::-1]
    rotation = eigenvectors[:, order]

    # Keep a right-handed frame
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1.0

    temporary_ellipsoid = EllipsoidModel(
        center=center,
        axes=np.ones(3),
        rotation=rotation,
    )

    local_points = world_to_ellipsoid_frame(points, temporary_ellipsoid)

    axes = np.max(np.abs(local_points), axis=0)
    axes = margin * axes
    axes = np.maximum(axes, min_axis)

    return EllipsoidModel(
        center=center,
        axes=axes,
        rotation=rotation,
    )



def estimate_point_normals(
    points: np.ndarray,
    camera_location: np.ndarray | None = None,
    radius: float | None = None,
    max_nn: int = 30,
) -> np.ndarray:
    """
    Estimate normals for a partial map point cloud (Open3D).

    camera_location: e.g. agent position — orients normals toward the viewer.
    Returns (N, 3); empty input -> (0, 3).
    """
    import open3d as o3d

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return np.empty((0, 3), dtype=float)

    if radius is None:
        sample = points[: min(len(points), 200)]
        if len(sample) >= 2:
            dists = np.linalg.norm(sample[1:] - sample[:-1], axis=1)
            radius = float(max(np.median(dists) * 4.0, 0.01))
        else:
            radius = 0.05

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    if camera_location is not None:
        pcd.orient_normals_towards_camera_location(
            camera_location=np.asarray(camera_location, dtype=float).reshape(3)
        )
    return np.asarray(pcd.normals, dtype=float)



def project_points_radially_to_ellipsoid(
    points: np.ndarray,
    ellipsoid: EllipsoidModel,
) -> np.ndarray:
    """
    Project points onto the ellipsoid surface along rays from the ellipsoid center.

    In the ellipsoid frame, the surface is:

        (x/a)^2 + (y/b)^2 + (z/c)^2 = 1
    """
    points = np.asarray(points, dtype=float)

    local = world_to_ellipsoid_frame(points, ellipsoid)

    scaled_radius = np.sqrt(np.sum((local / ellipsoid.axes) ** 2, axis=1))
    scaled_radius = np.maximum(scaled_radius, 1e-12)

    projected_local = local / scaled_radius[:, None]

    return ellipsoid_to_world_frame(projected_local, ellipsoid)



def project_points_from_agent_to_ellipsoid(
    points: np.ndarray,
    agent_position: np.ndarray,
    ellipsoid: EllipsoidModel,
) -> np.ndarray:
    """
    Project observed target points onto the ellipsoid using the agent viewing rays.

    For each observed target point p, we define the ray:

        r(t) = agent_position + t * (p - agent_position)

    If the agent is outside the ellipsoid and the observed point is inside it,
    the useful intersection is usually the first ellipsoid intersection between
    the agent and the observed point.

    This is better than radial projection for viewpoint coverage.
    """
    points = np.asarray(points, dtype=float)
    agent_position = np.asarray(agent_position, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    if agent_position.shape != (3,):
        raise ValueError("agent_position must have shape (3,)")

    # Transform ray origin and target points into ellipsoid local frame.
    agent_local = world_to_ellipsoid_frame(
        agent_position[None, :],
        ellipsoid,
    )[0]

    points_local = world_to_ellipsoid_frame(
        points,
        ellipsoid,
    )

    directions_local = points_local - agent_local[None, :]

    a, b, c = ellipsoid.axes

    # Ellipsoid equation:
    # (x/a)^2 + (y/b)^2 + (z/c)^2 = 1
    #
    # Substitute ray:
    # x(t) = o + t d
    #
    # This gives quadratic:
    # A t^2 + B t + C = 0

    scaled_origin = agent_local / ellipsoid.axes
    scaled_directions = directions_local / ellipsoid.axes

    A = np.sum(scaled_directions * scaled_directions, axis=1)
    B = 2.0 * np.sum(scaled_origin[None, :] * scaled_directions, axis=1)
    C = np.sum(scaled_origin * scaled_origin) - 1.0

    discriminant = B * B - 4.0 * A * C

    projected_local = np.full_like(points_local, np.nan)

    valid = discriminant >= 0.0

    if np.any(valid):
        sqrt_disc = np.sqrt(discriminant[valid])

        A_valid = A[valid]
        B_valid = B[valid]

        t1 = (-B_valid - sqrt_disc) / (2.0 * A_valid)
        t2 = (-B_valid + sqrt_disc) / (2.0 * A_valid)

        # Prefer intersection between agent and observed point.
        # For an outside agent and inside target point, this is usually t in [0, 1].
        t_candidates = np.stack([t1, t2], axis=1)

        chosen_t = np.full(t1.shape, np.nan)

        for i in range(t_candidates.shape[0]):
            candidates = t_candidates[i]

            between_agent_and_point = candidates[
                (candidates >= 0.0) & (candidates <= 1.0)
            ]

            if between_agent_and_point.size > 0:
                chosen_t[i] = np.min(between_agent_and_point)
            else:
                positive = candidates[candidates >= 0.0]

                if positive.size > 0:
                    chosen_t[i] = np.min(positive)

        valid_indices = np.where(valid)[0]
        good = ~np.isnan(chosen_t)

        projected_local[valid_indices[good]] = (
            agent_local[None, :]
            + chosen_t[good, None] * directions_local[valid_indices[good]]
        )

    projected_world = ellipsoid_to_world_frame(
        projected_local,
        ellipsoid,
    )

    # Remove failed projections.
    good_rows = ~np.isnan(projected_world).any(axis=1)

    return projected_world[good_rows]


def project_points_from_normals_to_ellipsoid(
    points: np.ndarray,
    normals: np.ndarray,
    ellipsoid: EllipsoidModel,
    pseudo_observer_scale: float = 2.5,
) -> np.ndarray:
    """
    Project mapped target points onto the ellipsoid using surface normals.

    This is a proxy for viewpoint coverage when the original observing
    agent position is unknown.

    For each mapped point p with surface normal n:

        pseudo_observer = p + R * n

    Then we intersect the ray:

        r(t) = pseudo_observer + t * (p - pseudo_observer)

    with the ellipsoid.

    This approximates the idea that a surface patch was likely observed
    from the outward-normal side.

    Args:
        points:
            Mapped target points, shape (N, 3).

        normals:
            Surface normals at those points, shape (N, 3).

        ellipsoid:
            Current viewpoint ellipsoid.

        pseudo_observer_scale:
            Multiplier for max ellipsoid axis to place the pseudo-observer
            outside the ellipsoid.

    Returns:
        projected_points:
            Points on the ellipsoid surface, shape (M, 3), where M <= N.
            Failed projections are removed.
    """
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("normals must have shape (N, 3)")

    if points.shape[0] != normals.shape[0]:
        raise ValueError("points and normals must have the same number of rows")

    # Normalize normals.
    normal_norms = np.linalg.norm(normals, axis=1)
    valid_normals = normal_norms > 1e-12

    points = points[valid_normals]
    normals = normals[valid_normals]
    normal_norms = normal_norms[valid_normals]

    if points.shape[0] == 0:
        return np.empty((0, 3))

    normals = normals / normal_norms[:, None]

    # Force normals to point outward relative to ellipsoid center.
    outward_vectors = points - ellipsoid.center[None, :]
    outward_check = np.sum(normals * outward_vectors, axis=1)

    flip_mask = outward_check < 0.0
    normals[flip_mask] *= -1.0

    # Create pseudo-observers outside the ellipsoid.
    pseudo_distance = pseudo_observer_scale * float(np.max(ellipsoid.axes))
    pseudo_observers = points + pseudo_distance * normals

    # Transform rays into ellipsoid frame.
    origins_local = world_to_ellipsoid_frame(
        pseudo_observers,
        ellipsoid,
    )

    points_local = world_to_ellipsoid_frame(
        points,
        ellipsoid,
    )

    directions_local = points_local - origins_local

    # Ellipsoid equation:
    # (x/a)^2 + (y/b)^2 + (z/c)^2 = 1
    #
    # Ray:
    # x(t) = o + t d
    #
    # Quadratic:
    # A t^2 + B t + C = 0

    scaled_origins = origins_local / ellipsoid.axes[None, :]
    scaled_directions = directions_local / ellipsoid.axes[None, :]

    A = np.sum(scaled_directions * scaled_directions, axis=1)
    B = 2.0 * np.sum(scaled_origins * scaled_directions, axis=1)
    C = np.sum(scaled_origins * scaled_origins, axis=1) - 1.0

    discriminant = B * B - 4.0 * A * C

    projected_local = np.full_like(points_local, np.nan)

    valid = (discriminant >= 0.0) & (A > 1e-12)

    if np.any(valid):
        sqrt_disc = np.sqrt(discriminant[valid])

        A_valid = A[valid]
        B_valid = B[valid]

        t1 = (-B_valid - sqrt_disc) / (2.0 * A_valid)
        t2 = (-B_valid + sqrt_disc) / (2.0 * A_valid)

        t_candidates = np.stack([t1, t2], axis=1)

        chosen_t = np.full(t1.shape, np.nan)

        for i in range(t_candidates.shape[0]):
            candidates = t_candidates[i]

            # Prefer intersection between pseudo-observer and mapped point.
            between = candidates[
                (candidates >= 0.0) & (candidates <= 1.0)
            ]

            if between.size > 0:
                chosen_t[i] = np.min(between)
            else:
                positive = candidates[candidates >= 0.0]

                if positive.size > 0:
                    chosen_t[i] = np.min(positive)

        valid_indices = np.where(valid)[0]
        good = ~np.isnan(chosen_t)

        projected_local[valid_indices[good]] = (
            origins_local[valid_indices[good]]
            + chosen_t[good, None] * directions_local[valid_indices[good]]
        )

    projected_world = ellipsoid_to_world_frame(
        projected_local,
        ellipsoid,
    )

    good_rows = ~np.isnan(projected_world).any(axis=1)

    return projected_world[good_rows]


def plot_ellipsoid_fit(
    points: np.ndarray,
    ellipsoid: EllipsoidModel,
    projected_points: np.ndarray | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Static diagnostic plot.

    Shows:
    - mock satellite point cloud from mock_data.py
    - fitted viewpoint ellipsoid
    - ellipsoid center
    - optional projected points on the ellipsoid surface
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=2,
        alpha=0.35,
        label="satellite point cloud",
    )

    xw, yw, zw = ellipsoid_wireframe(ellipsoid, n_u=48, n_v=24)

    ax.plot_wireframe(
        xw,
        yw,
        zw,
        linewidth=0.5,
        alpha=0.45,
        label="viewpoint ellipsoid",
    )

    ax.scatter(
        [ellipsoid.center[0]],
        [ellipsoid.center[1]],
        [ellipsoid.center[2]],
        s=70,
        marker="x",
        label="ellipsoid center",
    )

    if projected_points is not None:
        ax.scatter(
            projected_points[:, 0],
            projected_points[:, 1],
            projected_points[:, 2],
            s=5,
            alpha=0.35,
            label="projected points",
        )

    ax.set_title("PCA Viewpoint Ellipsoid (Open3D estimated normals)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    ax.legend(loc="upper left")
    ax.view_init(elev=22, azim=-42)

    set_axes_equal(ax)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved preview to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    import matplotlib

    try:
        matplotlib.use("QtAgg")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="Fit a PCA viewpoint ellipsoid; normals from Open3D on mock points."
    )

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--margin", type=float, default=1.75)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    points, _, labels = generate_mock_satellite_point_cloud(seed=args.seed)

    ellipsoid = fit_viewpoint_ellipsoid_pca(
        points=points,
        margin=args.margin,
    )

    # Mock agent on +x side of cloud (for normal orientation only).
    cloud_center = points.mean(axis=0)
    agent_position = cloud_center + np.array(
        [2.5 * float(np.max(np.ptp(points, axis=0))), 0.0, 0.0]
    )

    observed_points = points[::12]
    observed_normals = estimate_point_normals(
        observed_points,
        camera_location=agent_position,
    )

    projected_points = project_points_from_normals_to_ellipsoid(
        points=observed_points,
        normals=observed_normals,
        ellipsoid=ellipsoid,
    )

    print("Input point cloud from mock_data.py")
    print(f"  points shape:           {points.shape}")
    print(f"  labels shape:           {labels.shape}")
    print(f"  observed subset:        {observed_points.shape[0]} points")
    print(f"  estimated normals:      {observed_normals.shape}")
    print(f"  mock agent position:    {np.round(agent_position, 3)}")
    print()

    print("Fitted ellipsoid")
    print(f"  center: {np.round(ellipsoid.center, 3)}")
    print(f"  axes:   {np.round(ellipsoid.axes, 3)}")
    print("  rotation:")
    print(np.round(ellipsoid.rotation, 3))
    print(f"  projected points:       {projected_points.shape[0]}")

    plot_ellipsoid_fit(
        points=points,
        ellipsoid=ellipsoid,
        projected_points=projected_points,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
