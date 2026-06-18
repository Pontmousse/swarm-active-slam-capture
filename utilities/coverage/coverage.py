"""
coverage.py

Simple ellipsoid coverage utility.

Current prototype:
    - sample/observed point cloud is provided outside the coverage functions
    - ellipsoid is fitted from the observed point cloud
    - ellipsoid surface is discretized into patch objects
    - observed points are projected onto the ellipsoid
    - nearest patches are marked as covered

Run from project root:

    python utilities/coverage.py

Or from inside utilities:

    python coverage.py
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

UTILITIES_DIR = Path(__file__).resolve().parents[1]
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

try:
    from ..data.mock_data import (
        EllipsoidModel,
        generate_mock_satellite_point_cloud,
        ellipsoid_wireframe,
        set_axes_equal,
    )
    from .ellipsoid import (
        fit_viewpoint_ellipsoid_pca,
        project_points_from_agent_to_ellipsoid,
        project_points_radially_to_ellipsoid,
        project_points_from_normals_to_ellipsoid,
        ellipsoid_to_world_frame,
    )
except ImportError:
    from data.mock_data import (
        EllipsoidModel,
        generate_mock_satellite_point_cloud,
        ellipsoid_wireframe,
        set_axes_equal,
    )
    from ellipsoid import (
        fit_viewpoint_ellipsoid_pca,
        project_points_from_agent_to_ellipsoid,
        project_points_radially_to_ellipsoid,
        project_points_from_normals_to_ellipsoid,
        ellipsoid_to_world_frame,
    )


@dataclass
class EllipsoidPatch:
    patch_id: int
    theta: float
    phi: float
    center: np.ndarray
    normal: np.ndarray
    corners: np.ndarray
    covered: bool = False
    hit_count: int = 0


def ellipsoid_point_from_angles(
    ellipsoid,
    theta: float,
    phi: float,
) -> np.ndarray:
    """
    Return one world-frame ellipsoid surface point from angular coordinates.
    """
    a, b, c = ellipsoid.axes

    local_point = np.array([
        a * np.cos(theta) * np.sin(phi),
        b * np.sin(theta) * np.sin(phi),
        c * np.cos(phi),
    ])

    return ellipsoid_to_world_frame(
        local_point[None, :],
        ellipsoid,
    )[0]


def create_ellipsoid_patches(
    ellipsoid,
    n_theta: int = 48,
    n_phi: int = 24,
) -> list[EllipsoidPatch]:
    """
    Create simple patch objects on the ellipsoid surface.

    Each patch stores a center for coverage logic and four corners for drawing.
    """
    patches = []

    # Avoid exact poles because many theta values collapse to the same point there.
    theta_values = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phi_values = np.linspace(0.05, np.pi - 0.05, n_phi)
    dtheta = 2.0 * np.pi / n_theta
    dphi = (np.pi - 0.10) / max(n_phi - 1, 1)

    a, b, c = ellipsoid.axes

    patch_id = 0

    for phi in phi_values:
        for theta in theta_values:
            local_center = np.array([
                a * np.cos(theta) * np.sin(phi),
                b * np.sin(theta) * np.sin(phi),
                c * np.cos(phi),
            ])

            # Normal of implicit ellipsoid:
            # F = x^2/a^2 + y^2/b^2 + z^2/c^2 - 1
            local_normal = np.array([
                local_center[0] / (a * a),
                local_center[1] / (b * b),
                local_center[2] / (c * c),
            ])

            local_normal = normalize(local_normal)

            center_world = ellipsoid_to_world_frame(
                local_center[None, :],
                ellipsoid,
            )[0]

            normal_world = local_normal @ ellipsoid.rotation.T
            normal_world = normalize(normal_world)

            theta_minus = theta - 0.5 * dtheta
            theta_plus = theta + 0.5 * dtheta
            phi_minus = max(0.001, phi - 0.5 * dphi)
            phi_plus = min(np.pi - 0.001, phi + 0.5 * dphi)

            corners = np.array([
                ellipsoid_point_from_angles(ellipsoid, theta_minus, phi_minus),
                ellipsoid_point_from_angles(ellipsoid, theta_plus, phi_minus),
                ellipsoid_point_from_angles(ellipsoid, theta_plus, phi_plus),
                ellipsoid_point_from_angles(ellipsoid, theta_minus, phi_plus),
            ])

            patches.append(
                EllipsoidPatch(
                    patch_id=patch_id,
                    theta=float(theta),
                    phi=float(phi),
                    center=center_world,
                    normal=normal_world,
                    corners=corners,
                )
            )

            patch_id += 1

    return patches


def update_coverage_from_projected_points(
    patches: list[EllipsoidPatch],
    projected_points: np.ndarray,
) -> list[EllipsoidPatch]:
    """
    Mark nearest ellipsoid patches as covered.

    Each projected point votes for exactly one nearest patch.
    """
    if len(patches) == 0:
        return patches

    projected_points = np.asarray(projected_points, dtype=float)

    if projected_points.size == 0:
        return patches

    patch_centers = np.array([patch.center for patch in patches])

    tree = cKDTree(patch_centers)

    _, nearest_patch_ids = tree.query(projected_points, k=1)

    unique_patch_ids, hit_counts = np.unique(nearest_patch_ids, return_counts=True)

    for patch_id, hit_count in zip(unique_patch_ids, hit_counts):
        patches[int(patch_id)].covered = True
        patches[int(patch_id)].hit_count += int(hit_count)

    return patches


def orient_normals_for_projection(
    points: np.ndarray,
    normals: np.ndarray,
    ellipsoid,
    agent_position: np.ndarray,
    ambiguity_threshold: float = 0.15,
) -> np.ndarray:
    """
    Orient normals before normal-based ellipsoid projection.

    Default rule:
        normals should point outward from the ellipsoid center.

    Ambiguous case:
        if the normal is almost perpendicular to point - ellipsoid.center,
        use the agent position instead and orient the normal toward the agent.
    """
    oriented_normals = np.asarray(normals, dtype=float).copy()

    for i, (point, normal) in enumerate(zip(points, oriented_normals)):
        normal = normalize(normal)

        outward_dir = normalize(point - ellipsoid.center)
        d_out = float(np.dot(normal, outward_dir))

        if abs(d_out) < ambiguity_threshold:
            agent_dir = normalize(agent_position - point)

            if np.dot(normal, agent_dir) < 0:
                normal = -normal

        elif d_out < 0:
            normal = -normal

        oriented_normals[i] = normal

    return oriented_normals


def update_coverage_from_observed_points(
    patches: list[EllipsoidPatch],
    observed_points: np.ndarray,
    observed_normals: np.ndarray,
    agent_position: np.ndarray,
    ellipsoid: EllipsoidModel,
    projection_mode: str = "normal",
) -> tuple[list[EllipsoidPatch], np.ndarray]:
    """
    Project observed points onto the ellipsoid and update patch coverage.

    projection_mode:
        "normal" -> use surface-normal proxy projection
        "agent"  -> use agent-position ray projection
        "radial" -> use center radial projection
    """
    if projection_mode == "normal":
        oriented_normals = orient_normals_for_projection(
            points=observed_points,
            normals=observed_normals,
            ellipsoid=ellipsoid,
            agent_position=agent_position,
        )

        projected_points = project_points_from_normals_to_ellipsoid(
            points=observed_points,
            normals=oriented_normals,
            ellipsoid=ellipsoid,
        )

    elif projection_mode == "agent":
        projected_points = project_points_from_agent_to_ellipsoid(
            points=observed_points,
            agent_position=agent_position,
            ellipsoid=ellipsoid,
        )

    elif projection_mode == "radial":
        projected_points = project_points_radially_to_ellipsoid(
            points=observed_points,
            ellipsoid=ellipsoid,
        )

    else:
        raise ValueError(f"Unknown projection_mode: {projection_mode}")

    patches = update_coverage_from_projected_points(
        patches=patches,
        projected_points=projected_points,
    )

    return patches, projected_points


def compute_coverage_ratio(patches: list[EllipsoidPatch]) -> float:
    """
    Compute simple binary coverage ratio.

    coverage = number of covered patches / total number of patches
    """
    if len(patches) == 0:
        return 0.0

    covered_count = sum(patch.covered for patch in patches)
    return covered_count / len(patches)


def get_covered_patches(patches: list[EllipsoidPatch]) -> list[EllipsoidPatch]:
    return [patch for patch in patches if patch.covered]


def get_uncovered_patches(patches: list[EllipsoidPatch]) -> list[EllipsoidPatch]:
    return [patch for patch in patches if not patch.covered]


def patches_to_points(patches: list[EllipsoidPatch]) -> np.ndarray:
    """
    Convert patch centers to an array of shape (N, 3).
    """
    if len(patches) == 0:
        return np.empty((0, 3))

    return np.array([patch.center for patch in patches])


def add_patch_collection(
    ax,
    patches: list[EllipsoidPatch],
    alpha: float,
    label: str,
) -> None:
    """
    Draw ellipsoid patches as small quadrilateral surface cells.
    """
    if len(patches) == 0:
        return

    faces = [patch.corners for patch in patches]

    collection = Poly3DCollection(
        faces,
        alpha=alpha,
        linewidths=0.15,
        edgecolors="k",
    )

    collection.set_label(label)
    ax.add_collection3d(collection)


def enable_scroll_zoom(fig, ax, zoom_step: float = 0.9) -> None:
    """
    Bind mouse-wheel scrolling to 3D zoom by rescaling all axis limits.
    """
    def zoom_limits(limits, scale: float) -> tuple[float, float]:
        center = 0.5 * (limits[0] + limits[1])
        radius = 0.5 * (limits[1] - limits[0]) * scale
        return center - radius, center + radius

    def on_scroll(event) -> None:
        if event.inaxes != ax:
            return

        if event.button == "up":
            scale = zoom_step
        elif event.button == "down":
            scale = 1.0 / zoom_step
        else:
            return

        ax.set_xlim3d(*zoom_limits(ax.get_xlim3d(), scale))
        ax.set_ylim3d(*zoom_limits(ax.get_ylim3d(), scale))
        ax.set_zlim3d(*zoom_limits(ax.get_zlim3d(), scale))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)

    if norm < 1e-12:
        return v

    return v / norm


def select_demo_observed_points_and_normals(
    full_points: np.ndarray,
    full_normals: np.ndarray,
    side: str = "x_pos",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select a partial point cloud and its corresponding normals.

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
    observed_normals = full_normals[mask]

    if observed_points.shape[0] < 10:
        raise ValueError("Too few observed points selected.")

    return observed_points, observed_normals

def demo_agent_position_from_side(
    ellipsoid: EllipsoidModel,
    side: str,
    distance_scale: float = 2.5,
) -> np.ndarray:
    """
    Create a simple mock agent position consistent with the observed side.

    Example:
        side="x_pos" means the agent is placed on the +x side of the target.
    """
    direction_map = {
        "x_pos": np.array([1.0, 0.0, 0.0]),
        "x_neg": np.array([-1.0, 0.0, 0.0]),
        "y_pos": np.array([0.0, 1.0, 0.0]),
        "y_neg": np.array([0.0, -1.0, 0.0]),
        "z_pos": np.array([0.0, 0.0, 1.0]),
        "z_neg": np.array([0.0, 0.0, -1.0]),
    }

    if side not in direction_map:
        raise ValueError(f"Unknown side: {side}")

    direction = direction_map[side]

    # Put the mock agent outside the ellipsoid.
    distance = distance_scale * float(np.max(ellipsoid.axes))

    return ellipsoid.center + distance * direction

def plot_coverage(
    full_points: np.ndarray,
    observed_points: np.ndarray,
    agent_position: np.ndarray,
    ellipsoid: EllipsoidModel,
    patches: list[EllipsoidPatch],
    projected_points: np.ndarray | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Static diagnostic plot.

    Shows:
        - full target point cloud as faint ground truth
        - observed point cloud as current partial map
        - current fitted ellipsoid
        - covered patches
        - uncovered patches
    """
    covered_patches = get_covered_patches(patches)
    uncovered_patches = get_uncovered_patches(patches)

    coverage_ratio = compute_coverage_ratio(patches)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        full_points[:, 0],
        full_points[:, 1],
        full_points[:, 2],
        s=1,
        alpha=0.08,
        label="full mock target cloud",
    )

    ax.scatter(
        observed_points[:, 0],
        observed_points[:, 1],
        observed_points[:, 2],
        s=2,
        alpha=0.35,
        label="observed partial cloud",
    )

    xw, yw, zw = ellipsoid_wireframe(ellipsoid, n_u=48, n_v=24)

    ax.plot_wireframe(
        xw,
        yw,
        zw,
        linewidth=0.4,
        alpha=0.25,
        label="current fitted ellipsoid",
    )

    add_patch_collection(
        ax=ax,
        patches=uncovered_patches,
        alpha=0.12,
        label="uncovered patches",
    )

    add_patch_collection(
        ax=ax,
        patches=covered_patches,
        alpha=0.65,
        label="covered patches",
    )

    if projected_points is not None:
        ax.scatter(
            projected_points[:, 0],
            projected_points[:, 1],
            projected_points[:, 2],
            s=2,
            alpha=0.20,
            label="projected observed points",
        )

    ax.set_title(f"Ellipsoid Patch Coverage: {100.0 * coverage_ratio:.1f}%")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    ax.view_init(elev=22, azim=-42)
    ax.legend(loc="upper left")

    set_axes_equal(ax)
    enable_scroll_zoom(fig, ax)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved preview to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple ellipsoid patch coverage demo."
    )

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--side", type=str, default="x_pos")
    parser.add_argument("--margin", type=float, default=1.25)
    parser.add_argument("--n-theta", type=int, default=48)
    parser.add_argument("--n-phi", type=int, default=24)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    full_points, full_normals, _ = generate_mock_satellite_point_cloud(seed=args.seed)

    observed_points, observed_normals = select_demo_observed_points_and_normals(
        full_points=full_points,
        full_normals=full_normals,
        side=args.side,
    )

    # Important:
    # The ellipsoid is fitted from the currently observed partial map,
    # but centered on the estimated target body center, not the partial-map centroid.
    target_center_estimate = np.array([0.0, 0.0, 0.0])

    ellipsoid = fit_viewpoint_ellipsoid_pca(
        points=observed_points,
        center=target_center_estimate,
        margin=args.margin,
    )

    patches = create_ellipsoid_patches(
        ellipsoid=ellipsoid,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
    )

    agent_position = demo_agent_position_from_side(
        ellipsoid=ellipsoid,
        side=args.side,
    )

    patches, projected_points = update_coverage_from_observed_points(
        patches=patches,
        observed_points=observed_points,
        observed_normals=observed_normals,
        agent_position=agent_position,
        ellipsoid=ellipsoid,
        projection_mode="normal",
    )

    coverage_ratio = compute_coverage_ratio(patches)

    print("Coverage demo")
    print(f"  full point cloud:     {full_points.shape}")
    print(f"  observed point cloud: {observed_points.shape}")
    print(f"  selected side:        {args.side}")
    print(f"  ellipsoid center:     {np.round(ellipsoid.center, 3)}")
    print(f"  ellipsoid axes:       {np.round(ellipsoid.axes, 3)}")
    print(f"  patches:              {len(patches)}")
    print(f"  covered patches:      {len(get_covered_patches(patches))}")
    print(f"  uncovered patches:    {len(get_uncovered_patches(patches))}")
    print(f"  coverage ratio:       {100.0 * coverage_ratio:.2f}%")

    plot_coverage(
        full_points=full_points,
        agent_position=agent_position,
        observed_points=observed_points,
        ellipsoid=ellipsoid,
        patches=patches,
        projected_points=projected_points,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
