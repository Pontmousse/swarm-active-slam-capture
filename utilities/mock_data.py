"""
mock_data.py

Self-contained mock-data generator for swarm active-SLAM + capture experiments.

Purpose
-------
Generate simple but useful synthetic data for prototyping the pipeline:

    mock satellite point cloud
    mock agent trajectories
    mock viewpoint ellipsoid
    mock ellipsoid surface patches
    mock coverage values
    mock partial observations
    mock contact-point candidates

The script can also produce a static diagnostic visualization.

Example
-------
From the project root:

    python utilities/mock_data.py

Or directly:

    python mock_data.py --num-agents 5 --step 40 --save mock_preview.png

Dependencies
------------
Required:
    numpy
    matplotlib

This file intentionally avoids Open3D/PyVista so it remains lightweight and easy
for early debugging. Later modules can convert the returned arrays into Open3D
or PyVista objects.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # Needed for 3D projection registration.


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class AgentState:
    """Minimal agent state used by controller/perception mockups."""

    agent_id: int
    position: np.ndarray  # shape: (3,)
    velocity: np.ndarray  # shape: (3,)


@dataclass
class EllipsoidModel:
    """Viewpoint shell / exploration manifold."""

    center: np.ndarray  # shape: (3,)
    axes: np.ndarray  # shape: (3,), semi-axes [a, b, c]
    rotation: np.ndarray  # shape: (3, 3), world-from-ellipsoid rotation


@dataclass
class ContactPoint:
    """Mock contact-point candidate extracted from target geometry."""

    cp_id: int
    position: np.ndarray  # shape: (3,)
    normal: np.ndarray  # shape: (3,)
    patch_area: float
    confidence: float
    label: str


@dataclass
class MockScene:
    """Bundle of synthetic data produced by build_mock_scene()."""

    target_points: np.ndarray
    target_normals: np.ndarray
    target_labels: np.ndarray
    trajectories: np.ndarray  # shape: (T, N, 3)
    velocities: np.ndarray  # shape: (T, N, 3)
    agent_states: List[AgentState]
    ellipsoid: EllipsoidModel
    patch_centers: np.ndarray
    patch_normals: np.ndarray
    patch_coverage: np.ndarray
    observed_points: np.ndarray
    observed_agent_ids: np.ndarray
    frontier_patch_ids: np.ndarray
    assigned_patch_ids: Dict[int, int]
    contact_points: List[ContactPoint]


# -----------------------------------------------------------------------------
# Small math helpers
# -----------------------------------------------------------------------------


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a single vector."""
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize row vectors in an array of shape (N, D)."""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, eps)


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Create a 3D rotation matrix from roll, pitch, yaw in radians."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


# -----------------------------------------------------------------------------
# Mock satellite point cloud
# -----------------------------------------------------------------------------


def sample_box_surface(
    center: Iterable[float],
    size: Iterable[float],
    n: int,
    label_prefix: str,
    rng: np.random.Generator,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uniformly sample points on the surface of an axis-aligned box.

    Returns
    -------
    points : (n, 3)
    normals : (n, 3)
    labels : (n,)
        Face labels such as bus_+x, bus_-z, etc.
    """
    center = np.asarray(center, dtype=float)
    size = np.asarray(size, dtype=float)
    half = size / 2.0

    # Face areas for +/-x, +/-y, +/-z.
    face_specs = [
        ("+x", np.array([1.0, 0.0, 0.0]), size[1] * size[2]),
        ("-x", np.array([-1.0, 0.0, 0.0]), size[1] * size[2]),
        ("+y", np.array([0.0, 1.0, 0.0]), size[0] * size[2]),
        ("-y", np.array([0.0, -1.0, 0.0]), size[0] * size[2]),
        ("+z", np.array([0.0, 0.0, 1.0]), size[0] * size[1]),
        ("-z", np.array([0.0, 0.0, -1.0]), size[0] * size[1]),
    ]
    probs = np.array([spec[2] for spec in face_specs], dtype=float)
    probs /= probs.sum()
    face_ids = rng.choice(len(face_specs), size=n, p=probs)

    points = np.zeros((n, 3), dtype=float)
    normals = np.zeros((n, 3), dtype=float)
    labels = np.empty(n, dtype=object)

    for idx, (face_name, normal, _) in enumerate(face_specs):
        mask = face_ids == idx
        m = int(mask.sum())
        if m == 0:
            continue

        local = rng.uniform(-half, half, size=(m, 3))
        axis = int(np.argmax(np.abs(normal)))
        local[:, axis] = half[axis] * np.sign(normal[axis])

        points[mask] = center + local
        normals[mask] = normal
        labels[mask] = f"{label_prefix}_{face_name}"

    if noise_std > 0:
        points += rng.normal(scale=noise_std, size=points.shape)

    return points, normals, labels


def sample_rect_plane(
    center: Iterable[float],
    u_vec: Iterable[float],
    v_vec: Iterable[float],
    n: int,
    label: str,
    rng: np.random.Generator,
    noise_std: float = 0.0,
    both_sided_normals: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a rectangular plane defined by center +/- u_vec +/- v_vec.

    The full rectangle side lengths are 2*||u_vec|| and 2*||v_vec||.
    """
    center = np.asarray(center, dtype=float)
    u_vec = np.asarray(u_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)

    alpha = rng.uniform(-1.0, 1.0, size=(n, 1))
    beta = rng.uniform(-1.0, 1.0, size=(n, 1))
    points = center + alpha * u_vec + beta * v_vec

    normal = normalize(np.cross(u_vec, v_vec))
    normals = np.repeat(normal[None, :], n, axis=0)

    if both_sided_normals:
        flips = rng.choice([-1.0, 1.0], size=(n, 1))
        normals = normals * flips

    labels = np.array([label] * n, dtype=object)

    if noise_std > 0:
        points += rng.normal(scale=noise_std, size=points.shape)

    return points, normals, labels


def generate_mock_satellite_point_cloud(
    n_bus: int = 2500,
    n_panels_each: int = 900,
    noise_std: float = 0.015,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a simple satellite-like point cloud:

        - central cuboid bus
        - two rectangular solar-panel planes
        - one small mast/antenna represented as a thin box

    This is intentionally simple but provides planar regions, boundaries, normals,
    and non-trivial spatial extent for debugging coverage/contact algorithms.
    """
    rng = np.random.default_rng(seed)

    bus_points, bus_normals, bus_labels = sample_box_surface(
        center=[0.0, 0.0, 0.0],
        size=[1.8, 1.2, 1.0],
        n=n_bus,
        label_prefix="bus",
        rng=rng,
        noise_std=noise_std,
    )

    # Solar panels lie approximately in the x-y plane, normal along +z.
    left_panel_points, left_panel_normals, left_panel_labels = sample_rect_plane(
        center=[-2.25, 0.0, 0.0],
        u_vec=[0.95, 0.0, 0.0],
        v_vec=[0.0, 0.85, 0.0],
        n=n_panels_each,
        label="left_panel",
        rng=rng,
        noise_std=noise_std,
    )

    right_panel_points, right_panel_normals, right_panel_labels = sample_rect_plane(
        center=[2.25, 0.0, 0.0],
        u_vec=[0.95, 0.0, 0.0],
        v_vec=[0.0, 0.85, 0.0],
        n=n_panels_each,
        label="right_panel",
        rng=rng,
        noise_std=noise_std,
    )

    mast_points, mast_normals, mast_labels = sample_box_surface(
        center=[0.0, 0.0, 0.9],
        size=[0.18, 0.18, 0.8],
        n=300,
        label_prefix="mast",
        rng=rng,
        noise_std=noise_std,
    )

    points = np.vstack([bus_points, left_panel_points, right_panel_points, mast_points])
    normals = np.vstack([bus_normals, left_panel_normals, right_panel_normals, mast_normals])
    labels = np.concatenate([bus_labels, left_panel_labels, right_panel_labels, mast_labels])

    return points, normals, labels


# -----------------------------------------------------------------------------
# Mock swarm trajectories
# -----------------------------------------------------------------------------


def generate_mock_agent_trajectories(
    num_agents: int = 4,
    num_steps: int = 100,
    orbit_radius: float = 5.0,
    z_amplitude: float = 1.2,
    angular_rate: float = 1.25,
    seed: int = 11,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate simple non-colliding orbital-ish trajectories around the target.

    Returns
    -------
    trajectories : (num_steps, num_agents, 3)
    velocities : (num_steps, num_agents, 3)
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, num_steps)
    trajectories = np.zeros((num_steps, num_agents, 3), dtype=float)

    for i in range(num_agents):
        phase = 2.0 * np.pi * i / num_agents
        phase += rng.normal(scale=0.05)

        # Slightly different orbit radii and vertical phase per agent.
        r_i = orbit_radius * (1.0 + 0.08 * rng.normal())
        omega_i = angular_rate * (1.0 + 0.05 * rng.normal())

        theta = 2.0 * np.pi * omega_i * t + phase
        x = r_i * np.cos(theta)
        y = 0.72 * r_i * np.sin(theta)
        z = z_amplitude * np.sin(theta + 0.7 * i)

        trajectories[:, i, :] = np.column_stack([x, y, z])

    velocities = np.gradient(trajectories, axis=0)
    return trajectories, velocities


def get_agent_states_at_step(
    trajectories: np.ndarray,
    velocities: np.ndarray,
    step: int,
) -> List[AgentState]:
    """Convert trajectory arrays into a list of AgentState at a selected time step."""
    step = int(np.clip(step, 0, trajectories.shape[0] - 1))
    states = []
    for i in range(trajectories.shape[1]):
        states.append(
            AgentState(
                agent_id=i,
                position=trajectories[step, i].copy(),
                velocity=velocities[step, i].copy(),
            )
        )
    return states


# -----------------------------------------------------------------------------
# Viewpoint ellipsoid and surface patches
# -----------------------------------------------------------------------------


def generate_mock_ellipsoid(
    center: Iterable[float] = (0.0, 0.0, 0.0),
    axes: Iterable[float] = (4.4, 3.1, 2.25),
    euler_rpy: Iterable[float] = (0.0, 0.0, 0.0),
) -> EllipsoidModel:
    """Generate a mock viewpoint ellipsoid around the target."""
    return EllipsoidModel(
        center=np.asarray(center, dtype=float),
        axes=np.asarray(axes, dtype=float),
        rotation=rotation_matrix_from_euler(*euler_rpy),
    )


def sample_ellipsoid_patches(
    ellipsoid: EllipsoidModel,
    n_theta: int = 34,
    n_phi: int = 18,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample ellipsoid surface patch centers on a theta/phi grid.

    Parameters
    ----------
    theta : azimuth angle in [0, 2*pi)
    phi : polar angle in [0, pi]

    Returns
    -------
    patch_centers : (M, 3)
    patch_normals : (M, 3)
    theta_grid : (M,)
    phi_grid : (M,)
    """
    theta_values = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    # Avoid the exact poles to prevent duplicated points.
    phi_values = np.linspace(0.08 * np.pi, 0.92 * np.pi, n_phi)

    centers_local = []
    normals_local = []
    theta_grid = []
    phi_grid = []

    a, b, c = ellipsoid.axes
    for phi in phi_values:
        for theta in theta_values:
            x = a * np.sin(phi) * np.cos(theta)
            y = b * np.sin(phi) * np.sin(theta)
            z = c * np.cos(phi)
            local = np.array([x, y, z], dtype=float)

            # Ellipsoid implicit normal in local frame.
            normal_local = normalize(np.array([x / (a * a), y / (b * b), z / (c * c)]))

            centers_local.append(local)
            normals_local.append(normal_local)
            theta_grid.append(theta)
            phi_grid.append(phi)

    centers_local = np.asarray(centers_local)
    normals_local = np.asarray(normals_local)

    patch_centers = ellipsoid.center + centers_local @ ellipsoid.rotation.T
    patch_normals = normalize_rows(normals_local @ ellipsoid.rotation.T)

    return patch_centers, patch_normals, np.asarray(theta_grid), np.asarray(phi_grid)


def ellipsoid_wireframe(
    ellipsoid: EllipsoidModel,
    n_u: int = 48,
    n_v: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate X/Y/Z arrays for plotting an ellipsoid wireframe."""
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    a, b, c = ellipsoid.axes

    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))

    xyz = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    xyz_world = ellipsoid.center + xyz @ ellipsoid.rotation.T

    return (
        xyz_world[:, 0].reshape(x.shape),
        xyz_world[:, 1].reshape(y.shape),
        xyz_world[:, 2].reshape(z.shape),
    )


# -----------------------------------------------------------------------------
# Mock visibility, coverage, frontiers, and observations
# -----------------------------------------------------------------------------


def compute_patch_coverage_from_agents(
    patch_centers: np.ndarray,
    patch_normals: np.ndarray,
    agent_positions: np.ndarray,
    target_center: np.ndarray,
    fov_deg: float = 70.0,
    max_surface_angle_deg: float = 78.0,
    max_range: float = 12.0,
) -> np.ndarray:
    """
    Compute a soft coverage value for each ellipsoid patch from agent viewpoints.

    Assumption: every agent camera roughly points toward target_center.
    """
    coverage = np.zeros(patch_centers.shape[0], dtype=float)
    cos_fov = np.cos(np.deg2rad(fov_deg / 2.0))
    cos_surface = np.cos(np.deg2rad(max_surface_angle_deg))

    for agent_pos in agent_positions:
        camera_axis = normalize(target_center - agent_pos)
        ray_to_patch = normalize_rows(patch_centers - agent_pos[None, :])
        dist = np.linalg.norm(patch_centers - agent_pos[None, :], axis=1)

        # Patch should be inside camera cone.
        fov_quality = np.clip((ray_to_patch @ camera_axis - cos_fov) / (1.0 - cos_fov), 0.0, 1.0)

        # Agent should be on the outward-normal side of the patch.
        patch_to_agent = normalize_rows(agent_pos[None, :] - patch_centers)
        surface_quality = np.clip((np.sum(patch_normals * patch_to_agent, axis=1) - cos_surface) / (1.0 - cos_surface), 0.0, 1.0)

        range_quality = np.clip(1.0 - dist / max_range, 0.0, 1.0)
        candidate = fov_quality * surface_quality * range_quality

        # Multiple agents combine as independent coverage evidence.
        coverage = 1.0 - (1.0 - coverage) * (1.0 - candidate)

    return np.clip(coverage, 0.0, 1.0)


def generate_mock_partial_observations(
    target_points: np.ndarray,
    target_normals: np.ndarray,
    agent_positions: np.ndarray,
    target_center: np.ndarray,
    fov_deg: float = 75.0,
    max_surface_angle_deg: float = 85.0,
    max_points_per_agent: int = 450,
    seed: int = 19,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create mock observed point-cloud measurements from the current agents.

    Returns
    -------
    observed_points : (K, 3)
    observed_agent_ids : (K,)
    """
    rng = np.random.default_rng(seed)
    observed_batches = []
    agent_id_batches = []

    cos_fov = np.cos(np.deg2rad(fov_deg / 2.0))
    cos_surface = np.cos(np.deg2rad(max_surface_angle_deg))

    for i, agent_pos in enumerate(agent_positions):
        camera_axis = normalize(target_center - agent_pos)
        ray_to_point = normalize_rows(target_points - agent_pos[None, :])
        point_to_agent = normalize_rows(agent_pos[None, :] - target_points)

        in_fov = (ray_to_point @ camera_axis) > cos_fov
        visible_surface = np.sum(target_normals * point_to_agent, axis=1) > cos_surface
        mask = in_fov & visible_surface
        idx = np.flatnonzero(mask)

        if idx.size > max_points_per_agent:
            idx = rng.choice(idx, size=max_points_per_agent, replace=False)

        if idx.size > 0:
            noisy_obs = target_points[idx] + rng.normal(scale=0.01, size=(idx.size, 3))
            observed_batches.append(noisy_obs)
            agent_id_batches.append(np.full(idx.size, i, dtype=int))

    if not observed_batches:
        return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=int)

    return np.vstack(observed_batches), np.concatenate(agent_id_batches)


def find_frontier_patches(
    patch_coverage: np.ndarray,
    threshold: float = 0.15,
    max_frontiers: Optional[int] = 60,
) -> np.ndarray:
    """Return patch IDs with low coverage."""
    ids = np.flatnonzero(patch_coverage < threshold)
    if max_frontiers is not None and ids.size > max_frontiers:
        # Keep the most uncovered patches.
        order = np.argsort(patch_coverage[ids])
        ids = ids[order[:max_frontiers]]
    return ids


def assign_frontiers_greedily(
    agent_states: List[AgentState],
    patch_centers: np.ndarray,
    patch_coverage: np.ndarray,
    frontier_patch_ids: np.ndarray,
    distance_weight: float = 0.18,
    duplicate_penalty: float = 1.5,
) -> Dict[int, int]:
    """
    Greedy mock assignment of agents to frontier patches.

    Score is intentionally simple:
        high uncovered value, lower travel distance, avoid duplicate assignments.
    """
    assigned: Dict[int, int] = {}
    used: set[int] = set()

    if frontier_patch_ids.size == 0:
        return assigned

    for state in agent_states:
        agent_pos = state.position
        d = np.linalg.norm(patch_centers[frontier_patch_ids] - agent_pos[None, :], axis=1)
        info_gain = 1.0 - patch_coverage[frontier_patch_ids]
        penalty = np.array([duplicate_penalty if int(pid) in used else 0.0 for pid in frontier_patch_ids])
        score = info_gain - distance_weight * d - penalty
        selected = int(frontier_patch_ids[int(np.argmax(score))])
        assigned[state.agent_id] = selected
        used.add(selected)

    return assigned


# -----------------------------------------------------------------------------
# Mock contact points
# -----------------------------------------------------------------------------


def generate_mock_contact_points(seed: int = 23) -> List[ContactPoint]:
    """
    Generate plausible contact-point candidates on the satellite surfaces.

    In the real implementation, these should come from point-cloud segmentation,
    patching, normals, and scoring. Here they are deterministic mock affordances.
    """
    rng = np.random.default_rng(seed)

    raw = [
        # Bus side surfaces.
        ([0.90, 0.00, 0.00], [1, 0, 0], 0.55, "bus_+x"),
        ([-0.90, 0.00, 0.00], [-1, 0, 0], 0.55, "bus_-x"),
        ([0.00, 0.60, 0.00], [0, 1, 0], 0.50, "bus_+y"),
        ([0.00, -0.60, 0.00], [0, -1, 0], 0.50, "bus_-y"),
        ([0.00, 0.00, 0.50], [0, 0, 1], 0.70, "bus_+z"),
        # Solar panel central affordances.
        ([-2.25, 0.00, 0.02], [0, 0, 1], 1.2, "left_panel"),
        ([2.25, 0.00, 0.02], [0, 0, 1], 1.2, "right_panel"),
    ]

    contact_points: List[ContactPoint] = []
    for cp_id, (position, normal, area, label) in enumerate(raw):
        position_arr = np.asarray(position, dtype=float)
        normal_arr = normalize(np.asarray(normal, dtype=float))
        confidence = float(np.clip(0.80 + 0.10 * rng.normal(), 0.55, 0.98))
        contact_points.append(
            ContactPoint(
                cp_id=cp_id,
                position=position_arr,
                normal=normal_arr,
                patch_area=float(area),
                confidence=confidence,
                label=label,
            )
        )
    return contact_points


# -----------------------------------------------------------------------------
# Full scene builder
# -----------------------------------------------------------------------------


def build_mock_scene(
    num_agents: int = 4,
    num_steps: int = 100,
    step: int = 38,
    seed: int = 5,
) -> MockScene:
    """Build one complete synthetic scene for static visualization/debugging."""
    target_points, target_normals, target_labels = generate_mock_satellite_point_cloud(
        n_bus=2600,
        n_panels_each=850,
        noise_std=0.012,
        seed=seed + 1,
    )

    trajectories, velocities = generate_mock_agent_trajectories(
        num_agents=num_agents,
        num_steps=num_steps,
        orbit_radius=5.2,
        z_amplitude=1.3,
        seed=seed + 2,
    )
    agent_states = get_agent_states_at_step(trajectories, velocities, step=step)
    agent_positions = np.vstack([s.position for s in agent_states])

    ellipsoid = generate_mock_ellipsoid(
        center=[0.0, 0.0, 0.0],
        axes=[4.4, 3.0, 2.25],
        euler_rpy=[0.0, 0.0, 0.0],
    )
    patch_centers, patch_normals, _, _ = sample_ellipsoid_patches(
        ellipsoid,
        n_theta=38,
        n_phi=20,
    )

    patch_coverage = compute_patch_coverage_from_agents(
        patch_centers=patch_centers,
        patch_normals=patch_normals,
        agent_positions=agent_positions,
        target_center=ellipsoid.center,
        fov_deg=74.0,
        max_surface_angle_deg=78.0,
        max_range=12.0,
    )

    observed_points, observed_agent_ids = generate_mock_partial_observations(
        target_points=target_points,
        target_normals=target_normals,
        agent_positions=agent_positions,
        target_center=ellipsoid.center,
        seed=seed + 3,
    )

    frontier_patch_ids = find_frontier_patches(
        patch_coverage,
        threshold=0.12,
        max_frontiers=80,
    )
    assigned_patch_ids = assign_frontiers_greedily(
        agent_states=agent_states,
        patch_centers=patch_centers,
        patch_coverage=patch_coverage,
        frontier_patch_ids=frontier_patch_ids,
    )

    contact_points = generate_mock_contact_points(seed=seed + 4)

    return MockScene(
        target_points=target_points,
        target_normals=target_normals,
        target_labels=target_labels,
        trajectories=trajectories,
        velocities=velocities,
        agent_states=agent_states,
        ellipsoid=ellipsoid,
        patch_centers=patch_centers,
        patch_normals=patch_normals,
        patch_coverage=patch_coverage,
        observed_points=observed_points,
        observed_agent_ids=observed_agent_ids,
        frontier_patch_ids=frontier_patch_ids,
        assigned_patch_ids=assigned_patch_ids,
        contact_points=contact_points,
    )


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------


def set_axes_equal(ax: plt.Axes) -> None:
    """Set equal scale on a 3D Matplotlib axis."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_mock_scene(scene: MockScene, save_path: Optional[str] = None, show: bool = True) -> plt.Figure:
    """
    Create one static diagnostic figure for the generated mock scene.

    The figure contains:
        - 3D scene with target point cloud, ellipsoid coverage, agents, frontiers, CPs
        - coverage histogram
        - simple readiness/summary panel
    """
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.25, 1.0], height_ratios=[1.0, 1.0])

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, 1])

    # -----------------------------
    # 3D scene
    # -----------------------------
    tp = scene.target_points
    ax3d.scatter(tp[:, 0], tp[:, 1], tp[:, 2], s=1.0, alpha=0.22, label="Target point cloud")

    if scene.observed_points.size > 0:
        op = scene.observed_points
        ax3d.scatter(op[:, 0], op[:, 1], op[:, 2], s=2.5, alpha=0.65, label="Observed points")

    # Ellipsoid wireframe.
    xw, yw, zw = ellipsoid_wireframe(scene.ellipsoid, n_u=40, n_v=20)
    ax3d.plot_wireframe(xw, yw, zw, linewidth=0.35, alpha=0.28)

    # Coverage patches.
    coverage = scene.patch_coverage
    patch_colors = cm.viridis(coverage)
    pc = scene.patch_centers
    ax3d.scatter(
        pc[:, 0], pc[:, 1], pc[:, 2],
        s=18,
        c=patch_colors,
        alpha=0.88,
        label="Ellipsoid coverage patches",
    )

    # Frontier patches.
    if scene.frontier_patch_ids.size > 0:
        fp = scene.patch_centers[scene.frontier_patch_ids]
        ax3d.scatter(fp[:, 0], fp[:, 1], fp[:, 2], s=35, marker="x", label="Uncovered frontiers")

    # Agent trajectories and current states.
    trajectories = scene.trajectories
    for i, state in enumerate(scene.agent_states):
        traj_i = trajectories[:, i, :]
        ax3d.plot(traj_i[:, 0], traj_i[:, 1], traj_i[:, 2], linewidth=0.75, alpha=0.35)
        p = state.position
        v = state.velocity
        ax3d.scatter([p[0]], [p[1]], [p[2]], s=70, marker="o", label=f"Agent {i}" if i == 0 else None)
        ax3d.quiver(p[0], p[1], p[2], v[0], v[1], v[2], length=4.5, normalize=True, linewidth=1.0)

        # Line to assigned frontier target.
        if state.agent_id in scene.assigned_patch_ids:
            pid = scene.assigned_patch_ids[state.agent_id]
            q = scene.patch_centers[pid]
            ax3d.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], linestyle="--", linewidth=1.0, alpha=0.8)
            ax3d.scatter([q[0]], [q[1]], [q[2]], s=60, marker="*")

    # Contact points and normals.
    for cp in scene.contact_points:
        p = cp.position
        n = cp.normal
        ax3d.scatter([p[0]], [p[1]], [p[2]], s=90, marker="D", label="Contact points" if cp.cp_id == 0 else None)
        ax3d.quiver(p[0], p[1], p[2], n[0], n[1], n[2], length=0.55, normalize=True, linewidth=1.5)

    ax3d.set_title("Mock swarm active-SLAM/capture scene")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.view_init(elev=23, azim=-48)
    ax3d.legend(loc="upper left", fontsize=8)
    set_axes_equal(ax3d)

    # Add colorbar for coverage.
    mappable = cm.ScalarMappable(cmap=cm.viridis)
    mappable.set_array(coverage)
    cbar = fig.colorbar(mappable, ax=ax3d, shrink=0.58, pad=0.02)
    cbar.set_label("Patch coverage")

    # -----------------------------
    # Coverage histogram
    # -----------------------------
    ax_hist.hist(coverage, bins=20, range=(0.0, 1.0), alpha=0.85)
    ax_hist.axvline(0.12, linestyle="--", linewidth=1.0, label="frontier threshold")
    ax_hist.set_title("Ellipsoid patch coverage distribution")
    ax_hist.set_xlabel("coverage")
    ax_hist.set_ylabel("patch count")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.25)

    # -----------------------------
    # Info panel
    # -----------------------------
    coverage_ratio = float(np.mean(coverage > 0.12))
    mean_coverage = float(np.mean(coverage))
    uncovered_ratio = float(1.0 - coverage_ratio)
    n_contact_points = len(scene.contact_points)
    n_agents = len(scene.agent_states)
    observed_count = int(scene.observed_points.shape[0])
    readiness_score = float(
        np.clip(
            0.45 * coverage_ratio
            + 0.25 * np.clip(n_contact_points / max(n_agents, 1), 0.0, 1.0)
            + 0.15 * np.clip(observed_count / 1200.0, 0.0, 1.0)
            + 0.15 * (1.0 - uncovered_ratio),
            0.0,
            1.0,
        )
    )

    ax_info.axis("off")
    summary_lines = [
        "Mock scene summary",
        "",
        f"Agents: {n_agents}",
        f"Target cloud points: {scene.target_points.shape[0]}",
        f"Observed points: {observed_count}",
        f"Ellipsoid patches: {scene.patch_centers.shape[0]}",
        f"Frontier patches: {scene.frontier_patch_ids.size}",
        f"Contact candidates: {n_contact_points}",
        "",
        f"Mean coverage: {mean_coverage:.3f}",
        f"Covered patch ratio: {coverage_ratio:.3f}",
        f"Uncovered patch ratio: {uncovered_ratio:.3f}",
        f"Mock readiness score: {readiness_score:.3f}",
        "",
        "Interpretation:",
        "- bright ellipsoid points: covered viewpoints",
        "- x markers: uncovered frontier patches",
        "- stars/lines: current frontier assignments",
        "- diamonds/arrows: contact points + normals",
    ]
    ax_info.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved static mock-data preview to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mock data for swarm active-SLAM/capture debugging.")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of swarm agents.")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of trajectory steps.")
    parser.add_argument("--step", type=int, default=38, help="Time step used for the static visualization.")
    parser.add_argument("--seed", type=int, default=5, help="Random seed.")
    parser.add_argument("--save", type=str, default="mock_data_preview.png", help="Path for saved preview image.")
    parser.add_argument("--no-show", action="store_true", help="Save figure without opening a plot window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scene = build_mock_scene(
        num_agents=args.num_agents,
        num_steps=args.num_steps,
        step=args.step,
        seed=args.seed,
    )

    print("MockScene generated")
    print(f"  target_points:      {scene.target_points.shape}")
    print(f"  target_normals:     {scene.target_normals.shape}")
    print(f"  trajectories:       {scene.trajectories.shape}")
    print(f"  velocities:         {scene.velocities.shape}")
    print(f"  ellipsoid axes:     {scene.ellipsoid.axes}")
    print(f"  patches:            {scene.patch_centers.shape}")
    print(f"  observed_points:    {scene.observed_points.shape}")
    print(f"  frontier_patches:   {scene.frontier_patch_ids.shape}")
    print(f"  contact_points:     {len(scene.contact_points)}")

    plot_mock_scene(scene, save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
