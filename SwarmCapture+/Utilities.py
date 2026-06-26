import sys
import types
from dataclasses import dataclass

import matplotlib
import numpy as np

from shared_config import PROJECT_ROOT


@dataclass
class EllipsoidModel:
    center: np.ndarray
    axes: np.ndarray
    rotation: np.ndarray


def _as_points_array(points):
    if points is None:
        return np.array([]).reshape(0, 3)
    points = np.asarray(points)
    if points.size == 0:
        return np.array([]).reshape(0, 3)
    return points.reshape(-1, 3)


def _register_mock_data_shim_if_needed(utilities_dir):
    mock_data_path = utilities_dir / "data" / "mock_data.py"
    if mock_data_path.is_file():
        return
    if "data.mock_data" in sys.modules:
        return

    mod = types.ModuleType("data.mock_data")
    mod.EllipsoidModel = EllipsoidModel

    def _cli_only(*_args, **_kwargs):
        raise NotImplementedError(
            "utilities/data/mock_data.py is required for coverage CLI demos"
        )

    mod.generate_mock_satellite_point_cloud = _cli_only
    mod.ellipsoid_wireframe = _cli_only
    mod.set_axes_equal = _cli_only
    sys.modules["data.mock_data"] = mod


def _load_coverage_modules():
    utilities_dir = PROJECT_ROOT / "utilities"
    coverage_dir = utilities_dir / "coverage"
    for path in (coverage_dir, utilities_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    matplotlib.use("Agg")
    _register_mock_data_shim_if_needed(utilities_dir)

    from ellipsoid import fit_viewpoint_ellipsoid_pca, estimate_point_normals
    from coverage import (
        create_ellipsoid_patches,
        update_coverage_from_observed_points,
        compute_coverage_ratio,
        get_uncovered_patches,
    )

    return {
        "fit_viewpoint_ellipsoid_pca": fit_viewpoint_ellipsoid_pca,
        "estimate_point_normals": estimate_point_normals,
        "create_ellipsoid_patches": create_ellipsoid_patches,
        "update_coverage_from_observed_points": update_coverage_from_observed_points,
        "compute_coverage_ratio": compute_coverage_ratio,
        "get_uncovered_patches": get_uncovered_patches,
    }


_COVERAGE = None


def _coverage():
    global _COVERAGE
    if _COVERAGE is None:
        _COVERAGE = _load_coverage_modules()
    return _COVERAGE


def _update_map_coverage_and_explore(Spacecraft, target_com=None, min_points=10, explore_offset_distance=0.0):
    cov = _coverage()
    pts = _as_points_array(Spacecraft.get("MergedMapSet"))
    if len(pts) < min_points:
        Spacecraft["MapCoverageRatio"] = 0.0
        Spacecraft["MapEllipsoid"] = None
        Spacecraft["MapCoveragePatches"] = []
        Spacecraft["MapExploreTarget"] = None
        Spacecraft["MapExploreDirection"] = np.array([])
        return

    agent_pos = np.asarray(Spacecraft["State"][:3], dtype=float)
    com = np.asarray(target_com if target_com is not None else pts.mean(axis=0), dtype=float)[:3]

    normals = cov["estimate_point_normals"](pts, camera_location=agent_pos)
    ellipsoid = cov["fit_viewpoint_ellipsoid_pca"](pts, center=com, margin=1.1, min_axis=0.5)
    patches = cov["create_ellipsoid_patches"](ellipsoid, n_theta=24, n_phi=12)
    patches, _ = cov["update_coverage_from_observed_points"](
        patches, pts, normals, agent_pos, ellipsoid, projection_mode="normal"
    )

    Spacecraft["MapEllipsoid"] = ellipsoid
    Spacecraft["MapCoveragePatches"] = patches
    Spacecraft["MapCoverageRatio"] = float(cov["compute_coverage_ratio"](patches))

    uncovered = cov["get_uncovered_patches"](patches)
    if not uncovered:
        Spacecraft["MapExploreTarget"] = None
        Spacecraft["MapExploreDirection"] = np.array([])
        return

    pole_margin = 0.15
    selectable = [
        p for p in uncovered
        if pole_margin <= float(p.phi) <= np.pi - pole_margin
    ]
    if not selectable:
        selectable = uncovered

    centers = np.array([p.center for p in selectable])
    dists = np.linalg.norm(centers - agent_pos, axis=1)
    best = selectable[int(np.argmin(dists))]

    target = np.asarray(best.center, dtype=float) + float(explore_offset_distance) * np.asarray(best.normal, dtype=float)
    Spacecraft["MapExploreTarget"] = target
    direction = target - agent_pos
    n = np.linalg.norm(direction)
    Spacecraft["MapExploreDirection"] = direction / n if n > 1e-12 else np.array([])
