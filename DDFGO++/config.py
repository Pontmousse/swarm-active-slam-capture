"""
Centralized configuration for standalone DDFGO++.
All runtime controls should be defined here and consumed by SwarmDDFGO++.py.
"""

from pathlib import Path
import os
import sys
import numpy as np

#############################################################################################
# Paths
#############################################################################################
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
DATA_DIR = PROJECT_ROOT / "Data" / "Dynamic_Target"
RESULTS_DIR = MODULE_DIR / "Results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import shared_config

#############################################################################################
# Simulation Parameters
#############################################################################################
DT = shared_config.DT
N = shared_config.N
D = shared_config.D
object_name = shared_config.object_name

#############################################################################################
# Runtime / Mode Controls (multi-agent + SLAM)
#############################################################################################
history_downsample_step = 1
animation_downsample_step = 1

sw = 30
Max_Land = 20
unc = 0.6

#############################################################################################
# Feature and dense-map scaffolding
#############################################################################################
DESCRIPTOR_DIM = 33
USE_LEGACY_RANDOM_SAMPLING = False
USE_RANDOM_FEATURE_FILL = True
# ID namespace policy for graph/communication descriptors:
# - "registry": use LandmarkRegistry IDs everywhere
# - "source_feature_idx": use source FeatureIdxSet (or selected index fallback)
# - "auto": registry for single-agent, source_feature_idx for multi-agent
feature_id_namespace_policy = "registry"

map_mode = "hybrid"  # sparse, dense, hybrid
voxel_size = 0.01
icp_threshold = 1.5
scan_downsample_voxel_size = 0.01
feature_noise_std = None  # None => use Feature_Processing default

# Post-optimization outlier rejection (Phase C)
enable_outlier_rejection = True
outlier_max_distance_from_com = 20.0
outlier_max_stale_frames = 150000

#############################################################################################
# Animation visualization (render-only; does not affect SLAM)
#############################################################################################

# Agent whose dense merged map is shown in multi-agent animations (1-indexed).
animation_selected_agent_id = 2

# Dark background for demo/presentation videos.
animation_dark_background = True
animation_background_rgba = (0.90, 0.92, 0.96, 1.0)

# Sparse landmark rendering as spheres.
animation_sparse_sphere_radius = 0.3
animation_sparse_sphere_resolution = 10

# Dense merged map visualization controls.
# - voxel downsampling here is for rendering clarity/perf only.
animation_dense_voxel_size = voxel_size * 8.0
animation_dense_point_size = 2.0

# Target truth visualization downsampling for animation.
animation_target_voxel_size = voxel_size * 12.0

# Feature observations (instantaneous) visualization controls.
animation_observation_point_size = 4.0

# Optional per-script toggles (scripts may override locally).
animation_show_sparse_map = True
animation_show_dense_map = True
animation_show_observations = False
animation_show_target_truth = True
animation_cubesat_size_m = float(shared_config.VIS_CUBESAT_SIZE_M)

#############################################################################
# SLAM optimization controls
#############################################################################

Verbose = False
Verbose_map = False
Verbose_kinem = False
Update_Sliding_Window = False

Init_Pose_Only = False
Odom = True
Kinem = "n_step_Kinem"

Decentralized = True
Qn = 2
Ql = 5
kappa_seconds = 1.0  # cadence for communication/registration

FGO_Param = False
calculate_w_method = "w4"  # options: w1, w2, w3, w4
low_pass_filter_coeff = 0.1
relinearize_threshold = 0.1
relinearize_skip = 1

# Prior strategy (window anchor + sparse GPS)
use_window_anchor_prior = False
window_anchor_stride = 25
window_anchor_rpy_sigma = 30.0
window_anchor_xyz_sigma = 2.0
use_sparse_gps_prior = False
gps_stride = 30
gps_rpy_sigma_weak = 15.0
gps_xyz_sigma_weak = 1.0

#############################################################################################
# Robust hooks and priors (Phase A wiring; defaults preserve existing behavior)
#############################################################################################
use_robust_bearing = False
use_robust_kinematic = False
huber_param = 1.345
cauchy_param = 1.0
landmark_prior_sigma = 2.5

#############################################################################################
# Noise Parameters (base values, scaled by unc)
#############################################################################################
prior_rpy_sigma_base = 2.0
prior_xyz_sigma_base = 0.05

odometry_rpy_sigma_base = 0.1
odometry_xyz_sigma_base = 0.01

bearing_sigma_base = 0.5
range_sigma_base = 0.01
kinematic_sigma_base = 0.001
pvw_sigma_base = 0.001

target_noise_std = np.array([
    0.05, 0.05, 0.05,
    0.005, 0.005, 0.005,
    0.005, 0.005, 0.005,
])
target_noise_bias = np.array([2.0, 1.0, 1.0])

#############################################################################################
# Saving / notification controls
#############################################################################################
save_num_chunks = 10
enable_notify = False
notify_topic = "DDFGO-standalone"

# Quick execution profile
# Usage:
#   DDFGO_PROFILE=smoke python SwarmDDFGO++.py
run_profile = os.getenv("DDFGO_PROFILE", "default").strip().lower()
if run_profile == "smoke":
    # Keep dataset identity (N/D/DT/object_name) unchanged so files still resolve,
    # but reduce optimization load for fast sanity checks.
    history_downsample_step = 120
    sw = 8
    Max_Land = 12
    Ql = min(Ql, 3)
    map_mode = "sparse"
    save_num_chunks = 20
    Verbose = True


#############################################################################################
# Helper Functions
#############################################################################################
def get_tag():
    return shared_config.get_tag(n=N, d=D, dt=DT, name=object_name)


def get_dec_tag():
    if Decentralized:
        return f"Multi_Qn{Qn}_Ql{Ql}"
    return "No_Multi"


def get_results_tag():
    return f"{get_tag()}_{Kinem}_{get_dec_tag()}"


def get_data_paths():
    return shared_config.get_sim_data_paths(n=N, d=D, dt=DT, name=object_name)


def get_results_paths():
    tag = get_results_tag()
    return {
        "agents": str(RESULTS_DIR / f"Agents_History_{tag}.pkl"),
        "target": str(RESULTS_DIR / f"Target_History_{tag}.pkl"),
    }

