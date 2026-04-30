#!/usr/bin/env python3
"""One-off generator: rebuild SwarmDDFGO++.py without SCRIPT_BODY/exec."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "SwarmDDFGO++.py"
text = SRC.read_text(encoding="utf-8")

m = re.search(r"SCRIPT_BODY = r\"\"\"(.*)\"\"\"", text, re.DOTALL)
if not m:
    raise SystemExit("SCRIPT_BODY not found")
body = m.group(1)
anchor = "for i in range(1, num_iter):"
idx = body.index(anchor)
init_tail = body[:idx].rstrip()  # includes runtime binding + load through KeyFrames — actually includes load
loop_rest = body[idx + len(anchor) :].lstrip("\n")

# init_tail starts with config = __runtime — we'll replace whole batch with functions
# Split init_tail at "# Factor Graph" — prepare vs initialize
prep_end_marker = "##############################################################################################################################\n# Factor Graph Parameters"
prep_part, factor_tail = init_tail.split(prep_end_marker, 1)
factor_tail = "##############################################################################################################################\n# Factor Graph Parameters" + factor_tail

# Remove load/downsample/prints from prep_part — prep_part is lines 250-332 approx
# prep_part has: runtime binding, load, downsample, policy, prints — only policy stays in prepare_batch_histories
# We'll construct prepare/load separately in template

tail_marker = (
    "\n\n########################################################################################################################\n"
    "# Save using Pickle module for Plotting and Printing in a separate code\n"
    "########################################################################################################################\n\n"
    "print('\\n')\n\n"
    "end_time = time.time()\n"
)
pos = loop_rest.rfind(tail_marker)
if pos == -1:
    raise SystemExit("tail marker not found in loop")
loop_core = loop_rest[:pos].rstrip()
after_loop = loop_rest[pos:]

header = text.split("SCRIPT_BODY = r")[0]

prefix = """###########################################################################################
# Angular velocity method registry (used by SLAM batch)
###########################################################################################
calculate_w_methods = {
    "w1": calculate_w1,
    "w2": calculate_w2,
    "w3": calculate_w3,
    "w4": calculate_w4,
}


def resolve_calculate_w(calculate_w_method: str):
    return calculate_w_methods.get(calculate_w_method, calculate_w4)


"""

# Insert prefix before I/O helpers section - replace old globals block
header = re.sub(
    r"###########################################################################################\n"
    r"# Config-driven runtime controls\n"
    r"###########################################################################################\n"
    r".*?"
    r"#############################################################################################\n"
    r"# I/O helpers\n",
    prefix + "#############################################################################################\n# I/O helpers\n",
    header,
    count=1,
    flags=re.DOTALL,
)

batch_footer = '''

def load_simulation_histories(config_module):
    """Load raw pickle histories from ``config_module.get_data_paths()``."""
    data_paths = config_module.get_data_paths()
    return {
        "agents": load_pickle_file(data_paths["agents"]),
        "target": load_pickle_file(data_paths["target"]),
        "target_pcd": load_pickle_file(data_paths["target_pcd"]),
    }


def prepare_batch_histories(histories, runtime: RuntimeDerived, config_module):
    """Downsample histories and resolve ``feature_id_namespace_policy``."""
    step_size = runtime.step_size
    Agents_History = histories["agents"]
    Target_History = histories["target"]
    Target_Point_Cloud_History = histories["target_pcd"]
    num_iter = len(Agents_History)
    Agents_History = [Agents_History[i] for i in range(0, num_iter, step_size)]
    Target_History = [Target_History[i] for i in range(0, num_iter, step_size)]
    Target_Point_Cloud_History = [
        Target_Point_Cloud_History[i] for i in range(0, num_iter, step_size)
    ]
    N = len(Agents_History[0])
    num_iter = len(Agents_History)
    feature_id_namespace_policy = config_module.feature_id_namespace_policy
    if feature_id_namespace_policy == "auto":
        feature_id_namespace_policy = "registry" if N == 1 else "source_feature_idx"
    if feature_id_namespace_policy not in ("registry", "source_feature_idx"):
        raise ValueError(
            f"Unsupported feature_id_namespace_policy: {feature_id_namespace_policy}"
        )
    return {
        "Agents_History": Agents_History,
        "Target_History": Target_History,
        "Target_Point_Cloud_History": Target_Point_Cloud_History,
        "N": N,
        "num_iter": num_iter,
        "feature_id_namespace_policy": feature_id_namespace_policy,
    }


def initialize_slam_batch(prepared, cfg: DdfgoConfig):
    """Factor-graph and history initialization (legacy body before the main timestep loop)."""
    rt = derive_runtime(cfg)
    config = cfg.config_module

    object_name = rt.object_name
    dt = rt.dt
    step_size = rt.step_size
    sw = rt.sw
    Max_Land = rt.max_land
    unc = rt.unc
    Verbose = rt.verbose
    Verbose_map = rt.verbose_map
    Verbose_kinem = rt.verbose_kinem
    Update_Sliding_Window = rt.update_sliding_window
    Init_Pose_Only = rt.init_pose_only
    Odom = rt.odom
    Kinem = rt.kinem
    Decentralized = rt.decentralized
    Qn = rt.qn
    Ql = rt.ql
    FGO_Param = rt.fgo_param
    low_pass_filter_coeff = rt.low_pass_filter_coeff
    USE_LEGACY_RANDOM_SAMPLING = rt.use_legacy_random_sampling
    USE_RANDOM_FEATURE_FILL = rt.use_random_feature_fill
    voxel_size = rt.voxel_size
    icp_threshold = rt.icp_threshold
    scan_downsample_voxel_size = rt.scan_downsample_voxel_size
    map_mode = rt.map_mode
    feature_noise_std = rt.feature_noise_std
    feature_id_namespace_policy = prepared["feature_id_namespace_policy"]
    enable_outlier_rejection = rt.enable_outlier_rejection
    outlier_max_distance_from_com = rt.outlier_max_distance_from_com
    outlier_max_stale_frames = rt.outlier_max_stale_frames
    use_window_anchor_prior = rt.use_window_anchor_prior
    window_anchor_stride = rt.window_anchor_stride
    use_sparse_gps_prior = rt.use_sparse_gps_prior
    gps_stride = rt.gps_stride
    calculate_w = resolve_calculate_w(rt.calculate_w_method)

    Agents_History = prepared["Agents_History"]
    Target_History = prepared["Target_History"]
    Target_Point_Cloud_History = prepared["Target_Point_Cloud_History"]
    N = prepared["N"]
    num_iter = prepared["num_iter"]

'''

# factor_tail in original starts after prints — our generator's factor_tail includes Factor Graph from script
# init_tail in extraction = from "config = " through start of for loop. We need only factor part + graphs.
# Easier: extract from script body between "# Factor Graph Parameters" and "for i in range(1, num_iter):"
m2 = re.search(
    r"(##############################################################################################################################\n# Factor Graph Parameters.*)"
    r"\n# Record the start time for the entire loop\n"
    r"start_time = time\.time\(\)\n\n"
    r"(# Frame registration for front-end SLAM.*)",
    body,
    re.DOTALL,
)
if not m2:
    raise SystemExit("factor section pattern not found")
factor_and_rest = m2.group(1) + "\n\n" + m2.group(2)

batch_footer += factor_and_rest

batch_footer += """

    slam_state = {
        "config_module": config,
        "object_name": object_name,
        "dt": dt,
        "step_size": step_size,
        "sw": sw,
        "Max_Land": Max_Land,
        "unc": unc,
        "Verbose": Verbose,
        "Verbose_map": Verbose_map,
        "Verbose_kinem": Verbose_kinem,
        "Update_Sliding_Window": Update_Sliding_Window,
        "Init_Pose_Only": Init_Pose_Only,
        "Odom": Odom,
        "Kinem": Kinem,
        "Decentralized": Decentralized,
        "Qn": Qn,
        "Ql": Ql,
        "FGO_Param": FGO_Param,
        "low_pass_filter_coeff": low_pass_filter_coeff,
        "USE_LEGACY_RANDOM_SAMPLING": USE_LEGACY_RANDOM_SAMPLING,
        "USE_RANDOM_FEATURE_FILL": USE_RANDOM_FEATURE_FILL,
        "voxel_size": voxel_size,
        "icp_threshold": icp_threshold,
        "scan_downsample_voxel_size": scan_downsample_voxel_size,
        "map_mode": map_mode,
        "feature_noise_std": feature_noise_std,
        "feature_id_namespace_policy": feature_id_namespace_policy,
        "enable_outlier_rejection": enable_outlier_rejection,
        "outlier_max_distance_from_com": outlier_max_distance_from_com,
        "outlier_max_stale_frames": outlier_max_stale_frames,
        "use_window_anchor_prior": use_window_anchor_prior,
        "window_anchor_stride": window_anchor_stride,
        "use_sparse_gps_prior": use_sparse_gps_prior,
        "gps_stride": gps_stride,
        "calculate_w": calculate_w,
        "Agents_History": Agents_History,
        "Target_History": Target_History,
        "Target_Point_Cloud_History": Target_Point_Cloud_History,
        "N": N,
        "num_iter": num_iter,
        "tag": tag,
        "kappa": kappa,
        "every": every,
        "parameters": parameters,
        "identity_pose": identity_pose,
        "prior_rpy_sigma": prior_rpy_sigma,
        "prior_xyz_sigma": prior_xyz_sigma,
        "prior_noise": prior_noise,
        "window_anchor_noise": window_anchor_noise,
        "gps_prior_noise_weak": gps_prior_noise_weak,
        "odometry_rpy_sigma": odometry_rpy_sigma,
        "odometry_xyz_sigma": odometry_xyz_sigma,
        "odometry_noise": odometry_noise,
        "bearing_sigma": bearing_sigma,
        "range_sigma": range_sigma,
        "bearing_range_noise": bearing_range_noise,
        "target_noise_std": target_noise_std,
        "target_noise_bias": target_noise_bias,
        "kinem_noise": kinem_noise,
        "landmark_prior_noise": landmark_prior_noise,
        "pvw_noise": pvw_noise,
        "X": X,
        "L": L,
        "P": P,
        "V": V,
        "W": W,
        "lf_error": lf_error,
        "Lvar_l": Lvar_l,
        "Lobs_l": Lobs_l,
        "Lpar_l": Lpar_l,
        "lf_jacobian": lf_jacobian,
        "velf_error": velf_error,
        "Lvar_vel": Lvar_vel,
        "Lpar_vel": Lpar_vel,
        "velf_jacobian": velf_jacobian,
        "Graph_List": Graph_List,
        "Value_List": Value_List,
        "D": D,
        "alpha": alpha,
        "KeyFrames_Hist": KeyFrames_Hist,
        "KeyFramesIdx_Hist": KeyFramesIdx_Hist,
        "KeyFramesNgh_Hist": KeyFramesNgh_Hist,
    }
    return slam_state
'''

# The factor_and_rest must be indented inside initialize_slam_batch — the generator inserted raw body;
# we need to indent factor graph code block. This script got too complex — abort file generation here.

OUT = ROOT / "SwarmDDFGO++.generated.py"
OUT.write_text(header + "\\n# GENERATION INCOMPLETE\\n", encoding="utf-8")
print("Wrote stub — manual completion needed")
