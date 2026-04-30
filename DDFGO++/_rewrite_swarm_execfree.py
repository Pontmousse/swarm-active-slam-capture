#!/usr/bin/env python3
"""Rewrite SwarmDDFGO++.py: strip SCRIPT_BODY/exec; emit batch functions."""
import re
from pathlib import Path
from textwrap import indent

ROOT = Path(__file__).resolve().parent
src_path = ROOT / "SwarmDDFGO++.py"
text = src_path.read_text(encoding="utf-8")

m = re.search(r"SCRIPT_BODY = r\"\"\"(.*)\"\"\"\s*\n", text, re.DOTALL)
if not m:
    raise SystemExit("SCRIPT_BODY not found")
body = m.group(1)

# --- split body ---
config_block, rest = body.split("#############################################################################################\n# Load Simulation History", 1)
# drop config_block (replaced by prepare/init)
_, rest = rest.split("Target_Point_Cloud_History = load_pickle_file", 1)
# from downsample
rest = "Target_Point_Cloud_History = load_pickle_file" + rest
# find start of downsample comment
down_start = rest.index("###########################################################################################\n# Downsample History")
_, after_down = rest[down_start:].split("if feature_id_namespace_policy == \"auto\":", 1)
policy_and_rest = "if feature_id_namespace_policy == \"auto\":" + after_down
# policy block until prints
pol_end = policy_and_rest.index("print(f\"\\nNumber of agents:")
policy_block, mid = policy_and_rest[:pol_end], policy_and_rest[pol_end:]
# mid: prints + factor graph
fg_start = mid.index("##############################################################################################################################\n# Factor Graph Parameters")
prints_block, factor_block = mid[:fg_start], mid[fg_start:]

# Remove start_time from factor tail — locate "# Record the start time"
st_marker = "\n\n# Record the start time for the entire loop\nstart_time = time.time()\n\n"
if st_marker not in factor_block:
    raise SystemExit("start_time marker missing")
factor_block = factor_block.replace(st_marker, "\n\n")

# Split factor_block at loop
anchor = "for i in range(1, num_iter):"
if anchor not in factor_block:
    raise SystemExit("loop anchor missing")
init_suffix, loop_with = factor_block.split(anchor, 1)
loop_core = loop_with.lstrip("\n")

# Split loop_core from final tail (after last periodic notify)
tail_pat = (
    "\n\n########################################################################################################################\n"
    "# Save using Pickle module for Plotting and Printing in a separate code\n"
    "########################################################################################################################\n\n"
    "print('\\n')\n\n"
    "end_time = time.time()\n"
)
ti = loop_core.find(tail_pat)
if ti == -1:
    raise SystemExit("final tail pattern missing")
loop_only = loop_core[:ti].rstrip()
after_loop = loop_core[ti:]

# Indent helpers
def ind(s: str, n: int = 4) -> str:
    return indent(s, " " * n)


HEADER_INIT = '''def initialize_slam_batch(prepared, cfg: DdfgoConfig):
    """Factor-graph init (legacy SCRIPT_BODY before timestep loop; no start_time)."""
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

SLAM_STATE_DICT = """
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
"""

TIMESTEP_UNPACK = '''def run_slam_timestep(slam_state, i):
    config = slam_state["config_module"]
    object_name = slam_state["object_name"]
    dt = slam_state["dt"]
    step_size = slam_state["step_size"]
    sw = slam_state["sw"]
    Max_Land = slam_state["Max_Land"]
    unc = slam_state["unc"]
    Verbose = slam_state["Verbose"]
    Verbose_map = slam_state["Verbose_map"]
    Verbose_kinem = slam_state["Verbose_kinem"]
    Update_Sliding_Window = slam_state["Update_Sliding_Window"]
    Init_Pose_Only = slam_state["Init_Pose_Only"]
    Odom = slam_state["Odom"]
    Kinem = slam_state["Kinem"]
    Decentralized = slam_state["Decentralized"]
    Qn = slam_state["Qn"]
    Ql = slam_state["Ql"]
    FGO_Param = slam_state["FGO_Param"]
    low_pass_filter_coeff = slam_state["low_pass_filter_coeff"]
    USE_LEGACY_RANDOM_SAMPLING = slam_state["USE_LEGACY_RANDOM_SAMPLING"]
    USE_RANDOM_FEATURE_FILL = slam_state["USE_RANDOM_FEATURE_FILL"]
    voxel_size = slam_state["voxel_size"]
    icp_threshold = slam_state["icp_threshold"]
    scan_downsample_voxel_size = slam_state["scan_downsample_voxel_size"]
    map_mode = slam_state["map_mode"]
    feature_noise_std = slam_state["feature_noise_std"]
    feature_id_namespace_policy = slam_state["feature_id_namespace_policy"]
    enable_outlier_rejection = slam_state["enable_outlier_rejection"]
    outlier_max_distance_from_com = slam_state["outlier_max_distance_from_com"]
    outlier_max_stale_frames = slam_state["outlier_max_stale_frames"]
    use_window_anchor_prior = slam_state["use_window_anchor_prior"]
    window_anchor_stride = slam_state["window_anchor_stride"]
    use_sparse_gps_prior = slam_state["use_sparse_gps_prior"]
    gps_stride = slam_state["gps_stride"]
    calculate_w = slam_state["calculate_w"]
    Agents_History = slam_state["Agents_History"]
    Target_History = slam_state["Target_History"]
    Target_Point_Cloud_History = slam_state["Target_Point_Cloud_History"]
    N = slam_state["N"]
    num_iter = slam_state["num_iter"]
    tag = slam_state["tag"]
    kappa = slam_state["kappa"]
    every = slam_state["every"]
    parameters = slam_state["parameters"]
    identity_pose = slam_state["identity_pose"]
    prior_rpy_sigma = slam_state["prior_rpy_sigma"]
    prior_xyz_sigma = slam_state["prior_xyz_sigma"]
    prior_noise = slam_state["prior_noise"]
    window_anchor_noise = slam_state["window_anchor_noise"]
    gps_prior_noise_weak = slam_state["gps_prior_noise_weak"]
    odometry_rpy_sigma = slam_state["odometry_rpy_sigma"]
    odometry_xyz_sigma = slam_state["odometry_xyz_sigma"]
    odometry_noise = slam_state["odometry_noise"]
    bearing_sigma = slam_state["bearing_sigma"]
    range_sigma = slam_state["range_sigma"]
    bearing_range_noise = slam_state["bearing_range_noise"]
    target_noise_std = slam_state["target_noise_std"]
    target_noise_bias = slam_state["target_noise_bias"]
    kinem_noise = slam_state["kinem_noise"]
    landmark_prior_noise = slam_state["landmark_prior_noise"]
    pvw_noise = slam_state["pvw_noise"]
    X = slam_state["X"]
    L = slam_state["L"]
    P = slam_state["P"]
    V = slam_state["V"]
    W = slam_state["W"]
    lf_error = slam_state["lf_error"]
    Lvar_l = slam_state["Lvar_l"]
    Lobs_l = slam_state["Lobs_l"]
    Lpar_l = slam_state["Lpar_l"]
    lf_jacobian = slam_state["lf_jacobian"]
    velf_error = slam_state["velf_error"]
    Lvar_vel = slam_state["Lvar_vel"]
    Lpar_vel = slam_state["Lpar_vel"]
    velf_jacobian = slam_state["velf_jacobian"]
    Graph_List = slam_state["Graph_List"]
    Value_List = slam_state["Value_List"]
    D = slam_state["D"]
    alpha = slam_state["alpha"]
    KeyFrames_Hist = slam_state["KeyFrames_Hist"]
    KeyFramesIdx_Hist = slam_state["KeyFramesIdx_Hist"]
    KeyFramesNgh_Hist = slam_state["KeyFramesNgh_Hist"]
    start_time = slam_state["start_time"]
'''

LOAD_PREP = '''
def load_simulation_histories(config_module):
    """Load raw simulation pickles from ``config_module.get_data_paths()``."""
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

'''

# Rename shadowing `runtime` in after_loop tail (duration variable)
after_loop_fixed = after_loop.replace(
    "runtime = end_time-start_time", "runtime_sec = end_time - start_time"
).replace("hours, remainder = divmod(runtime, 3600)", "hours, remainder = divmod(runtime_sec, 3600)")

FINAL_SAVE_PRINT = "print('Saving files using Pickle...')"
if FINAL_SAVE_PRINT not in after_loop_fixed:
    raise SystemExit("final save print not found in tail")
timing_only = after_loop_fixed.split(FINAL_SAVE_PRINT, 1)[0].rstrip() + "\n"
# Batch duration uses slam_state clock set in run_slam_batch (not iteration locals)
timing_only = timing_only.replace(
    "runtime_sec = end_time - start_time",
    "runtime_sec = end_time - slam_state['start_time']",
)

SAVE_BATCH = '''
def save_batch_results(slam_state, config_module):
    """Final pickle save via ``get_results_paths()`` (same contract as legacy tail)."""
    tag = slam_state["tag"]
    Agents_History = slam_state["Agents_History"]
    Target_History = slam_state["Target_History"]
    N = slam_state["N"]
    num_iter = slam_state["num_iter"]
    print("Saving files using Pickle...")
    save_pickle_files(Agents_History, Target_History, config_module)
    print("\\nTwo pickle files saved")
    if config_module.enable_notify:
        notify(
            f"DDFGO++ completed | Agents={N}, Iter={num_iter}, Tag={tag}",
            title="DDFGO++ Complete",
            topic=config_module.notify_topic,
            verbose=True,
        )
    print("\\n")
    print("%" * 30)
    print("iSAM Computation Finished !!!!")
    print("%" * 30)
    print("\\n")
    print("CODE COMPILED WITHOUT ERRORS")
    print(f'Check files with names ending in "{tag}"')
    print("\\n")
'''

RUN_FROM_CFG = '''
def run_batch_from_config(config_module=config):
    """Orchestrate load → prepare → init → SLAM batch (no import-time side effects)."""
    cfg = DdfgoConfig.from_module(config_module)
    runtime = derive_runtime(cfg)
    histories = load_simulation_histories(config_module)
    prepared = prepare_batch_histories(histories, runtime, config_module)
    N = prepared["N"]
    num_iter = prepared["num_iter"]
    print(f"\\nNumber of agents: {N}")
    print(f"Number of iterations: {num_iter}")
    print(f"Configuration tag: {config_module.get_results_tag()}\\n")
    if config_module.enable_notify:
        notify(
            f"DDFGO++ started | Agents={N}, Iter={num_iter}, Tag={config_module.get_results_tag()}",
            title="DDFGO++ Started",
            topic=config_module.notify_topic,
            verbose=True,
        )
    slam_state = initialize_slam_batch(prepared, cfg)
    run_slam_batch(slam_state)
    return slam_state


def main():
    return run_batch_from_config()


if __name__ == "__main__":
    main()
'''

RUN_BATCH_BODY = (
    "def run_slam_batch(slam_state):\n"
    '    """Run main timestep loop and post-loop timing / final persistence."""\n'
    "    slam_state['start_time'] = time.time()\n"
    "    num_iter = slam_state['num_iter']\n"
    "    for i in range(1, num_iter):\n"
    "        run_slam_timestep(slam_state, i)\n"
    + ind(timing_only, 4)
    + "\n    save_batch_results(slam_state, slam_state['config_module'])\n"
)

out = (
    text.split("SCRIPT_BODY = r")[0].rstrip()
    + "\n\n"
    + LOAD_PREP
    + "\n"
    + HEADER_INIT
    + ind(init_suffix.strip("\n") + "\n", 4)
    + SLAM_STATE_DICT
    + "\n"
    + TIMESTEP_UNPACK
    + loop_only
    + "\n\n"
    + RUN_BATCH_BODY
    + "\n"
    + SAVE_BATCH
    + RUN_FROM_CFG
)

backup = src_path.with_suffix(".py.bak_script_body")
backup.write_text(text, encoding="utf-8")
out_path = ROOT / "SwarmDDFGO++.py"
out_path.write_text(out, encoding="utf-8")
print("Wrote", out_path, "backup", backup)
