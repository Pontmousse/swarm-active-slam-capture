import numpy as np
import os
from functools import partial
import copy
import time
from datetime import timedelta
import gtsam
import open3d as o3d
from gtsam import symbol_shorthand
import pickle
import random
from dataclasses import dataclass
import shared_config
from helper import *
from Custom_Factors import *
from LandmarkRegistry import LandmarkRegistry
from Feature_Processing import compute_features, add_noise_to_features, get_descriptor_dim
from map_merging import (
    store_scan_local,
    build_merged_map,
    compute_merged_map_error,
    merge_voxel,
)
from notify_helper import notify
import config


@dataclass
class DdfgoConfig:
    config_module: object
    object_name: str
    dt: float
    step_size: int
    sw: int
    max_land: int
    unc: float
    verbose: bool
    verbose_map: bool
    verbose_kinem: bool
    update_sliding_window: bool
    init_pose_only: bool
    odom: bool
    kinem: bool
    decentralized: bool
    qn: float
    ql: float
    fgo_param: float
    low_pass_filter_coeff: float
    use_legacy_random_sampling: bool
    use_random_feature_fill: bool
    voxel_size: float
    icp_threshold: float
    scan_downsample_voxel_size: float
    map_mode: str
    feature_noise_std: float
    feature_id_namespace_policy: str
    enable_outlier_rejection: bool
    outlier_max_distance_from_com: float
    outlier_max_stale_frames: int
    calculate_w_method: str
    use_window_anchor_prior: bool
    window_anchor_stride: int
    use_sparse_gps_prior: bool
    gps_stride: int

    @classmethod
    def from_module(cls, config_module):
        step_size = int(config_module.history_downsample_step)
        if step_size <= 0:
            step_size = 1
        return cls(
            config_module=config_module,
            object_name=config_module.object_name,
            dt=1 / config_module.DT,
            step_size=step_size,
            sw=config_module.sw,
            max_land=config_module.Max_Land,
            unc=config_module.unc,
            verbose=config_module.Verbose,
            verbose_map=config_module.Verbose_map,
            verbose_kinem=config_module.Verbose_kinem,
            update_sliding_window=config_module.Update_Sliding_Window,
            init_pose_only=config_module.Init_Pose_Only,
            odom=config_module.Odom,
            kinem=config_module.Kinem,
            decentralized=config_module.Decentralized,
            qn=config_module.Qn,
            ql=config_module.Ql,
            fgo_param=config_module.FGO_Param,
            low_pass_filter_coeff=config_module.low_pass_filter_coeff,
            use_legacy_random_sampling=config_module.USE_LEGACY_RANDOM_SAMPLING,
            use_random_feature_fill=config_module.USE_RANDOM_FEATURE_FILL,
            voxel_size=config_module.voxel_size,
            icp_threshold=config_module.icp_threshold,
            scan_downsample_voxel_size=config_module.scan_downsample_voxel_size,
            map_mode=config_module.map_mode,
            feature_noise_std=config_module.feature_noise_std,
            feature_id_namespace_policy=config_module.feature_id_namespace_policy,
            enable_outlier_rejection=config_module.enable_outlier_rejection,
            outlier_max_distance_from_com=config_module.outlier_max_distance_from_com,
            outlier_max_stale_frames=config_module.outlier_max_stale_frames,
            calculate_w_method=config_module.calculate_w_method,
            use_window_anchor_prior=config_module.use_window_anchor_prior,
            window_anchor_stride=max(1, int(config_module.window_anchor_stride)),
            use_sparse_gps_prior=config_module.use_sparse_gps_prior,
            gps_stride=max(1, int(config_module.gps_stride)),
        )


@dataclass
class RuntimeDerived:
    object_name: str
    dt: float
    step_size: int
    sw: int
    max_land: int
    unc: float
    verbose: bool
    verbose_map: bool
    verbose_kinem: bool
    update_sliding_window: bool
    init_pose_only: bool
    odom: bool
    kinem: bool
    decentralized: bool
    qn: float
    ql: float
    fgo_param: float
    low_pass_filter_coeff: float
    use_legacy_random_sampling: bool
    use_random_feature_fill: bool
    voxel_size: float
    icp_threshold: float
    scan_downsample_voxel_size: float
    map_mode: str
    feature_noise_std: float
    feature_id_namespace_policy: str
    enable_outlier_rejection: bool
    outlier_max_distance_from_com: float
    outlier_max_stale_frames: int
    calculate_w_method: str
    use_window_anchor_prior: bool
    window_anchor_stride: int
    use_sparse_gps_prior: bool
    gps_stride: int


def derive_runtime(cfg: DdfgoConfig) -> RuntimeDerived:
    return RuntimeDerived(
        object_name=cfg.object_name,
        dt=cfg.dt,
        step_size=cfg.step_size,
        sw=cfg.sw,
        max_land=cfg.max_land,
        unc=cfg.unc,
        verbose=cfg.verbose,
        verbose_map=cfg.verbose_map,
        verbose_kinem=cfg.verbose_kinem,
        update_sliding_window=cfg.update_sliding_window,
        init_pose_only=cfg.init_pose_only,
        odom=cfg.odom,
        kinem=cfg.kinem,
        decentralized=cfg.decentralized,
        qn=cfg.qn,
        ql=cfg.ql,
        fgo_param=cfg.fgo_param,
        low_pass_filter_coeff=cfg.low_pass_filter_coeff,
        use_legacy_random_sampling=cfg.use_legacy_random_sampling,
        use_random_feature_fill=cfg.use_random_feature_fill,
        voxel_size=cfg.voxel_size,
        icp_threshold=cfg.icp_threshold,
        scan_downsample_voxel_size=cfg.scan_downsample_voxel_size,
        map_mode=cfg.map_mode,
        feature_noise_std=cfg.feature_noise_std,
        feature_id_namespace_policy=cfg.feature_id_namespace_policy,
        enable_outlier_rejection=cfg.enable_outlier_rejection,
        outlier_max_distance_from_com=cfg.outlier_max_distance_from_com,
        outlier_max_stale_frames=cfg.outlier_max_stale_frames,
        calculate_w_method=cfg.calculate_w_method,
        use_window_anchor_prior=cfg.use_window_anchor_prior,
        window_anchor_stride=cfg.window_anchor_stride,
        use_sparse_gps_prior=cfg.use_sparse_gps_prior,
        gps_stride=cfg.gps_stride,
    )

###########################################################################################
# Config-driven runtime controls
###########################################################################################
object_name = config.object_name
dt = 1 / config.DT
step_size = int(config.history_downsample_step)
if step_size <= 0:
    step_size = 1

sw = config.sw
Max_Land = config.Max_Land
unc = config.unc
Verbose = config.Verbose
Verbose_map = config.Verbose_map
Verbose_kinem = config.Verbose_kinem
Update_Sliding_Window = config.Update_Sliding_Window
Init_Pose_Only = config.Init_Pose_Only
Odom = config.Odom
Kinem = config.Kinem
Decentralized = config.Decentralized
Qn = config.Qn
Ql = config.Ql
FGO_Param = config.FGO_Param
low_pass_filter_coeff = config.low_pass_filter_coeff

# Phase A scaffolding hooks
USE_LEGACY_RANDOM_SAMPLING = config.USE_LEGACY_RANDOM_SAMPLING
USE_RANDOM_FEATURE_FILL = config.USE_RANDOM_FEATURE_FILL
voxel_size = config.voxel_size
icp_threshold = config.icp_threshold
scan_downsample_voxel_size = config.scan_downsample_voxel_size
map_mode = config.map_mode
feature_noise_std = config.feature_noise_std
feature_id_namespace_policy = config.feature_id_namespace_policy
enable_outlier_rejection = config.enable_outlier_rejection
outlier_max_distance_from_com = config.outlier_max_distance_from_com
outlier_max_stale_frames = config.outlier_max_stale_frames

calculate_w_methods = {
    "w1": calculate_w1,
    "w2": calculate_w2,
    "w3": calculate_w3,
    "w4": calculate_w4,
}
calculate_w = calculate_w_methods.get(config.calculate_w_method, calculate_w4)
use_window_anchor_prior = config.use_window_anchor_prior
window_anchor_stride = max(1, int(config.window_anchor_stride))
use_sparse_gps_prior = config.use_sparse_gps_prior
gps_stride = max(1, int(config.gps_stride))


#############################################################################################
# I/O helpers
#############################################################################################
def load_pickle_file(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle_files(agents_history, target_history, config_module=config):
    results_paths = config_module.get_results_paths()
    os.makedirs(os.path.dirname(results_paths["agents"]), exist_ok=True)
    with open(results_paths["agents"], "wb") as file:
        pickle.dump(agents_history, file)
    with open(results_paths["target"], "wb") as file:
        pickle.dump(target_history, file)


def load_simulation_histories(config_module=config):
    data_paths = config_module.get_data_paths()
    return (
        load_pickle_file(data_paths["agents"]),
        load_pickle_file(data_paths["target"]),
        load_pickle_file(data_paths["target_pcd"]),
    )


def prepare_batch_histories(histories, runtime: RuntimeDerived):
    agents_history, target_history, target_point_cloud_history = histories
    num_iter = len(agents_history)
    step_size = runtime.step_size
    agents_history = [agents_history[i] for i in range(0, num_iter, step_size)]
    target_history = [target_history[i] for i in range(0, num_iter, step_size)]
    target_point_cloud_history = [
        target_point_cloud_history[i] for i in range(0, num_iter, step_size)
    ]

    n_agents = len(agents_history[0])
    num_iter = len(agents_history)
    feature_id_namespace_policy = runtime.feature_id_namespace_policy
    if feature_id_namespace_policy == "auto":
        feature_id_namespace_policy = "registry" if n_agents == 1 else "source_feature_idx"
    if feature_id_namespace_policy not in ("registry", "source_feature_idx"):
        raise ValueError(
            f"Unsupported feature_id_namespace_policy: {feature_id_namespace_policy}"
        )

    return {
        "Agents_History": agents_history,
        "Target_History": target_history,
        "Target_Point_Cloud_History": target_point_cloud_history,
        "N": n_agents,
        "num_iter": num_iter,
        "feature_id_namespace_policy": feature_id_namespace_policy,
    }


def normalize_simulation_frames(simulation_frames):
    if simulation_frames is None:
        raise ValueError("simulation_frames must not be None")
    if isinstance(simulation_frames, dict):
        frames = [simulation_frames]
    else:
        frames = list(simulation_frames)
    if not frames:
        raise ValueError("simulation_frames must contain at least one frame")

    required = {
        "iteration",
        "sim_time",
        "dt",
        "agents_true_state",
        "target_true_state",
        "agent_observations",
        "communication_sets",
        "target_point_cloud",
    }
    for frame in frames:
        missing = required.difference(frame)
        if missing:
            raise KeyError(f"simulation_frame missing required keys: {sorted(missing)}")

    return sorted(frames, key=lambda frame: frame["iteration"])


def simulation_frame_to_history_entries(frame, runtime: RuntimeDerived):
    agents = []
    observations = frame["agent_observations"]
    communication_sets = frame["communication_sets"]
    for a, state in enumerate(frame["agents_true_state"]):
        land_set = observations[a] if a < len(observations) else []
        comm_set = communication_sets[a] if a < len(communication_sets) else []
        agents.append(
            {
                "ID": a,
                "State": state,
                "LandSet": land_set,
                "CommSet": comm_set,
                "DockSet": [],
                "CollSet": [],
                "AntFlkSet": [],
                "LC": [],
                "LCD": [],
                "LCD_Frame": [],
                "Odometry": [],
                "Target": [],
            }
        )
    return agents, frame["target_true_state"], frame["target_point_cloud"]


def _initialize_agent_slam_fields(agent, identity_pose, feature_id_namespace_policy):
    agent["State_Estim"] = identity_pose
    agent["State_Obs"] = identity_pose
    agent["Odom_Obs"] = identity_pose
    agent["Target_Measure"] = (None, None, None, None, None, None, None, None, None)
    agent["Target_True"] = (None, None, None, None, None, None, None, None, None)
    agent["Target_COM"] = (None, None, None)
    agent["Target_V"] = (None, None, None)
    agent["Target_W"] = (None, None, None)
    agent["Target_Estim"] = np.zeros(9)
    agent["FGO_Target_COM"] = (None, None, None)
    agent["FGO_Target_V"] = (None, None, None)
    agent["FGO_Target_W"] = (None, None, None)
    agent["Total_FGO_Error"] = None
    agent["CPU_time"] = None
    agent["MapSet"] = []
    agent["MapIdxSet"] = []
    agent["MapNghSet"] = []
    agent["MapGrowth"] = 0
    agent["MapEntropy"] = []
    agent["MergedMapSet"] = np.array([]).reshape(0, 3)
    agent["MergedMap_Error"] = {
        "chamfer_distance": 0.0,
        "rmse_est_to_gt": 0.0,
        "inlier_ratio": 0.0,
    }
    agent["KinLoopClosuresAdded"] = 0
    agent["KinLoopClosuresUnique"] = 0
    agent["ReobsCount"] = 0
    agent["LandmarkRegistry"] = LandmarkRegistry()
    agent["FeatureDescSet"] = np.array([]).reshape(0, get_descriptor_dim())
    agent["FeatureGraphIdSet"] = np.array([], dtype=int)
    agent["FeatureIdNamespace"] = feature_id_namespace_policy
    agent["OutliersRemovedDistance"] = 0
    agent["OutliersRemovedStale"] = 0


def initialize_slam_runtime(first_frame_or_histories, config_module=config, mode="batch"):
    cfg = DdfgoConfig.from_module(config_module)
    runtime = derive_runtime(cfg)
    if mode == "batch":
        prepared_histories = prepare_batch_histories(first_frame_or_histories, runtime)
    elif mode == "online":
        frame = normalize_simulation_frames(first_frame_or_histories)[0]
        histories = tuple([entry] for entry in simulation_frame_to_history_entries(frame, runtime))
        prepared_histories = prepare_batch_histories(histories, runtime)
    else:
        raise ValueError(f"Unsupported SLAM runtime mode: {mode}")

    slam_state = initialize_slam_batch(prepared_histories, cfg, runtime)
    slam_state["mode"] = mode
    slam_state["cfg"] = cfg
    slam_state["config_module"] = config_module
    slam_state["runtime_obj"] = runtime
    if mode == "online":
        # Online SLAM advances i by SLAM updates; cadence comes from shared_config so it
        # stays aligned with stride (sim scheduler), independent of batch kappa_seconds/dt.
        _shared_cfg = getattr(config_module, "shared_config", shared_config)
        _slam_period_s = float(getattr(_shared_cfg, "stride", runtime.dt))
        if _slam_period_s <= 0.0:
            _slam_period_s = runtime.dt
        _comm_period_s = float(
            getattr(_shared_cfg, "online_decentralized_comm_period_seconds", 1.0)
        )
        if _comm_period_s <= 0.0:
            _comm_period_s = _slam_period_s
        slam_state["every"] = max(1, int(np.floor(_comm_period_s / _slam_period_s)))
    return slam_state


def initialize_slam_online(initial_simulation_frame, config_module=config):
    frames = normalize_simulation_frames(initial_simulation_frame)
    slam_state = initialize_slam_runtime(frames[0], config_module=config_module, mode="online")
    slam_state["last_processed_iteration"] = frames[0]["iteration"]
    slam_state["last_slam_time"] = frames[0]["sim_time"]
    slam_state["frame_buffer_history"] = []
    slam_state["latest_slam_feedback"] = build_slam_feedback(slam_state)
    return slam_state


def _append_online_frame(slam_state, frame):
    runtime = slam_state["runtime_obj"]
    agents, target, target_point_cloud = simulation_frame_to_history_entries(frame, runtime)
    for agent in agents:
        _initialize_agent_slam_fields(
            agent,
            slam_state["identity_pose"],
            slam_state["feature_id_namespace_policy"],
        )

    slam_state["Agents_History"].append(agents)
    slam_state["Target_History"].append(target)
    slam_state["Target_Point_Cloud_History"].append(target_point_cloud)
    slam_state["num_iter"] = len(slam_state["Agents_History"])
    i_new = slam_state["num_iter"] - 1
    # Carry forward LandmarkRegistry across online SLAM steps. _initialize_agent_slam_fields always
    # installs an empty LandmarkRegistry(); without this, active_ids cannot reuse prior tracks.
    if i_new >= 1:
        hist = slam_state["Agents_History"]
        prev_agents = hist[i_new - 1]
        curr_agents = hist[i_new]
        for ai in range(min(len(curr_agents), len(prev_agents))):
            prev_reg = prev_agents[ai].get("LandmarkRegistry") if isinstance(prev_agents[ai], dict) else None
            if prev_reg is not None and isinstance(curr_agents[ai], dict):
                curr_agents[ai]["LandmarkRegistry"] = copy.deepcopy(prev_reg)
    return i_new


def prepare_slam_frames(simulation_frames):
    return normalize_simulation_frames(simulation_frames)


def select_slam_update_frame(slam_state, frames):
    latest_frame = frames[-1]
    last_processed = slam_state.get("last_processed_iteration")
    if last_processed is not None and latest_frame["iteration"] <= last_processed:
        return None
    return latest_frame


def append_online_slam_frame(slam_state, frame):
    return _append_online_frame(slam_state, frame)


def run_slam_update_phase(slam_state, slam_index):
    run_slam_timestep(slam_state, slam_index)


def finalize_slam_update(slam_state, latest_frame):
    slam_state["last_processed_iteration"] = latest_frame["iteration"]
    slam_state["last_slam_time"] = latest_frame["sim_time"]


def process_slam_update(slam_state, frame_buffer):
    frames = prepare_slam_frames(frame_buffer)
    latest_frame = select_slam_update_frame(slam_state, frames)
    if latest_frame is None:
        return
    slam_state["slam_update_counter"] = slam_state.get("slam_update_counter", 0) + 1
    total_sim = latest_frame.get("total_sim_iterations")
    if total_sim is not None:
        slam_state["expected_sim_iterations"] = total_sim
    slam_state["current_global_iteration"] = latest_frame["iteration"]
    slam_state["current_global_sim_time"] = latest_frame["sim_time"]
    elapsed = latest_frame["sim_time"] - slam_state.get("last_slam_time", frames[0]["sim_time"])
    slam_state["last_slam_elapsed"] = elapsed
    slam_state.setdefault("frame_buffer_history", []).append(frames)

    i = append_online_slam_frame(slam_state, latest_frame)
    run_slam_update_phase(slam_state, i)
    finalize_slam_update(slam_state, latest_frame)


def build_slam_feedback(slam_state):
    agents = slam_state["Agents_History"][-1]
    map_summaries = []
    for agent in agents:
        map_summaries.append(
            {
                "map_size": len(agent.get("MapSet", [])),
                "merged_map_size": len(agent.get("MergedMapSet", [])),
                "map_growth": agent.get("MapGrowth"),
                "merged_map_error": agent.get("MergedMap_Error"),
            }
        )

    first_agent = agents[0] if agents else {}
    feedback = {
        "iteration": slam_state.get("last_processed_iteration", 0),
        "slam_time": slam_state.get("last_slam_time", 0.0),
        "agent_pose_estimates": [agent.get("State_Estim") for agent in agents],
        "target_com_estimate": first_agent.get("Target_COM"),
        "target_velocity_estimate": first_agent.get("Target_V"),
        "target_angular_velocity_estimate": first_agent.get("Target_W"),
        "map_summaries": map_summaries,
        "map_quality": {
            "total_fgo_error": [agent.get("Total_FGO_Error") for agent in agents],
            "visible_fgo_error": [agent.get("Visible_FGO_Error") for agent in agents],
        },
        "diagnostics": {
            "last_slam_elapsed": slam_state.get("last_slam_elapsed", 0.0),
            "num_slam_steps": len(slam_state["Agents_History"]),
            "mode": slam_state.get("mode"),
        },
    }
    slam_state["latest_slam_feedback"] = feedback
    return feedback


def step_slam(slam_state, simulation_frames):
    process_slam_update(slam_state, simulation_frames)
    return build_slam_feedback(slam_state)


def _as_points_array(points):
    if points is None:
        return np.array([]).reshape(0, 3)
    points = np.asarray(points)
    if points.size == 0:
        return np.array([]).reshape(0, 3)
    return points.reshape(-1, 3)


def _build_shared_dense_map(agents_history, agent_idx, frame_idx, voxel_size):
    agent = agents_history[frame_idx][agent_idx]
    point_sets = []
    source_agents = []

    local_pts = _as_points_array(agent.get("MergedMapSet", np.array([]).reshape(0, 3)))
    if len(local_pts) > 0:
        point_sets.append(local_pts)
        source_agents.append(int(agent_idx))

    for neighbor_idx in agent.get("CommSet", []):
        neighbor_idx = int(neighbor_idx)
        if neighbor_idx < 0 or neighbor_idx >= len(agents_history[frame_idx]):
            continue
        neighbor_pts = _as_points_array(
            agents_history[frame_idx][neighbor_idx].get("MergedMapSet", np.array([]).reshape(0, 3))
        )
        if len(neighbor_pts) == 0 and frame_idx > 0:
            neighbor_pts = _as_points_array(
                agents_history[frame_idx - 1][neighbor_idx].get("MergedMapSet", np.array([]).reshape(0, 3))
            )
        if len(neighbor_pts) > 0:
            point_sets.append(neighbor_pts)
            source_agents.append(neighbor_idx)

    if point_sets:
        merged = np.vstack(point_sets)
        if voxel_size is not None and voxel_size > 0:
            merged = merge_voxel(merged, voxel_size)
    else:
        merged = np.array([]).reshape(0, 3)

    return {
        "MergedMapSharedSet": merged,
        "MergedMapSharedSourceSet": np.array(source_agents, dtype=int),
    }



_BATCH_STATE_NAMES = ['Agents_History', 'D', 'F', 'FeatureDescAll', 'FeatureDesc_sel', 'FeatureId_sel', 'FeatureSet_sel', 'Fidx', 'Graph_List', 'KeyFrames', 'KeyFramesIdx', 'KeyFramesIdx_Hist', 'KeyFramesNgh', 'KeyFramesNgh_Hist', 'KeyFrames_Hist', 'L', 'LandSet', 'LandSet_noisy', 'Lmap_com', 'Lobs_l', 'Lpar_l', 'Lpar_vel', 'Lvar_com', 'Lvar_l', 'Lvar_vel', 'Map', 'MapIdx', 'MapIdx_filtered', 'MapNgh', 'MapNgh_filtered', 'Map_filtered', 'N', 'Ngh', 'P', 'Points', 'Points_prev', 'Qll', 'Qnn', 'R_w', 'Target_History', 'Target_Point_Cloud_History', 'V', 'Value_List', 'W', 'X', 'a', 'alpha', 'ang_obs', 'ang_vel', 'ang_vel_factor', 'avg_time_per_iteration', 'bearing', 'bearing_obs', 'bearing_range_noise', 'bearing_range_noise_base', 'bearing_sigma', 'bias', 'calculate_w', 'com', 'com1_obs', 'com2_obs', 'com_factor', 'com_obs', 'comf_error', 'comf_jacobian', 'cond', 'config', 'decay', 'descriptor', 'elapsed_time', 'estimated_time_left', 'estimated_time_left_str', 'every', 'exist', 'f', 'feature', 'feature_id_namespace_policy', 'feature_obs', 'feature_result', 'filtered_pose', 'frame', 'frame_idx', 'frame_ngh', 'frames_stale', 'gps_prior_noise_weak', 'graph', 'graph_feature_ids', 'i', 'identity_pose', 'idx', 'initial_estimate', 'initial_pose_estim', 'inserted_landmarks', 'isam', 'iterations_left', 'j', 'k', 'kappa', 'key', 'keyframe', 'kin_added', 'kin_ids', 'kinem_factor', 'kinem_noise', 'kinem_noise_base', 'kinem_xyz_sigma', 'l', 'landmark_prior_noise', 'lf_error', 'lf_jacobian', 'list_symbs', 'lm_id', 'lm_id_int', 'lm_ngh', 'lm_pos', 'm', 'map_avg', 'merged_pcd', 'merged_pts', 'new_pose', 'noisy_gps_pose', 'noisy_odom', 'num_iter', 'num_variables', 'obs', 'odom_tf', 'odometry_noise', 'odometry_rpy_sigma', 'odometry_xyz_sigma', 'outliers_removed_distance', 'outliers_removed_stale', 'parameters', 'points', 'points_prev', 'pos', 'pos1', 'pos2', 'pos_arr', 'pose_estim', 'prev_pose', 'prior_noise', 'prior_rpy_sigma', 'prior_xyz_sigma', 'pvw_noise', 'pvw_xyz_sigma', 'ql', 'qn', 'qn_namespace', 'r', 'range_', 'range_obs', 'range_sigma', 'real_odom', 'registry', 'reobs', 'result', 'runtime', 'save_every', 'selectionA', 'selectionL', 'selection_idx', 'source_feature_idx_all', 'source_feature_idx_sel', 'start_time', 'start_time_iteration_agent', 'state_obs', 't_w', 'tag', 'target_noise_bias', 'target_noise_std', 'target_pcd', 'vel', 'vel_factor', 'vel_obs', 'velf_error', 'velf_jacobian', 'window_anchor_noise']


def _runtime_bindings(cfg: DdfgoConfig, runtime: RuntimeDerived):
    runtime_dict = runtime.__dict__
    return {
        "config": cfg.config_module,
        "runtime": runtime_dict,
        "object_name": runtime.object_name,
        "dt": runtime.dt,
        "step_size": runtime.step_size,
        "sw": runtime.sw,
        "Max_Land": runtime.max_land,
        "unc": runtime.unc,
        "Verbose": runtime.verbose,
        "Verbose_map": runtime.verbose_map,
        "Verbose_kinem": runtime.verbose_kinem,
        "Update_Sliding_Window": runtime.update_sliding_window,
        "Init_Pose_Only": runtime.init_pose_only,
        "Odom": runtime.odom,
        "Kinem": runtime.kinem,
        "Decentralized": runtime.decentralized,
        "Qn": runtime.qn,
        "Ql": runtime.ql,
        "FGO_Param": runtime.fgo_param,
        "low_pass_filter_coeff": runtime.low_pass_filter_coeff,
        "USE_LEGACY_RANDOM_SAMPLING": runtime.use_legacy_random_sampling,
        "USE_RANDOM_FEATURE_FILL": runtime.use_random_feature_fill,
        "voxel_size": runtime.voxel_size,
        "icp_threshold": runtime.icp_threshold,
        "scan_downsample_voxel_size": runtime.scan_downsample_voxel_size,
        "map_mode": runtime.map_mode,
        "feature_noise_std": runtime.feature_noise_std,
        "feature_id_namespace_policy": runtime.feature_id_namespace_policy,
        "enable_outlier_rejection": runtime.enable_outlier_rejection,
        "outlier_max_distance_from_com": runtime.outlier_max_distance_from_com,
        "outlier_max_stale_frames": runtime.outlier_max_stale_frames,
        "use_window_anchor_prior": runtime.use_window_anchor_prior,
        "window_anchor_stride": runtime.window_anchor_stride,
        "use_sparse_gps_prior": runtime.use_sparse_gps_prior,
        "gps_stride": runtime.gps_stride,
        "calculate_w": calculate_w_methods.get(runtime.calculate_w_method, calculate_w4),
    }


def _restore_slam_globals(slam_state):
    globals().update(slam_state)


def _capture_slam_globals(slam_state):
    for name in _BATCH_STATE_NAMES:
        if name in globals():
            slam_state[name] = globals()[name]


def _initialize_slam_batch_body(slam_state):
    global D, Graph_List, KeyFramesIdx_Hist, KeyFramesNgh_Hist, KeyFrames_Hist, L, Lobs_l, Lpar_l, Lpar_vel, Lvar_l, Lvar_vel, P
    global V, Value_List, W, X, a, alpha, bearing_range_noise, bearing_range_noise_base, bearing_sigma, every, gps_prior_noise_weak, graph
    global i, identity_pose, initial_estimate, initial_pose_estim, kappa, kinem_noise, kinem_noise_base, kinem_xyz_sigma, landmark_prior_noise, lf_error, lf_jacobian, noisy_gps_pose
    global odometry_noise, odometry_rpy_sigma, odometry_xyz_sigma, parameters, prior_noise, prior_rpy_sigma, prior_xyz_sigma, pvw_noise, pvw_xyz_sigma, range_sigma, start_time, tag
    global target_noise_bias, target_noise_std, velf_error, velf_jacobian, window_anchor_noise
    _restore_slam_globals(slam_state)
    ###########################################################################################
    ###########################################################################################
    ###########################################################################################

    ##############################################################################################################################
    # Factor Graph Parameters
    ##############################################################################################################################
    tag = config.get_results_tag()

    ########## General Rule ##########
    # kappa * dt * Max_Land ~= Qn * Ql
    ########## General Rule ##########

    # Register landmarks every kappa seconds
    kappa = config.kappa_seconds
    every = max(1, int(np.floor(kappa / dt)))

    # %%
    ##############################################################################################################################
    # Sensor Parameters
    ##############################################################################################################################
    # gtsam parameters
    parameters = gtsam.ISAM2Params()
    parameters.setRelinearizeThreshold(config.relinearize_threshold)
    parameters.relinearizeSkip = config.relinearize_skip
    identity_pose = gtsam.Pose3(gtsam.Rot3.Quaternion(1, 0, 0, 0), gtsam.Point3(0,0,0))

    # prior factor noise
    prior_rpy_sigma = unc * config.prior_rpy_sigma_base # degrees
    prior_xyz_sigma = unc * config.prior_xyz_sigma_base # meters
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_rpy_sigma*np.pi/180,
                                                            prior_rpy_sigma*np.pi/180,
                                                            prior_rpy_sigma*np.pi/180,
                                                            prior_xyz_sigma,
                                                            prior_xyz_sigma,
                                                            prior_xyz_sigma]))

    window_anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        config.window_anchor_rpy_sigma * np.pi / 180,
        config.window_anchor_rpy_sigma * np.pi / 180,
        config.window_anchor_rpy_sigma * np.pi / 180,
        config.window_anchor_xyz_sigma,
        config.window_anchor_xyz_sigma,
        config.window_anchor_xyz_sigma,
    ]))
    gps_prior_noise_weak = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        config.gps_rpy_sigma_weak * np.pi / 180,
        config.gps_rpy_sigma_weak * np.pi / 180,
        config.gps_rpy_sigma_weak * np.pi / 180,
        config.gps_xyz_sigma_weak,
        config.gps_xyz_sigma_weak,
        config.gps_xyz_sigma_weak,
    ]))

    # odometry factor noise
    odometry_rpy_sigma = unc * config.odometry_rpy_sigma_base # degrees
    odometry_xyz_sigma = unc * config.odometry_xyz_sigma_base # meters
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([odometry_rpy_sigma*np.pi/180,
                                                                odometry_rpy_sigma*np.pi/180,
                                                                odometry_rpy_sigma*np.pi/180,
                                                                odometry_xyz_sigma,
                                                                odometry_xyz_sigma,
                                                                odometry_xyz_sigma]))

    # landmark factor noise
    bearing_sigma = unc * config.bearing_sigma_base # degrees
    range_sigma = unc * config.range_sigma_base # meters
    bearing_range_noise_base = gtsam.noiseModel.Diagonal.Sigmas(np.array([bearing_sigma*np.pi/180,
                                                                           bearing_sigma*np.pi/180,
                                                                           range_sigma]))
    bearing_range_noise = bearing_range_noise_base
    if config.use_robust_bearing:
        bearing_range_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(config.huber_param),
            bearing_range_noise_base,
        )


    # Target kinematic parameters estimation gains and noise models
    target_noise_std = unc * config.target_noise_std
    target_noise_bias = unc * config.target_noise_bias

    # landmark kinematic factor noise
    kinem_xyz_sigma = unc * config.kinematic_sigma_base
    kinem_noise_base = gtsam.noiseModel.Diagonal.Sigmas(np.array([kinem_xyz_sigma,
                                                                   kinem_xyz_sigma,
                                                                   kinem_xyz_sigma]))
    kinem_noise = kinem_noise_base
    if config.use_robust_kinematic:
        kinem_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Cauchy.Create(config.cauchy_param),
            kinem_noise_base,
        )

    # weak landmark prior hook (wired in Phase A; applied in later phases)
    landmark_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, config.landmark_prior_sigma)


    # target center of mass, velocity and angular velocity factor noise
    pvw_xyz_sigma = unc * config.pvw_sigma_base
    pvw_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([pvw_xyz_sigma,
                                                           pvw_xyz_sigma,
                                                           pvw_xyz_sigma]))

    # %%
    ##############################################################################################################################
    # Main
    ##############################################################################################################################
    X = symbol_shorthand.X
    L = symbol_shorthand.L
    P = symbol_shorthand.P
    V = symbol_shorthand.V
    W = symbol_shorthand.W

    # define kinematic factor symbolic equations and jacobians
    lf_error, Lvar_l, Lobs_l, Lpar_l = landmark_factor_error()
    lf_jacobian = jacobian(lf_error, Lvar_l)

    # define target velocity factor symbolic equations and jacobians        
    velf_error, Lvar_vel, Lpar_vel = target_velocity_factor_error()
    velf_jacobian = jacobian(velf_error, Lvar_vel)


    # Set new keys in dicitonary structure
    for i in range(num_iter):
        for a in range(N):
            Agents_History[i][a]['State_Estim'] = identity_pose
            Agents_History[i][a]['State_Obs'] = identity_pose
            Agents_History[i][a]['Odom_Obs'] = identity_pose
            Agents_History[i][a]['Target_Measure'] = (None, None, None,None, None, None,None, None, None)
            Agents_History[i][a]['Target_True'] = (None, None, None,None, None, None,None, None, None)
        
            Agents_History[i][a]['Target_COM'] = (None, None, None)
            Agents_History[i][a]['Target_V'] = (None, None, None)
            Agents_History[i][a]['Target_W'] = (None, None, None)
            Agents_History[i][a]['Target_Estim'] = np.zeros(9)
        
            Agents_History[i][a]['FGO_Target_COM'] = (None, None, None)
            Agents_History[i][a]['FGO_Target_V'] = (None, None, None)
            Agents_History[i][a]['FGO_Target_W'] = (None, None, None)
        
            Agents_History[i][a]['Total_FGO_Error'] = None
            Agents_History[i][a]['CPU_time'] = None

            Agents_History[i][a]['MapSet'] = []
            Agents_History[i][a]['MapIdxSet'] = []
            Agents_History[i][a]['MapNghSet'] = [] # Indices of agent that shared landmark
            Agents_History[i][a]['MapGrowth'] = 0
            Agents_History[i][a]['MapEntropy'] = [] # Map landmarks entropy calculated using Gaussian assumption
            Agents_History[i][a]['MergedMapSet'] = np.array([]).reshape(0, 3)
            Agents_History[i][a]['MergedMap_Error'] = {
                'chamfer_distance': 0.0,
                'rmse_est_to_gt': 0.0,
                'inlier_ratio': 0.0
            }
            Agents_History[i][a]['KinLoopClosuresAdded'] = 0
            Agents_History[i][a]['KinLoopClosuresUnique'] = 0
            Agents_History[i][a]['ReobsCount'] = 0
            Agents_History[i][a]['LandmarkRegistry'] = LandmarkRegistry()
            Agents_History[i][a]['FeatureDescSet'] = np.array([]).reshape(0, get_descriptor_dim())
            Agents_History[i][a]['FeatureGraphIdSet'] = np.array([], dtype=int)
            Agents_History[i][a]['FeatureIdNamespace'] = feature_id_namespace_policy
            Agents_History[i][a]['OutliersRemovedDistance'] = 0
            Agents_History[i][a]['OutliersRemovedStale'] = 0


    Graph_List = []
    Value_List = []

    for a in range(N):
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
    
    
        initial_pose_estim = add_noise_to_pose(true_pose(Agents_History, a,0), prior_xyz_sigma, prior_rpy_sigma)
        noisy_gps_pose = add_noise_to_pose(true_pose(Agents_History, a,0), prior_xyz_sigma, prior_rpy_sigma)
    
        graph.add(gtsam.PriorFactorPose3(varX(X,a,0), noisy_gps_pose, prior_noise))
        initial_estimate.insert(varX(X,a,0), initial_pose_estim)

        Graph_List.append(graph)
        Value_List.append(initial_estimate)
    
        # History initialization
        Agents_History[0][a]['State_Estim'] = initial_pose_estim
        Agents_History[0][a]['State_Obs'] = noisy_gps_pose
        Agents_History[0][a]['Odom_Obs'] = identity_pose
        Agents_History[0][a]['Target_Measure'] = measure_target_params(Target_History[0], target_noise_std, target_noise_bias)
        Agents_History[0][a]['Target_True'] = measure_target_params(Target_History[0], 0*target_noise_std, 0*target_noise_bias)
    
        Agents_History[0][a]['Target_COM'] = np.array([0,0,0])
        Agents_History[0][a]['Target_V'] = np.array([0,0,0])
        Agents_History[0][a]['Target_W'] = np.array([0,0,0])
        Agents_History[0][a]['Target_Estim'] = np.array([0,0,0,0,0,0,0,0,0], dtype=np.float64)
    
        Agents_History[0][a]['FGO_Target_COM'] = np.array([0,0,0])
        Agents_History[0][a]['FGO_Target_V'] = np.array([0,0,0])
        Agents_History[0][a]['FGO_Target_W'] = np.array([0,0,0])
    
        Agents_History[0][a]['Total_FGO_Error'] = total_fgo_error(Agents_History[0][a], Target_Point_Cloud_History[0])
        Agents_History[0][a]['CPU_time'] = 0

    
    # time to reach convergence on target parameters estimates.
    D = dt*step_size*num_iter # Period or Duration
    alpha = 100*D/num_iter # higher means higher rate of convergence.

    # Record the start time for the entire loop
    start_time = time.time()

    # Frame registration for front-end SLAM and loop closure detection
    KeyFrames_Hist = [[]]
    KeyFramesIdx_Hist = [[]]
    KeyFramesNgh_Hist = [[]]

    _capture_slam_globals(slam_state)


def _run_slam_timestep_body(slam_state):
    global F, FeatureDescAll, FeatureDesc_sel, FeatureId_sel, FeatureSet_sel, Fidx, KeyFrames, KeyFramesIdx, KeyFramesNgh, LandSet, LandSet_noisy, Lmap_com
    global Lvar_com, Map, MapIdx, MapIdx_filtered, MapNgh, MapNgh_filtered, Map_filtered, Ngh, Points, Points_prev, Qll, Qnn
    global R_w, a, ang_obs, ang_vel, ang_vel_factor, avg_time_per_iteration, bearing, bearing_obs, bias, com, com1_obs, com2_obs
    global com_factor, com_obs, comf_error, comf_jacobian, cond, decay, descriptor, elapsed_time, estimated_time_left, estimated_time_left_str, exist, f
    global feature, feature_obs, feature_result, filtered_pose, frame, frame_idx, frame_ngh, frames_stale, graph_feature_ids, idx, inserted_landmarks, isam
    global iterations_left, j, k, key, keyframe, kin_added, kin_ids, kinem_factor, l, list_symbs, lm_id, lm_id_int
    global lm_ngh, lm_pos, m, map_avg, merged_pcd, merged_pts, new_pose, noisy_gps_pose, noisy_odom, num_variables, obs, odom_tf
    global outliers_removed_distance, outliers_removed_stale, points, points_prev, pos, pos1, pos2, pos_arr, pose_estim, prev_pose, ql, qn
    global qn_namespace, r, range_, range_obs, real_odom, registry, reobs, result, save_every, selectionA, selectionL, selection_idx
    global source_feature_idx_all, source_feature_idx_sel, start_time_iteration_agent, state_obs, t_w, target_pcd, vel, vel_factor, vel_obs
    _restore_slam_globals(slam_state)
    # Convergence to real kinematic parameters
    decay = np.exp(-alpha * i)
    bias = 15*target_noise_bias * decay
    # bias = 0*bias # line to cancel simulated convergence and input true target kinematic measures

    # register landmarks through communication with neighbours every kappa seconds
    cond = False
    if i % every == 0: 
        cond = True

    # This following creates lists [] in each iteration N times.
    # The result is a list containing N empty lists.
    frame = [[] for _ in range(N)]
    frame_idx = [[] for _ in range(N)]
    frame_ngh = [[] for _ in range(N)]

    for a in range(N):
        # CPU Time
        start_time_iteration_agent = time.time()
        Agents_History[i][a]['KinLoopClosuresAdded'] = 0
        Agents_History[i][a]['KinLoopClosuresUnique'] = 0
        Agents_History[i][a]['ReobsCount'] = 0

        ############################################################
        # Get Target estimates at given time step
        Agents_History[i][a]['Target_Measure'] = measure_target_params(Target_History[i], target_noise_std, bias)
        Agents_History[i][a]['Target_True'] = measure_target_params(Target_History[i], 0*target_noise_std, 0*target_noise_bias)
        _target_kin_truth_threshold = getattr(config, "target_kinematics_truth_unc_threshold", None)
        _use_truth_target_kinematics = (
            _target_kin_truth_threshold is not None
            and float(config.unc) <= float(_target_kin_truth_threshold)
        )

        # Feedback simulated convergence depending on decay and bias coefficient
        # com = Agents_History[i][a]['Target_Measure'][:3]
        # vel = Agents_History[i][a]['Target_Measure'][3:6]
        # ang_vel = Agents_History[i][a]['Target_Measure'][6:9]

        # Feedback true value
        # com = Agents_History[i][a]['Target_True'][:3]
        # vel = Agents_History[i][a]['Target_True'][3:6]
        # ang_vel = Agents_History[i][a]['Target_True'][6:9]

        # feedback estimates calculated outside the factor graph
        if _use_truth_target_kinematics:
            com = Agents_History[i][a]['Target_True'][:3]
            vel = Agents_History[i][a]['Target_True'][3:6]
            ang_vel = Agents_History[i][a]['Target_True'][6:9]
        else:
            com = Agents_History[i-1][a]['Target_COM']
            vel = Agents_History[i-1][a]['Target_V']
            ang_vel = Agents_History[i-1][a]['Target_W']

        # feedback estimates calculate inside the factor graph
        # com = Agents_History[i-1][a]['FGO_Target_COM']
        # vel = Agents_History[i-1][a]['FGO_Target_V']
        # ang_vel = Agents_History[i-1][a]['FGO_Target_W']

        ####################################################################
        # Front End
        ####################################################################
        if Verbose: print('\nFront-End running...')

        ####################################
        # GPS Factors
        ####################################

        # GPS Observation
        noisy_gps_pose = add_noise_to_pose(true_pose(Agents_History,a,i), prior_xyz_sigma, prior_rpy_sigma)
        Agents_History[i][a]['State_Obs'] = noisy_gps_pose  

        # Odometry Observation   
        real_odom = true_pose(Agents_History,a,i-1).transformPoseTo(true_pose(Agents_History,a,i))
        noisy_odom = add_noise_to_pose(real_odom, odometry_xyz_sigma, odometry_rpy_sigma)
        Agents_History[i][a]['Odom_Obs'] = noisy_odom

        # State Propagation
        pose_estim = Agents_History[i-1][a]['State_Estim'].compose(noisy_odom)
        Agents_History[i][a]['State_Estim'] = pose_estim

        ####################################
        # Pose Nodes
        ####################################

        # Define and add Pose variable node
        Value_List[a].insert(varX(X,a,i), pose_estim)

        # Define and add GPS Factor
        if not Init_Pose_Only:
            if use_window_anchor_prior and (i % window_anchor_stride == 0):
                Graph_List[a].add(gtsam.PriorFactorPose3(varX(X, a, i), pose_estim, window_anchor_noise))
                if Verbose: print(f'Added window-anchor prior to: X{a}_{i}')
            if use_sparse_gps_prior:
                if i % gps_stride == 0:
                    Graph_List[a].add(gtsam.PriorFactorPose3(varX(X, a, i), noisy_gps_pose, gps_prior_noise_weak))
                    if Verbose: print(f'Added sparse gps prior to: X{a}_{i}')
            else:
                Graph_List[a].add(gtsam.PriorFactorPose3(varX(X,a,i), noisy_gps_pose, prior_noise))
                if Verbose: print(f'Added gps factor to: X{a}_{i}')

        ####################################
        # Odometry Factors
        ####################################

        # Define and add Odometry Factors
        if Odom and i != 0:
            odom_tf = Agents_History[i-1][a]['Odom_Obs']
            Graph_List[a].add(gtsam.BetweenFactorPose3(varX(X,a,i-1), varX(X,a,i), odom_tf, odometry_noise))
            if Verbose: print(f'Added odometry factor between: X{a}_{i-1} -> X{a}_{i}')

        ####################################
        # Feature Preprocessing (Phase B)
        ####################################
        registry = Agents_History[i][a].get('LandmarkRegistry', Agents_History[0][a]['LandmarkRegistry'])
        Agents_History[i][a]['LandmarkRegistry'] = registry
        registry.propagate_positions(com, vel, ang_vel, step_size, num_steps=1)

        LandSet = Agents_History[i][a].get('LandSet', np.array([]))
        state_obs = Agents_History[i][a]['State_Obs']
        t_w = np.array(state_obs.translation())
        R_w = state_obs.rotation().matrix()

        if len(LandSet) > 0:
            source_feature_idx_all = np.array(
                Agents_History[i][a].get('FeatureIdxSet', np.arange(len(LandSet))),
                dtype=int,
            )
            if len(source_feature_idx_all) != len(LandSet):
                source_feature_idx_all = np.arange(len(LandSet), dtype=int)
            if feature_noise_std is None:
                LandSet_noisy = add_noise_to_features(LandSet)
            else:
                LandSet_noisy = add_noise_to_features(LandSet, feature_noise_std)
            FeatureDescAll = compute_features(LandSet_noisy)

            FeatureSet_sel, FeatureDesc_sel, FeatureId_sel, selection_idx = registry.select_features(
                frame_idx=i,
                landset_world=LandSet_noisy,
                desc_all=FeatureDescAll,
                t_w=t_w,
                R_w=R_w,
                max_land=Max_Land,
                sw=sw,
                use_legacy_random_sampling=USE_LEGACY_RANDOM_SAMPLING,
                use_random_feature_fill=USE_RANDOM_FEATURE_FILL,
                return_selection_idx=True,
            )
            source_feature_idx_sel = source_feature_idx_all[selection_idx] if len(selection_idx) > 0 else np.array([], dtype=int)

            if feature_id_namespace_policy == "registry":
                graph_feature_ids = FeatureId_sel
            else:
                graph_feature_ids = source_feature_idx_sel

            Agents_History[i][a]['FeatureSet'] = FeatureSet_sel
            Agents_History[i][a]['FeatureDescSet'] = FeatureDesc_sel
            Agents_History[i][a]['FeatureIdSet'] = FeatureId_sel
            Agents_History[i][a]['FeatureGraphIdSet'] = np.array(graph_feature_ids, dtype=int)
            Agents_History[i][a]['FeatureIdxSet'] = np.array(graph_feature_ids, dtype=int)
            Agents_History[i][a]['FeatureIdNamespace'] = feature_id_namespace_policy
            reobs = 0
            for lm_id in FeatureId_sel:
                lm_id_int = int(lm_id)
                if lm_id_int in registry.tracks and registry.tracks[lm_id_int].times_seen > 1:
                    reobs += 1
            Agents_History[i][a]['ReobsCount'] = reobs

            if map_mode in ("dense", "hybrid"):
                Agents_History[i][a].update(
                    store_scan_local(
                        LandSet_noisy,
                        true_pose(Agents_History, a, i),
                        a,
                        i,
                        scan_downsample_voxel_size,
                    )
                )
        else:
            Agents_History[i][a]['FeatureSet'] = np.array([]).reshape(0, 3)
            Agents_History[i][a]['FeatureDescSet'] = np.array([]).reshape(0, get_descriptor_dim())
            Agents_History[i][a]['FeatureIdSet'] = np.array([], dtype=int)
            Agents_History[i][a]['FeatureIdxSet'] = np.array([], dtype=int)
            Agents_History[i][a]['FeatureGraphIdSet'] = np.array([], dtype=int)
            Agents_History[i][a]['FeatureIdNamespace'] = feature_id_namespace_policy
            if map_mode in ("dense", "hybrid"):
                Agents_History[i][a]['ScanLocal'] = np.array([]).reshape(0, 3)


        ####################################
        # Landmark Nodes
        ####################################
        F = Agents_History[i][a]['FeatureSet'] # this is numpy array variable
        Fidx = Agents_History[i][a]['FeatureIdxSet'] # this is a list variable
        state_obs = Agents_History[i][a]['State_Obs']
        inserted_landmarks = set()
        kin_added = 0
        kin_ids = set()

        for l in range(len(Fidx)):

            ####################################
            # Landmark Factors
            ####################################

            # Extract landmark observation                 
            feature = F[l]
            descriptor = int(Fidx[l])

            if descriptor in inserted_landmarks:
                continue
            inserted_landmarks.add(descriptor)

            # convert feature global position into local bearing/range observation
            bearing, range_ = Cartesian2BearingRange3D(state_obs, feature)

            # Add extra noise to observations
            bearing_obs = add_noise_to_Unit3(bearing, bearing_sigma)
            range_obs = range_ + np.random.uniform(low=-range_sigma, high=range_sigma, size=(1,))

            # Convert bearing range observations into feature global position to act as initial estimate
            feature_obs = BearingRange2Cartesian3D(state_obs, bearing_obs, range_obs)

            # Add node to graph and initiate with noisy observation
            exist = iskeyingraph(Graph_List[a], varL(L,descriptor, i))     
            if not exist:
                Value_List[a].insertPoint3(varL(L,descriptor, i), gtsam.Point3(feature_obs))
                Graph_List[a].add(gtsam.PriorFactorPoint3(varL(L,descriptor, i), feature_obs, landmark_prior_noise))
                frame_ngh[a].append(a+1)
                frame_idx[a].append(descriptor)
                frame[a].append(feature_obs)
                if Verbose_map: print(f'Added landmark L{descriptor,i} to graph')

            # Add factor with observation data to the graph
            Graph_List[a].add(gtsam.BearingRangeFactor3D(varX(X,a,i), varL(L,descriptor, i),
                                                    bearing_obs, range_obs,
                                                    bearing_range_noise))
            if Verbose: print(f'Added landmark factor between: X{a}_{i} -> L{descriptor,i}')



            ####################################
            # Kinematic Factors (Loop closures)
            ####################################
            if Kinem == 'n_step_Kinem' and i > sw:
                # Extract the a-th agent's frame in past sliding window of memory
                KeyFramesIdx = [keyframe[a] for keyframe in KeyFramesIdx_Hist[-sw:]]
                KeyFrames = [keyframe[a] for keyframe in KeyFrames_Hist[-sw:]]                
                exist, f, _ = find_descriptor_in_keyframes(descriptor, KeyFramesIdx)

                if exist:
                    k = i-sw+f
                    idx = np.where(np.array(KeyFramesIdx[f]) == descriptor)[0][0]
                    pos1 = forward_kinematics(KeyFrames[f][idx], i-k, step_size, com, vel, ang_vel)
                    pos2 = feature_obs
                    obs = pos2 - pos1
                    kinem_factor = gtsam.CustomFactor(kinem_noise, [varL(L,descriptor, k), varL(L,descriptor, i)],
                                                    partial(kinem_error, obs, ang_vel, vel, com, dt*step_size,
                                                            lf_error, lf_jacobian,
                                                            Lvar_l, Lobs_l, Lpar_l))
                    Graph_List[a].add(kinem_factor)
                    kin_added += 1
                    kin_ids.add(int(descriptor))
                    if Verbose_kinem: print(f'\nAdded landmark kinematic factor between: L{descriptor,k} -> L{descriptor,i}')

        ####################################
        # Communicate with neighborhood
        ####################################

        if Decentralized and cond:
            Ngh = Agents_History[i][a]['CommSet']
            Qnn = min(Qn, len(Ngh))

            selectionA = random.sample(range(0, len(Ngh)), Qnn)            
            for r in selectionA:
                qn = Ngh[r]
                qn_namespace = Agents_History[i][qn].get('FeatureIdNamespace', feature_id_namespace_policy)
                if qn_namespace != feature_id_namespace_policy:
                    if Verbose:
                        print(
                            f"Skipping agent {qn} due to namespace mismatch: "
                            f"{qn_namespace} vs {feature_id_namespace_policy}"
                        )
                    continue
                qn_agent = Agents_History[i][qn]
                nbr_idx = np.asarray(qn_agent.get("FeatureIdxSet", []), dtype=int).reshape(-1)
                nbr_fs = np.asarray(qn_agent.get("FeatureSet", np.array([]).reshape(0, 3)))
                if nbr_fs.size == 0:
                    nbr_feat = np.array([]).reshape(0, 3)
                else:
                    nbr_feat = nbr_fs.reshape(-1, 3)
                if len(nbr_idx) == 0 or len(nbr_feat) == 0 or len(nbr_idx) != len(nbr_feat):
                    if Verbose_map:
                        print(
                            f"Skipping decentralized pull from agent {qn} for agent {a} at i={i}: "
                            f"no neighbor feature buffer yet (async ordering or empty LandSet)."
                        )
                    continue
                Qll = min(Ql, len(nbr_idx))
                if Qll <= 0:
                    continue

                selectionL = random.sample(range(0, len(nbr_idx)), Qll)
                for ql in selectionL:
                    feature = nbr_feat[ql]
                    descriptor = int(nbr_idx[ql])

                    if descriptor in inserted_landmarks:
                        continue
                    inserted_landmarks.add(descriptor)

                    # convert feature global position into local bearing/range observation
                    bearing, range_ = Cartesian2BearingRange3D(state_obs, feature)

                    # Add extra noise to observations
                    bearing_obs = add_noise_to_Unit3(bearing, bearing_sigma)
                    range_obs = range_ + np.random.uniform(low=-range_sigma, high=range_sigma, size=(1,))

                    # Convert bearing range observations into feature global position to act as initial estimate
                    feature_obs = BearingRange2Cartesian3D(state_obs, bearing_obs, range_obs)

                    # Add node to graph and initiate with noisy observation
                    exist = iskeyingraph(Graph_List[a], varL(L,descriptor, i))                
                    if not exist:                                  
                        Value_List[a].insertPoint3(varL(L,descriptor, i), gtsam.Point3(feature_obs))
                        Graph_List[a].add(gtsam.PriorFactorPoint3(varL(L,descriptor, i), feature_obs, landmark_prior_noise))
                        frame_ngh[a].append(qn+1)
                        frame_idx[a].append(descriptor)
                        frame[a].append(feature_obs)
                        if Verbose_map: print(f'Added landmark L{descriptor,i} to graph - through neighborhood communication')

                    # Add factor with observation data to the graph
                    Graph_List[a].add(gtsam.BearingRangeFactor3D(varX(X,a,i), varL(L,descriptor, i),
                                                            bearing_obs, range_obs,
                                                            bearing_range_noise))
                    if Verbose: print(f'Added landmark factor between: X{a}_{i} -> L{descriptor,i} - through neighborhood communication')


        ####################################
        # Target Center of Mass
        ####################################

        if FGO_Param and frame[a]:
            # define center of mass factor symbolic equations and jacobians        
            comf_error, Lvar_com, Lmap_com = center_of_mass_factor_error(len(frame[a]))
            comf_jacobian = jacobian(comf_error, Lvar_com)

            # calculate and set initial estimate
            map_avg = calculate_com(Agents_History[i-1][a]['MapSet'])
            com_obs = calculate_com(frame[a])
            if not np.isnan(map_avg).any():
                com_obs = (com_obs + map_avg) / 2

            # insert target COM variable at time step in graph
            Value_List[a].insertPoint3(varP(P,a,i) , gtsam.Point3(com_obs))


            # Add prior with observed landmarks
            Graph_List[a].add(gtsam.PriorFactorPoint3(varP(P,a,i), com_obs, pvw_noise))

            # list landmarks variable keys and add target com variable key
            list_symbs = descriptor2symbols(L,frame_idx[a],i)
            list_symbs.append(varP(P,a,i))

            # define and add factor node
            com_factor = gtsam.CustomFactor(pvw_noise, list_symbs,
                                            partial(com_error, com_obs, map_avg,
                                                    comf_error, comf_jacobian,
                                                    Lvar_com, Lmap_com))
            Graph_List[a].add(com_factor)


        ####################################
        # Target Velocity
        ####################################
        if FGO_Param and i > 1 and frame[a] and KeyFrames_Hist[i-1][a]:
            # calculate observation from landmarks:
            com2_obs = calculate_com(frame[a])
            com1_obs = calculate_com(KeyFrames_Hist[i-1][a])

            if not np.isnan(map_avg).any():
                com2_obs = (com2_obs + map_avg) / 2
                com1_obs = (com1_obs + map_avg) / 2

            vel_obs = (com2_obs - com1_obs) / (dt * step_size)

            # insert target vel variable at time step in graph            
            Value_List[a].insertPoint3(varV(V,a,i),gtsam.Point3(vel_obs))

            # Add prior with observed landmarks
            Graph_List[a].add(gtsam.PriorFactorPoint3(varV(V,a,i), vel_obs, pvw_noise))


            # define and add factor node
            list_symbs = [varV(V,a,i), varP(P,a,i-1), varP(P,a,i)]
            vel_factor = gtsam.CustomFactor(pvw_noise, list_symbs,
                                            partial(vel_error, vel_obs, dt*step_size,
                                                    velf_error, velf_jacobian,
                                                    Lvar_vel, Lpar_vel))
            Graph_List[a].add(vel_factor)


        ####################################
        # Target Angular Velocity
        ####################################
        if FGO_Param and i > 1 and frame[a] and KeyFrames_Hist[i-1][a]:
            # extract observations
            points, points_prev = extract_landmarks(frame[a],
                                                    frame_idx[a],
                                                    KeyFrames_Hist[i-1][a],
                                                    KeyFramesIdx_Hist[i-1][a])

            if points:
                # calculate and set initial estimate
                ang_obs = calculate_w(points, points_prev, dt*step_size) 

                # insert target COM variable at time step in graph
                Value_List[a].insertPoint3(varW(W,a,i) , gtsam.Point3(ang_obs))

                # Add prior with observed landmarks
                Graph_List[a].add(gtsam.PriorFactorPoint3(varW(W,a,i), ang_obs, pvw_noise))

                # list landmarks variable keys and add target com variable key
                list_symbs = descriptor2symbols(L,frame_idx[a],i)
                list_symbs = list_symbs + descriptor2symbols(L,KeyFramesIdx_Hist[i-1][a],i-1)
                list_symbs.append(varW(W,a,i))

                # define and add factor node
                ang_vel_factor = gtsam.CustomFactor(pvw_noise, list_symbs,
                                                partial(ang_vel_error, ang_obs,
                                                        dt*step_size, calculate_w,
                                                        len(points)))
                Graph_List[a].add(ang_vel_factor)
            else:
                # insert target COM variable at time step in graph
                Value_List[a].insertPoint3(varW(W,a,i) , gtsam.Point3(ang_vel))

        Agents_History[i][a]['KinLoopClosuresAdded'] = int(kin_added)
        Agents_History[i][a]['KinLoopClosuresUnique'] = int(len(kin_ids))


        ####################################################################
        # Back End
        ####################################################################
        if Verbose: print('\nBack-End running...')

        # Marginalize graph structure
        if i > 3*sw: Graph_List[a], Value_List[a] = marginalize_factor_graph(Graph_List[a], Value_List[a], 3*sw, i)       

        if Verbose:
            print(f"\nNumber of factors in graph of agent {a}: {Graph_List[a].size()}")
            num_variables = len([key for key in Value_List[a].keys()])
            print(f"Number of variables in the values object: {num_variables}\n")


        # Optimize variables
        isam = gtsam.ISAM2(parameters)
        # print(f'Agent: {a}') # Debug line
        isam.update(Graph_List[a], Value_List[a])
        result = isam.calculateEstimate()


        # Compute marginals
        # marginals = gtsam.Marginals(Graph_List[a], result)


        # Clear Variables
        # Value_List[a].clear()



        ####################################
        # Update State Estimates
        ####################################
        # Low pass filter (LERP for positions and SLERP for quaternions) if desired
        prev_pose = Agents_History[i-1][a]['State_Estim'] if i > 0 else None
        new_pose = result.atPose3(varX(X, a, i))
        _lp_co = float(low_pass_filter_coeff)
        _lp_th = getattr(config, "pose_low_pass_disable_unc_threshold", None)
        if _lp_th is not None and float(config.unc) <= float(_lp_th):
            _lp_co = 1.0
        filtered_pose = low_pass_filter_pose(prev_pose, new_pose, _lp_co)
        Agents_History[i][a]['State_Estim'] = filtered_pose

        # No low filter pass filter
        # Agents_History[i][a]['State_Estim'] = result.atPose3(varX(X,a,i))


        ####################################
        # Update Map Estimates
        ####################################

        Map = copy.deepcopy(Agents_History[i-1][a]['MapSet'])
        MapIdx = copy.deepcopy(Agents_History[i-1][a]['MapIdxSet'])
        MapNgh = copy.deepcopy(Agents_History[i-1][a]['MapNghSet'])
        # MapEntropy = copy.deepcopy(Agents_History[i-1][a]['MapEntropy'])

        ################################################################
        ################################################################

        if i > sw and Update_Sliding_Window:
            # Extract the a-th agent's frame in past sliding window of memory
            KeyFramesIdx = [keyframe[a] for keyframe in KeyFramesIdx_Hist[-sw:]]
            KeyFrames = [keyframe[a] for keyframe in KeyFrames_Hist[-sw:]]
            KeyFramesNgh = [keyframe[a] for keyframe in KeyFramesNgh_Hist[-sw:]]

            #### Sweep over Map features and look them up in keyframes to update
            for descriptor in MapIdx:
                exist, f, m = find_descriptor_in_keyframes(descriptor, KeyFramesIdx)
                idx = np.where(np.array(MapIdx) == descriptor)[0][0]

                # Update map feature
                if exist:
                    k = i-sw+f

                    # position
                    feature_result = result.atPoint3(varL(L,descriptor,k))   
                    feature_result = forward_kinematics(feature_result, i-k, step_size, com, vel, ang_vel)
                    Map[idx] = feature_result

                    # covariance
                    # feature_cov = marginals.marginalCovariance(varL(L,descriptor,k))
                    # feature_cov = forward_covariance(feature_cov, i-k, step_size, ang_vel, noise = 0.05)                    
                    # MapEntropy[idx] = Gaussian_Entropy(feature_cov)

                elif not exist:
                    Map[idx] = forward_kinematics(Map[idx], 1, step_size, com, vel, ang_vel) # One time step
                    # MapEntropy[idx] = Gaussian_Entropy(forward_covariance(feature_cov, 1, step_size, ang_vel, noise = 0.05))               


            #### Sweep over key frames and lookup features not in map to add them
            for j in range(len(KeyFramesIdx)-1, -1, -1): # reverse sweeping
                for m in range(len(KeyFramesIdx[j])):
                    descriptor = KeyFramesIdx[j][m]
                    if descriptor not in MapIdx:
                        k = i-sw+j

                        # position
                        feature_result = result.atPoint3(varL(L,descriptor,k))
                        feature_result = forward_kinematics(feature_result, i-k, step_size, com, vel, ang_vel)

                        # covariance
                        # feature_cov = marginals.marginalCovariance(varL(L,descriptor,k))
                        # feature_cov = forward_covariance(feature_cov, i-k, step_size, ang_vel, noise = 0.05)                    

                        Map.append(feature_result)
                        MapIdx.append(descriptor)
                        MapNgh.append(KeyFramesNgh[j][m])
                        # MapEntropy.append(Gaussian_Entropy(feature_cov))

        ################################################################
        ################################################################

        else:
            #### Sweep over Map features and look them up in current keyframe to update
            for descriptor in MapIdx:
                idx = np.where(np.array(MapIdx) == descriptor)[0][0]
                if descriptor in frame_idx[a]:
                    Map[idx] = result.atPoint3(varL(L,descriptor,i))
                else:
                    Map[idx] = forward_kinematics(Map[idx], 1, step_size, com, vel, ang_vel) # One time step

            #### Sweep over current keyframe and lookup features not in map to add them
            for m in range(len(frame_idx[a])):
                descriptor = frame_idx[a][m]
                if descriptor not in MapIdx:
                    feature_result = result.atPoint3(varL(L,descriptor,i))
                    Map.append(feature_result)
                    MapIdx.append(descriptor)
                    MapNgh.append(frame_ngh[a][m])

        ################################################################
        ################################################################ 

        outliers_removed_distance = 0
        outliers_removed_stale = 0
        if enable_outlier_rejection:
            Map_filtered = []
            MapIdx_filtered = []
            MapNgh_filtered = []
            for pos, lm_id, lm_ngh in zip(Map, MapIdx, MapNgh):
                lm_id_int = int(lm_id)
                pos_arr = np.array(pos, dtype=np.float64)

                if np.linalg.norm(pos_arr - np.array(com, dtype=np.float64)) > outlier_max_distance_from_com:
                    outliers_removed_distance += 1
                    registry.remove_landmark(lm_id_int)
                    continue

                if lm_id_int in registry.tracks:
                    frames_stale = i - registry.tracks[lm_id_int].last_seen_frame
                    if frames_stale > outlier_max_stale_frames:
                        outliers_removed_stale += 1
                        registry.remove_landmark(lm_id_int)
                        continue

                Map_filtered.append(pos)
                MapIdx_filtered.append(lm_id_int)
                MapNgh_filtered.append(lm_ngh)

            Map = Map_filtered
            MapIdx = MapIdx_filtered
            MapNgh = MapNgh_filtered

        Agents_History[i][a]['OutliersRemovedDistance'] = int(outliers_removed_distance)
        Agents_History[i][a]['OutliersRemovedStale'] = int(outliers_removed_stale)

        Agents_History[i][a]['MapSet'] = copy.deepcopy(Map)
        Agents_History[i][a]['MapIdxSet'] = copy.deepcopy(MapIdx)
        Agents_History[i][a]['MapNghSet'] = copy.deepcopy(MapNgh)
        for lm_id, lm_pos in zip(MapIdx, Map):
            lm_id_int = int(lm_id)
            if lm_id_int in registry.tracks:
                registry.update_position(lm_id_int, np.array(lm_pos, dtype=np.float64))
        # Agents_History[i][a]['MapEntropy'] = copy.deepcopy(MapEntropy)
        Agents_History[i][a]['MapGrowth'] = len(Agents_History[i][a]['MapSet'])

        if Verbose_map: print(f'Map size of agent {a}: {len(Agents_History[i][a]["MapSet"])}')


        # Provisional target state for dense-map propagation. The COM below may be
        # refined from the dense map, while sparse correspondences still drive V/W.
        Agents_History[i][a]['Target_Estim'] = np.concatenate((
            np.array(com, dtype=np.float64),
            np.array(vel, dtype=np.float64),
            np.array(ang_vel, dtype=np.float64),
        ))

        #######################################################################
        # Dense map (Phase B) - optional via map_mode
        #######################################################################
        if map_mode in ("dense", "hybrid"):
            _dense_map_pose_source = getattr(config, "dense_map_pose_source", "state_estim")
            _dense_map_pose_source_unc_tiny = getattr(config, "dense_map_pose_source_unc_tiny", None)
            if _dense_map_pose_source_unc_tiny is not None and float(config.unc) <= float(
                getattr(config, "pose_low_pass_disable_unc_threshold", 0.0)
            ):
                _dense_map_pose_source = _dense_map_pose_source_unc_tiny
            Agents_History[i][a].update(
                build_merged_map(
                    Agents_History,
                    a,
                    i,
                    sw,
                    step_size,
                    voxel_size,
                    pose_source=_dense_map_pose_source,
                )
            )
            Agents_History[i][a].update(_build_shared_dense_map(Agents_History, a, i, voxel_size))
            merged_pts = Agents_History[i][a].get('MergedMapSet', np.array([]).reshape(0, 3))
            merged_pcd = o3d.geometry.PointCloud()
            merged_pcd.points = o3d.utility.Vector3dVector(merged_pts)
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(Target_Point_Cloud_History[i])
            Agents_History[i][a]['MergedMap_Error'] = compute_merged_map_error(
                merged_pcd, target_pcd, voxel_size, icp_threshold
            )


        #######################################################################
        #######################################################################
        # Calculate Target Parameters
        #######################################################################
        #######################################################################
        Points, Points_prev = extract_landmarks(Agents_History[i][a]['MapSet'],
                                                Agents_History[i][a]['MapIdxSet'],
                                                Agents_History[i-1][a]['MapSet'],
                                                Agents_History[i-1][a]['MapIdxSet'])

        ####################################################################
        # Center of Mass
        ####################################################################
        # Agents_History[i][a]['Target_COM'] = calculate_com(Target_Point_Cloud_History[i])

        ##########
        dense_com_points = _as_points_array(
            Agents_History[i][a].get('MergedMapSharedSet', np.array([]).reshape(0, 3))
        )
        if len(dense_com_points) == 0:
            dense_com_points = _as_points_array(
                Agents_History[i][a].get('MergedMapSet', np.array([]).reshape(0, 3))
            )

        if len(dense_com_points) > 0:
            Agents_History[i][a]['Target_COM'] = calculate_com(dense_com_points)
        elif Points:
            Agents_History[i][a]['Target_COM'] = calculate_com(Points)
        else:
            Agents_History[i][a]['Target_COM'] = Agents_History[i-1][a]['Target_COM']

        Agents_History[i][a]['Target_COM'] = low_pass_filter(Agents_History[i][a]['Target_COM'],
                                                             Agents_History[i-1][a]['Target_COM'],
                                                             low_pass_filter_coeff)

        ##################################
        if FGO_Param and frame[a]:
            Agents_History[i][a]['FGO_Target_COM'] = result.atPoint3(varP(P,a,i))
        else:
            Agents_History[i][a]['FGO_Target_COM'] = Agents_History[i-1][a]['FGO_Target_COM']

        Agents_History[i][a]['FGO_Target_COM'] = low_pass_filter(Agents_History[i][a]['FGO_Target_COM'],
                                                             Agents_History[i-1][a]['FGO_Target_COM'],
                                                             low_pass_filter_coeff)


        ####################################################################
        # Translational velocity
        ####################################################################
        # Agents_History[i][a]['Target_V'] = calculate_v(Target_Point_Cloud_History[i],
        #                                                 Target_Point_Cloud_History[i-1], dt*step_size)

        ##########
        if Points:
            Agents_History[i][a]['Target_V'] = calculate_v(Points, Points_prev, dt*step_size)
        else:
            Agents_History[i][a]['Target_V'] = Agents_History[i-1][a]['Target_V']

        Agents_History[i][a]['Target_V'] = low_pass_filter(Agents_History[i][a]['Target_V'],
                                                             Agents_History[i-1][a]['Target_V'],
                                                             low_pass_filter_coeff)

        # vsw = 10 # Velocity Sliding Window using savgol filter
        # if i > vsw:
        #     Agents_History[i][a]['Target_V'] = calculate_v_savgol(extract_agent_history(a,Agents_History),
        #                                                       time_step=step_size*dt,
        #                                                       window_length=vsw,
        #                                                       polyorder=5)
        # else:
        #     Agents_History[i][a]['Target_V'] = Agents_History[i-1][a]['Target_V']

        ##################################
        if FGO_Param and i > 1 and frame[a] and KeyFrames_Hist[i-1][a]:
            Agents_History[i][a]['FGO_Target_V'] = result.atPoint3(varV(V,a,i))
        else:
            Agents_History[i][a]['FGO_Target_V'] = Agents_History[i-1][a]['FGO_Target_V']

        Agents_History[i][a]['FGO_Target_V'] = low_pass_filter(Agents_History[i][a]['FGO_Target_V'],
                                                             Agents_History[i-1][a]['FGO_Target_V'],
                                                             low_pass_filter_coeff)

        ####################################################################
        # Angular velocity
        ####################################################################
        # Agents_History[i][a]['Target_W'] = calculate_w4(Target_Point_Cloud_History[i],
        #                                                 Target_Point_Cloud_History[i-1], dt*step_size)

        ##########
        if Points:
            Agents_History[i][a]['Target_W'] = calculate_w4(Points, Points_prev, dt*step_size)
        else:
            Agents_History[i][a]['Target_W'] = Agents_History[i-1][a]['Target_W']

        Agents_History[i][a]['Target_W'] = low_pass_filter(Agents_History[i][a]['Target_W'],
                                                             Agents_History[i-1][a]['Target_W'],
                                                             low_pass_filter_coeff)

        ##################################
        if FGO_Param and i > 1 and frame[a] and KeyFrames_Hist[i-1][a]:
            Agents_History[i][a]['FGO_Target_W'] = result.atPoint3(varW(W,a,i))
        else:
            Agents_History[i][a]['FGO_Target_W'] = Agents_History[i-1][a]['FGO_Target_W']

        Agents_History[i][a]['FGO_Target_W'] = low_pass_filter(Agents_History[i][a]['FGO_Target_W'],
                                                             Agents_History[i-1][a]['FGO_Target_W'],
                                                             low_pass_filter_coeff)

        if _use_truth_target_kinematics:
            Agents_History[i][a]['Target_COM'] = np.array(Agents_History[i][a]['Target_True'][:3], dtype=np.float64)
            Agents_History[i][a]['Target_V'] = np.array(Agents_History[i][a]['Target_True'][3:6], dtype=np.float64)
            Agents_History[i][a]['Target_W'] = np.array(Agents_History[i][a]['Target_True'][6:9], dtype=np.float64)

        # Compatibility field for merged-map utilities
        Agents_History[i][a]['Target_Estim'] = np.concatenate((
            np.array(Agents_History[i][a]['Target_COM'], dtype=np.float64),
            np.array(Agents_History[i][a]['Target_V'], dtype=np.float64),
            np.array(Agents_History[i][a]['Target_W'], dtype=np.float64),
        ))

        #######################################################################
        #######################################################################
        # Calculate error
        #######################################################################
        #######################################################################

        Agents_History[i][a]['Visible_FGO_Error'] = visible_fgo_error(Agents_History[i][a], Target_Point_Cloud_History[i])
        Agents_History[i][a]['Total_FGO_Error'] = total_fgo_error(Agents_History[i][a], Target_Point_Cloud_History[i])



        # Calculate the elapsed time for this iteration for this agent
        Agents_History[i][a]['CPU_time'] = time.time() - start_time_iteration_agent

    if map_mode in ("dense", "hybrid"):
        for a_shared in range(N):
            Agents_History[i][a_shared].update(
                _build_shared_dense_map(Agents_History, a_shared, i, voxel_size)
            )

    KeyFrames_Hist.append(frame)
    KeyFramesIdx_Hist.append(frame_idx)        
    KeyFramesNgh_Hist.append(frame_ngh)

    #####################################################################################################
    # Calculate the elapsed time for this iteration
    elapsed_time = time.time() - start_time

    # Calculate the average time per iteration
    avg_time_per_iteration = elapsed_time / (i + 1)

    # Calculate the estimated time left for the remaining iterations
    iterations_left = num_iter - i - 1
    estimated_time_left = avg_time_per_iteration * iterations_left

    # Convert the estimated time left to a human-readable format
    estimated_time_left_str = str(timedelta(seconds=estimated_time_left))

    mode_run = slam_state.get("mode", "batch")
    global_iter = slam_state.get("current_global_iteration")
    global_time = slam_state.get("current_global_sim_time")
    slam_n = slam_state.get("slam_update_counter")
    expected_steps = slam_state.get("expected_sim_iterations")

    if mode_run == "online" and slam_n is not None and slam_n >= 1:
        g_part = ""
        if global_iter is not None:
            denom = ""
            if expected_steps is not None:
                denom = f" / {expected_steps} sim steps total"
            g_part = f"Global sim step index {global_iter}{denom} @ t={global_time:.5f}s | "
        print(
            f'[SLAM] iSAM DDFGO++ — agents={N} | '
            f'SLAM update #{slam_n} | '
            f'{g_part}'
            f'Estimated time left: {estimated_time_left_str}'
        )
    elif global_iter is None:
        print(
            f'[SLAM] iSAM DDFGO++ — agents={N} | '
            f'SLAM timestep index {i} / {max(num_iter - 1, 1)} | '
            f'Estimated time left: {estimated_time_left_str}'
        )
    else:
        denom = ""
        if expected_steps is not None:
            denom = f" / {expected_steps} sim steps total"
        print(
            f'[SLAM] iSAM DDFGO++ — agents={N} | '
            f'SLAM timestep index {i} / {max(num_iter - 1, 1)} | '
            f'Global sim step index {global_iter}{denom} @ t={global_time:.5f}s | '
            f'Estimated time left: {estimated_time_left_str}'
        )


    ########################################################################################################################
    # Delete unused keys for memory saving
    ########################################################################################################################
    for a in range(N):
        del Agents_History[i][a]['CommSet']
        del Agents_History[i][a]['LandSet']
        del Agents_History[i][a]['DockSet']
        del Agents_History[i][a]['CollSet']
        del Agents_History[i][a]['AntFlkSet']
        del Agents_History[i][a]['LC']
        del Agents_History[i][a]['LCD']
        del Agents_History[i][a]['LCD_Frame']
        del Agents_History[i][a]['Odometry']
        del Agents_History[i][a]['Target']

        # delete all vizualization data to make the pickle saving lightweight INSHALLAH
        # del Agents_History[i-1][a]['FeatureSet']
        # del Agents_History[i-1][a]['FeatureIdxSet']
        # del Agents_History[i-1][a]['MapSet']
        # del Agents_History[i-1][a]['MapIdxSet']
        # del Agents_History[i-1][a]['MapNghSet']





    ########################################################################################################################
    # Save using Pickle module for Plotting and Printing in a separate code
    ########################################################################################################################

    # Save periodically based on config
    save_every_online = max(1, getattr(config, "save_every_slam_updates", 10))
    if mode_run == "online":
        save_every_batch = save_every_online
        do_save_chk = slam_n >= 1 and (slam_n % save_every_batch == 0)
    else:
        save_every_batch = max(1, int((num_iter - 1) / max(1, config.save_num_chunks)))
        do_save_chk = i % save_every_batch == 0
    if do_save_chk:
        if mode_run == "online":
            print(f"\n[SLAM-SAVE] Saving checkpoint pickles (every {save_every_batch} SLAM updates)\n")
        else:
            print(f"\n[SLAM-SAVE] Saving checkpoint pickles (batch save every={save_every_batch} timestep indices)\n")
        save_pickle_files(Agents_History, Target_History, config)
        print('[SLAM-SAVE] Two pickle files saved')
        print(f'[SLAM-SAVE] Check files with name tag: {tag} \n')
        if config.enable_notify:
            notify(
                f"DDFGO++ progress: Iteration {i}/{num_iter-1} saved.",
                title="DDFGO++ Progress",
                topic=config.notify_topic,
                verbose=True,
            )

    _capture_slam_globals(slam_state)



def initialize_slam_batch(prepared_histories, cfg: DdfgoConfig, runtime: RuntimeDerived):
    slam_state = {}
    slam_state.update(_runtime_bindings(cfg, runtime))
    slam_state.update(prepared_histories)
    # Prepared histories resolve auto namespace policy after N is known.
    slam_state["feature_id_namespace_policy"] = prepared_histories["feature_id_namespace_policy"]

    slam_state["slam_update_counter"] = 0

    print(f"\n[SLAM-INIT] Number of agents: {slam_state['N']}")
    print(f"[SLAM-INIT] Number of iterations (SLAM frames loaded): {slam_state['num_iter']}")
    print(f"[SLAM-INIT] Configuration tag: {cfg.config_module.get_results_tag()}\n")
    if cfg.config_module.enable_notify:
        notify(
            f"DDFGO++ started | Agents={slam_state['N']}, Iter={slam_state['num_iter']}, Tag={cfg.config_module.get_results_tag()}",
            title="DDFGO++ Started",
            topic=cfg.config_module.notify_topic,
            verbose=True,
        )

    _initialize_slam_batch_body(slam_state)
    return slam_state

def run_slam_timestep(slam_state, i):
    slam_state["i"] = i
    _run_slam_timestep_body(slam_state)

def run_slam_batch(slam_state):
    for i in range(1, slam_state["num_iter"]):
        run_slam_timestep(slam_state, i)
    return {
        "Agents_History": slam_state["Agents_History"],
        "Target_History": slam_state["Target_History"],
        "Target_Point_Cloud_History": slam_state["Target_Point_Cloud_History"],
        "tag": slam_state["tag"],
        "num_iter": slam_state["num_iter"],
        "N": slam_state["N"],
        "state": slam_state,
    }


def save_batch_results(results, config_module=config):
    save_pickle_files(results["Agents_History"], results["Target_History"], config_module)


def _execute_script_body(cfg: DdfgoConfig):
    histories = load_simulation_histories(cfg.config_module)
    slam_state = initialize_slam_runtime(histories, config_module=cfg.config_module, mode="batch")
    results = run_slam_batch(slam_state)

    print('\n')
    end_time = time.time()
    runtime_seconds = end_time - slam_state["start_time"]
    hours, remainder = divmod(runtime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    end_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print(f"End time: {end_time_formatted}")
    print(f"Runtime: {int(hours)} hours, {int(minutes)} minutes, {seconds} seconds")
    print('\n')

    print('[SLAM-SAVE] Saving files using Pickle...')
    save_batch_results(results, cfg.config_module)

    print('\n[SLAM-SAVE] Two pickle files saved')
    if cfg.config_module.enable_notify:
        notify(
            f"DDFGO++ completed | Agents={results['N']}, Iter={results['num_iter']}, Tag={results['tag']}",
            title="DDFGO++ Complete",
            topic=cfg.config_module.notify_topic,
            verbose=True,
        )

    print('\n')
    print('%'*30)
    print('iSAM Computation Finished !!!!')
    print('%'*30)
    print('\n')
    print('CODE COMPILED WITHOUT ERRORS')
    print(f'Check files with names ending in "{results["tag"]}"')
    print('\n')
    return results


def run_batch_from_config(config_module=config):
    cfg = DdfgoConfig.from_module(config_module)
    return _execute_script_body(cfg=cfg)


def main():
    return run_batch_from_config()


if __name__ == "__main__":
    main()