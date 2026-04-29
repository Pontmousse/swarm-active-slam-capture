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
from helper import *
from Custom_Factors import *
from LandmarkRegistry import LandmarkRegistry
from Feature_Processing import compute_features, add_noise_to_features, get_descriptor_dim
from map_merging import store_scan_local, build_merged_map, compute_merged_map_error
from notify_helper import notify
import config

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


def save_pickle_files(agents_history, target_history):
    results_paths = config.get_results_paths()
    os.makedirs(os.path.dirname(results_paths["agents"]), exist_ok=True)
    with open(results_paths["agents"], "wb") as file:
        pickle.dump(agents_history, file)
    with open(results_paths["target"], "wb") as file:
        pickle.dump(target_history, file)


#############################################################################################
# Load Simulation History
data_paths = config.get_data_paths()
path_agents = data_paths["agents"]
path_target = data_paths["target"]
path_target_pcd = data_paths["target_pcd"]

Agents_History = load_pickle_file(path_agents)
Target_History = load_pickle_file(path_target)
Target_Point_Cloud_History = load_pickle_file(path_target_pcd)

###########################################################################################
# Downsample History
num_iter = len(Agents_History)
Agents_History = [Agents_History[i] for i in range(0, num_iter, step_size)]
Target_History = [Target_History[i] for i in range(0, num_iter, step_size)]
Target_Point_Cloud_History = [Target_Point_Cloud_History[i] for i in range(0, num_iter, step_size)]

N = len(Agents_History[0]) # Number of agents
num_iter = len(Agents_History)

if feature_id_namespace_policy == "auto":
    feature_id_namespace_policy = "registry" if N == 1 else "source_feature_idx"
if feature_id_namespace_policy not in ("registry", "source_feature_idx"):
    raise ValueError(
        f"Unsupported feature_id_namespace_policy: {feature_id_namespace_policy}"
    )


print(f"\nNumber of agents: {N}")
print(f"Number of iterations: {num_iter}")
print(f"Configuration tag: {config.get_results_tag()}\n")
if config.enable_notify:
    notify(
        f"DDFGO++ started | Agents={N}, Iter={num_iter}, Tag={config.get_results_tag()}",
        title="DDFGO++ Started",
        topic=config.notify_topic,
        verbose=True,
    )


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

for i in range(1, num_iter):
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

        # Feedback simulated convergence depending on decay and bias coefficient
        # com = Agents_History[i][a]['Target_Measure'][:3]
        # vel = Agents_History[i][a]['Target_Measure'][3:6]
        # ang_vel = Agents_History[i][a]['Target_Measure'][6:9]

        # Feedback true value
        # com = Agents_History[i][a]['Target_True'][:3]
        # vel = Agents_History[i][a]['Target_True'][3:6]
        # ang_vel = Agents_History[i][a]['Target_True'][6:9]
        
        # feedback estimates calculated outside the factor graph
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
                Qll = min(Ql, len(Agents_History[i][qn]['FeatureIdxSet']))

                selectionL = random.sample(range(0, len(Agents_History[i][qn]['FeatureIdxSet'])), Qll)
                for ql in selectionL:
                    feature = Agents_History[i][qn]['FeatureSet'][ql]
                    descriptor = int(Agents_History[i][qn]['FeatureIdxSet'][ql])

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
        filtered_pose = low_pass_filter_pose(prev_pose, new_pose, low_pass_filter_coeff)
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
        if Points:
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

        # Compatibility field for merged-map utilities
        Agents_History[i][a]['Target_Estim'] = np.concatenate((
            np.array(Agents_History[i][a]['Target_COM'], dtype=np.float64),
            np.array(Agents_History[i][a]['Target_V'], dtype=np.float64),
            np.array(Agents_History[i][a]['Target_W'], dtype=np.float64),
        ))

        #######################################################################
        # Dense map (Phase B) - optional via map_mode
        #######################################################################
        if map_mode in ("dense", "hybrid"):
            Agents_History[i][a].update(build_merged_map(Agents_History, a, i, sw, step_size, voxel_size))
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
        # Calculate error
        #######################################################################
        #######################################################################
            
        Agents_History[i][a]['Visible_FGO_Error'] = visible_fgo_error(Agents_History[i][a], Target_Point_Cloud_History[i])
        Agents_History[i][a]['Total_FGO_Error'] = total_fgo_error(Agents_History[i][a], Target_Point_Cloud_History[i])

        
        
        # Calculate the elapsed time for this iteration for this agent
        Agents_History[i][a]['CPU_time'] = time.time() - start_time_iteration_agent

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

    print(f'iSAM DDFGO++ - № Agents = {N} | Iteration {i} / {num_iter-1} | Estimated time left: {estimated_time_left_str}')


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
    save_every = max(1, int((num_iter - 1) / max(1, config.save_num_chunks)))
    if i % save_every == 0:
        print('\nSaving files using Pickle...')
        save_pickle_files(Agents_History, Target_History)
        print('Two pickle files saved')
        print(f'Check files with name tag: {tag} \n')
        if config.enable_notify:
            notify(
                f"DDFGO++ progress: Iteration {i}/{num_iter-1} saved.",
                title="DDFGO++ Progress",
                topic=config.notify_topic,
                verbose=True,
            )
    

########################################################################################################################
# Save using Pickle module for Plotting and Printing in a separate code
########################################################################################################################

print('\n')

end_time = time.time()
runtime = end_time-start_time

# Convert the runtime to hours, minutes, and seconds
hours, remainder = divmod(runtime, 3600)
minutes, seconds = divmod(remainder, 60)

# Print the end time
end_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
print(f"End time: {end_time_formatted}")

# Print the runtime
print(f"Runtime: {int(hours)} hours, {int(minutes)} minutes, {seconds} seconds")
print('\n')


########################################################################################################################
# Save using Pickle module for Plotting and Printing in a separate code
########################################################################################################################
print('Saving files using Pickle...')
save_pickle_files(Agents_History, Target_History)

print('\nTwo pickle files saved')
if config.enable_notify:
    notify(
        f"DDFGO++ completed | Agents={N}, Iter={num_iter}, Tag={tag}",
        title="DDFGO++ Complete",
        topic=config.notify_topic,
        verbose=True,
    )

##############################################################################################################################
# End
##############################################################################################################################

print('\n')
print('%'*30)
print('iSAM Computation Finished !!!!')
print('%'*30)
print('\n')
print('CODE COMPILED WITHOUT ERRORS')
print(f'Check files with names ending in "{tag}"')
print('\n')