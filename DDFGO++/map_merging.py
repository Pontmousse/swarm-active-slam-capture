import numpy as np
import open3d as o3d
import gtsam
from helper import forward_kinematics


def store_scan_local(LandSet_noisy, state, a, i, downsample_voxel_size=None, key_suffix=""):
    """
    Store scan in local/sensor frame.
    
    Args:
        LandSet_noisy: Nx3 numpy array of points in world frame
        state: gtsam.Pose3 pose
        a: agent ID
        i: timestep
        downsample_voxel_size: Optional voxel size for downsampling before storage (None = no downsampling)
        key_suffix: Suffix for dictionary keys (e.g., "_Odom", "_Static", "_ICP", "_EKF")
    
    Returns:
        Dictionary with 'ScanLocal{suffix}' (Nx3 numpy array) and 'ScanPoseKey{suffix}' (tuple)
    """
    scan_key = f"ScanLocal{key_suffix}"
    pose_key = f"ScanPoseKey{key_suffix}"
    
    if len(LandSet_noisy) == 0:
        return {scan_key: np.array([]).reshape(0, 3), pose_key: (a, i)}
    
    # Downsample before storing if voxel_size is provided
    if downsample_voxel_size is not None and downsample_voxel_size > 0:
        LandSet_noisy = merge_voxel(LandSet_noisy, downsample_voxel_size)
    
    # Convert from world frame to local/sensor frame
    R = state.rotation().matrix()
    t = np.array(state.translation())
    ScanLocal = (R.T @ (LandSet_noisy - t).T).T  # Nx3
    
    return {
        scan_key: ScanLocal,
        pose_key: (a, i)
    }


def transform_points(pose, pts_local):
    """
    Transform Nx3 points from local frame to world frame using gtsam.Pose3.
    
    Args:
        pose: gtsam.Pose3 transformation
        pts_local: Nx3 numpy array of points in local frame
    
    Returns:
        Nx3 numpy array of points in world frame
    """
    if len(pts_local) == 0:
        return np.array([]).reshape(0, 3)
    
    R = pose.rotation().matrix()
    t = np.array(pose.translation())
    pts_world = (R @ pts_local.T).T + t  # Nx3
    
    return pts_world


def forward_kinematics_pointcloud(pts_world, steps, step_size, com, vel, ang_vel):
    """
    Vectorized version of forward_kinematics for point clouds.
    Propagates points forward in time using target motion model.
    
    Args:
        pts_world: Nx3 numpy array of points in world frame
        steps: number of time steps to propagate
        step_size: step size multiplier
        com: center of mass (3,)
        vel: velocity (3,)
        ang_vel: angular velocity (3,)
    
    Returns:
        Nx3 numpy array of propagated points
    """
    if len(pts_world) == 0:
        return np.array([]).reshape(0, 3)
    
    # Vectorized propagation
    pts_propagated = pts_world.copy()
    for idx in range(len(pts_propagated)):
        pts_propagated[idx] = forward_kinematics(pts_propagated[idx], steps, step_size, com, vel, ang_vel)
    
    return pts_propagated


def merge_voxel(points, voxel_size):
    """
    Merge point cloud using voxel downsampling.
    
    Args:
        points: Nx3 numpy array of points
        voxel_size: voxel size in meters
    
    Returns:
        Mx3 downsampled points
    """
    if len(points) == 0:
        return np.array([]).reshape(0, 3)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Voxel downsample
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)
    
    # Convert back to numpy
    points_downsampled = np.asarray(pcd_downsampled.points)
    
    return points_downsampled


def build_merged_map(Agents_History, a, i, sw, step_size, voxel_size):
    """
    Build merged dense map from sliding window scans.
    
    Args:
        Agents_History: full agent history
        a: agent ID
        i: current timestep
        sw: sliding window size
        step_size: step size for propagation
        voxel_size: voxel size for downsampling
    
    Returns:
        Dictionary with 'MergedMapSet' (Mx3 numpy array) - Open3D pcd can be created on-the-fly when needed
    """
    all_points = []
    
    # Get target parameters at current time i
    if 'Target_Estim' in Agents_History[i][a]:
        com = Agents_History[i][a]['Target_Estim'][:3]
        vel = Agents_History[i][a]['Target_Estim'][3:6]
        ang_vel = Agents_History[i][a]['Target_Estim'][6:9]
    else:
        # SwarmDDFGO++ stores target kinematics in separate fields.
        com = np.array(Agents_History[i][a].get('Target_COM', [0.0, 0.0, 0.0]), dtype=np.float64)
        vel = np.array(Agents_History[i][a].get('Target_V', [0.0, 0.0, 0.0]), dtype=np.float64)
        ang_vel = np.array(Agents_History[i][a].get('Target_W', [0.0, 0.0, 0.0]), dtype=np.float64)
    
    # Step 1: Collect all window scans
    for j in range(sw):
        k = i - sw + j + 1
        
        # Check window boundaries
        if k < 0 or k >= len(Agents_History):
            continue
        
        # Check if scan exists
        if 'ScanLocal' not in Agents_History[k][a]:
            continue
        
        ScanLocal = Agents_History[k][a]['ScanLocal']
        
        # Skip empty scans
        if len(ScanLocal) == 0:
            continue
        
        # Step 2: Transform scan using pose estimate from Agents_History        
        T_post = Agents_History[k][a]['State_Estim']
        # T_post = state_array_to_pose3(Agents_History[k][a]['State'])
        Pw_k = transform_points(T_post, ScanLocal)
        
        # Step 3: Propagate to current time i
        steps = i - k
        if steps > 0:
            Pw_i = forward_kinematics_pointcloud(Pw_k, steps, step_size, com, vel, ang_vel)
        else:
            Pw_i = Pw_k
        
        # Accumulate points
        all_points.append(Pw_i)
    
    # Step 4: Merge via voxel downsampling
    if len(all_points) == 0:
        return {
            'MergedMapSet': np.array([]).reshape(0, 3)
        }
    
    # Concatenate all points
    concatenated_points = np.vstack(all_points)
    
    # Voxel downsample
    merged_points = merge_voxel(concatenated_points, voxel_size)
    
    return {
        'MergedMapSet': merged_points
    }


def state_array_to_pose3(state_array):
    """
    Convert State array (position + quaternion) to gtsam.Pose3 format (State_Estim type).
    
    Args:
        state_array: numpy array with [x, y, z, ..., qw, qx, qy, qz] format (State array)
    
    Returns:
        gtsam.Pose3 object (State_Estim type)
    """
    pos = state_array[:3]
    quat = state_array[6:10]
    rotation = gtsam.Rot3.Quaternion(quat[3], quat[0], quat[1], quat[2])
    translation = gtsam.Point3(pos[0], pos[1], pos[2])
    return gtsam.Pose3(rotation, translation)


    
def compute_merged_map_error(merged_pcd, target_pcd, voxel_size, icp_threshold):
    """
    Compute error metrics between merged map and target point cloud (NO ICP).

    Notes:
    - This intentionally does NOT run ICP alignment. The goal is for the error curve to reflect
      your estimator/map frame consistency, not ICP's ability to re-align clouds each iteration.
    - `icp_threshold` is still used as the distance threshold for the inlier ratio.
    
    Args:
        merged_pcd: Open3D point cloud of merged map
        target_pcd: Open3D point cloud of target
        voxel_size: voxel size for downsampling
        icp_threshold: distance threshold for inlier ratio
    
    Returns:
        Dictionary with 'chamfer_distance', 'rmse_est_to_gt', 'inlier_ratio'
    """
    # Handle empty point clouds
    if len(merged_pcd.points) == 0 or len(target_pcd.points) == 0:
        return {
            'chamfer_distance': float('inf'),
            'rmse_est_to_gt': float('inf'),
            'inlier_ratio': 0.0
        }
    
    # Step 1: Downsample both point clouds to same voxel_size
    merged_down = merged_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)
    
    if len(merged_down.points) == 0 or len(target_down.points) == 0:
        return {
            'chamfer_distance': float('inf'),
            'rmse_est_to_gt': float('inf'),
            'inlier_ratio': 0.0
        }
    
    # Step 2: Compute nearest neighbor distances both ways (NO ICP)
    merged_points = np.asarray(merged_down.points)
    target_points = np.asarray(target_down.points)
    
    # Build KDTree for efficient NN search
    target_tree = o3d.geometry.KDTreeFlann(target_down)
    merged_tree = o3d.geometry.KDTreeFlann(merged_down)
    
    # est -> gt distances
    est_to_gt_distances = []
    for point in merged_points:
        [_, idx, dist] = target_tree.search_knn_vector_3d(point, 1)
        est_to_gt_distances.append(np.sqrt(dist[0]))
    
    # gt -> est distances
    gt_to_est_distances = []
    for point in target_points:
        [_, idx, dist] = merged_tree.search_knn_vector_3d(point, 1)
        gt_to_est_distances.append(np.sqrt(dist[0]))
    
    est_to_gt_distances = np.array(est_to_gt_distances)
    gt_to_est_distances = np.array(gt_to_est_distances)
    
    # Step 4: Calculate metrics
    # Chamfer distance: mean of both directions
    chamfer_distance = np.mean(est_to_gt_distances**2) + np.mean(gt_to_est_distances**2)
    
    # RMSE (est -> gt)
    rmse_est_to_gt = np.sqrt(np.mean(est_to_gt_distances**2))
    
    # Inlier ratio (points within threshold)
    inlier_ratio = (np.sum(est_to_gt_distances < icp_threshold) + 
                   np.sum(gt_to_est_distances < icp_threshold)) / (len(est_to_gt_distances) + len(gt_to_est_distances))
    
    return {
        'chamfer_distance': chamfer_distance,
        'rmse_est_to_gt': rmse_est_to_gt,
        'inlier_ratio': inlier_ratio
    }

