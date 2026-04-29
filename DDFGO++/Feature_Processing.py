"""
Feature Processing Module for SwarmDFGO+

This module provides feature descriptor computation for point cloud matching.
Currently uses FPFH with L2 normalization for improved viewpoint robustness.

Note: Open3D does not have native SHOT support. If you need true viewpoint-invariant
descriptors, consider using PCL via python-pcl or implementing custom SHOT.
The current FPFH with normalization provides reasonable results for moderate rotations.
"""

import numpy as np
import open3d as o3d
import config


# Descriptor dimension (FPFH = 33, SHOT would be 352)
DESCRIPTOR_DIM = getattr(config, 'DESCRIPTOR_DIM', 33)


def add_noise_to_features(FeatureSet, noise_std=0.01):
    """
    Add Gaussian noise to feature positions to simulate sensor noise.
    
    Args:
        FeatureSet: List or array of 3D feature positions (N, 3)
        noise_std: Standard deviation of Gaussian noise in meters (default 0.01)
        
    Returns:
        Noisy FeatureSet as numpy array (N, 3)
    """
    FeatureSet = np.array(FeatureSet)
    if len(FeatureSet) == 0:
        return FeatureSet
    noise = np.random.normal(0, noise_std, FeatureSet.shape)
    F_noisy = FeatureSet + noise
    return F_noisy


def compute_fpfh_features(FeatureSet, radius_normal=0.1, radius_feature=0.2, normalize=True):
    """
    Compute FPFH (Fast Point Feature Histograms) descriptors for all features.
    Maintains 1:1 correspondence with FeatureSet indices.
    
    Args:
        FeatureSet: Array of 3D feature positions (N, 3)
        radius_normal: Radius for normal estimation (default 0.1)
        radius_feature: Radius for FPFH feature computation (default 0.2)
        normalize: If True, L2-normalize descriptors for better matching (default True)
        
    Returns:
        FPFH descriptors array (N, 33), aligned 1:1 with FeatureSet indices
    """
    FeatureSet = np.array(FeatureSet)
    
    if len(FeatureSet) == 0:
        return np.array([]).reshape(0, 33)
    
    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(FeatureSet)
    
    # Estimate normals using all points
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30))
    
    # Orient normals consistently (helps with viewpoint changes)
    pcd.orient_normals_consistent_tangent_plane(k=15)
    
    # Compute FPFH features for all points
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    
    # Extract FPFH descriptors (33-dimensional)
    # pcd_fpfh.data is (33, N), transpose to (N, 33)
    fpfh_descriptors = np.array(pcd_fpfh.data).T
    
    # L2 normalize for better matching across viewpoints
    if normalize and len(fpfh_descriptors) > 0:
        norms = np.linalg.norm(fpfh_descriptors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        fpfh_descriptors = fpfh_descriptors / norms
    
    return fpfh_descriptors


def compute_features(FeatureSet, radius_normal=None, radius_feature=None):
    """
    Main entry point for computing feature descriptors.
    Currently uses FPFH with L2 normalization.
    
    This function can be modified to use different descriptor types
    (e.g., SHOT via PCL) without changing the calling code.
    
    Radii are automatically computed based on point cloud density if not provided.
    
    Args:
        FeatureSet: Array of 3D feature positions (N, 3)
        radius_normal: Radius for normal estimation (None = auto-compute)
        radius_feature: Radius for feature computation (None = auto-compute)
        
    Returns:
        Descriptors array (N, DESCRIPTOR_DIM), aligned 1:1 with FeatureSet indices
    """
    FeatureSet = np.array(FeatureSet)
    
    if len(FeatureSet) == 0:
        return np.array([]).reshape(0, DESCRIPTOR_DIM)
    
    # Auto-compute radii based on point cloud density if not provided
    if radius_normal is None or radius_feature is None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(FeatureSet)
        
        # Compute average nearest neighbor distance to estimate point density
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        nn_dists = []
        sample_size = min(100, len(FeatureSet))
        sample_idx = np.random.choice(len(FeatureSet), sample_size, replace=False)
        for i in sample_idx:
            [k, idx, dist] = kdtree.search_knn_vector_3d(pcd.points[i], 3)
            if len(dist) > 1:
                nn_dists.append(np.sqrt(dist[1]))  # exclude self
        
        if len(nn_dists) > 0:
            avg_nn_dist = np.mean(nn_dists)
        else:
            avg_nn_dist = 0.5  # fallback
        
        # Set radii based on point density (need enough neighbors for good FPFH)
        if radius_normal is None:
            radius_normal = max(0.1, avg_nn_dist * 2.5)
        if radius_feature is None:
            radius_feature = max(0.2, avg_nn_dist * 4.0)
    
    return compute_fpfh_features(FeatureSet, radius_normal, radius_feature, normalize=True)


def get_descriptor_dim():
    """
    Get the dimension of the feature descriptors.
    
    Returns:
        int: Descriptor dimension (33 for FPFH, 352 for SHOT)
    """
    return DESCRIPTOR_DIM
