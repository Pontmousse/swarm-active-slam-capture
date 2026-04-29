import pybullet as p
import numpy as np
import open3d as o3d

def Set_Landmarks_Detected(ray_results):
    
    Landm = []
    Lc = np.zeros(3)

    count = 0
    for i, result in enumerate(ray_results):
        if result[0] == 0: # check if result[0] == 0 or 1 or 2 or ... or b-1
            Landm.append(result[3])
            Lc += np.array(result[3])
            count += 1

    if count > 0:
        Lc = Lc/count
    else:
        Lc = []

    return Landm, Lc

def Landmarks_Detected(target_pcd, agent_pos, Rdet):
    Landm = []
    Lc = np.zeros(3)
    
    print(target_pcd)
    
    count = 0
    for i, point in enumerate(target_pcd):
        print(point)
        Distance = np.linalg.norm(agent_pos - point)
        if Distance < Rdet:
            Landm.append(point)
    
    if count > 0:
        Lc = Lc/count
    
    return Landm, Lc
    
def Set_Features_Indices_Using_KDTree(Landm, target_pcd, target_kdtree, agent_state, max_lidar_pcd, max_angle_pcd, max_dist):   
    Indices = []
    F = np.array(target_pcd.points)
    N = np.array(target_pcd.normals)
    search_radius = max_lidar_pcd  # Adjust as needed
    max_neighbors = 30    # Adjust as needed
    
    
    for i in range(len(Landm)):
        point_lidar = Landm[i]
        
        # Search in KDTree Structure (run time: Hybrid < knn < radius (or rnn))
        # [M, idx, _] = target_kdtree.search_hybrid_vector_3d(point_lidar, search_radius, max_neighbors)
        [M, idx, _] = target_kdtree.search_knn_vector_3d(point_lidar, max_neighbors)
        # [M, idx, _] = target_kdtree.search_radius_vector_3d(point_lidar, search_radius)

        for j in range(M):
            point_pcd = F[idx[j]]
            line_of_sight = point_pcd-agent_state[:3]
            normal_pcd = N[idx[j]]
            angle = np.dot(normal_pcd, line_of_sight)
            D = np.linalg.norm(line_of_sight)

            if D < max_dist:
                Indices.append(idx[j])

            # if angle < np.cos(np.radians(max_angle_pcd)):
            #     if len(Indices) != 0:
            #         if idx[j] not in Indices:
            #             Indices.append(idx[j])
            #     else:
            #         Indices.append(idx[j])

        # Debug
        # print(len(Indices))

    return Indices


def downsample_features(Indices, M):
        
    if M < len(Indices):

        # M is the number of features to randmly extract from the Indices list
        selected_indices = np.random.choice(len(Indices), M, replace=False)

        # Extract elements from these indices
        Indices = [Indices[i] for i in selected_indices]

    return Indices

def Observe_Features_from_Indices(target_pcd, Indices, std):
    # std is standard deviation

    # Extract feature positions
    F = np.array(target_pcd.points)
    Features = F[Indices]

    # Add noise to observations
    # Noise = np.random.normal(loc=0.0, scale=std, size=Features.shape) # Normal Distribution
    Noise = np.random.uniform(low=-std, high=std, size=Features.shape) # Uniform Distribution
    Features = Features+Noise

    return Features


def Set_Attachment_Points_Detected(attachment_points, iter, agent_pos, agent_lcd, Rdet_ap):
    APs = []

    for k in range(len(attachment_points)):
        point = attachment_points[k]

        distance = np.linalg.norm(np.array(agent_pos) - point.position[iter])
        angle = np.dot(point.normal[iter], -np.array(agent_lcd))

        if distance < Rdet_ap and angle > np.cos(np.radians(45)):
            APs.append(point.idx)

    return APs