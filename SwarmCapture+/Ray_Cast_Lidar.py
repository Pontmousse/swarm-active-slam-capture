import pybullet as p
import numpy as np
import open3d as o3d
import Load_Target as lt
import Controllers as C

def lidar_sensor(body_id, sensor_pos, num_rays_theta, num_rays_phi, max_distance, 
                 visualize_rays, visualize_hits, visualize_fraction, cond_viz, lidar_fov):
    # Get the current position and orientation of the body
    body_pos, body_rot = p.getBasePositionAndOrientation(body_id)

    # Calculate the ray directions in the conic field of view
    ray_directions = []
    for i in range(num_rays_theta):
        for j in range(num_rays_phi):
            # Calculate the azimuthal angle (θ) and polar angle (φ)
            theta = i * (lidar_fov / (num_rays_theta-1))-lidar_fov/2
            phi = j * (lidar_fov / (num_rays_phi-1))-lidar_fov/2
            
            # Convert the spherical coordinates to Cartesian coordinates
            ray_dir = [
                np.cos(phi) * np.sin(theta),
                np.cos(phi) * np.cos(theta),
                np.sin(phi)
            ]

            # Transform the ray direction according to the body's orientation
            ray_dir = p.rotateVector(body_rot, ray_dir)
            ray_dir = [-ray_dir[0], -ray_dir[1], -ray_dir[2]] # flip direction
            ray_directions.append(ray_dir)

    # Get body rotation matrix
    body_rot_mat = np.array(p.getMatrixFromQuaternion(body_rot)).reshape(3, 3)
    # Rotation matrix around z axis by 90 degrees
    rot_mat = np.array(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])).reshape(3, 3)
    # Get the final rotation matrix
    matrix = np.matmul(body_rot_mat, rot_mat)

    # Cast the rays from the Lidar sensor
    ray_from = np.array(body_pos) + matrix.dot(sensor_pos)
    ray_to = [ray_from + max_distance * np.array(dir) for dir in ray_directions]
    ray_results = p.rayTestBatch([ray_from]*len(ray_directions), ray_to)

    # Visualize the raycasts
    if (visualize_rays or visualize_hits) and cond_viz:
        visualize_every = int(1 / visualize_fraction)
        for i, result in enumerate(ray_results[::visualize_every]):
            # Each result is a tuple (objectUniqueId, linkIndex, hit_fraction, hit_position, hit_normal)
            if result[0] != -1:  # Check if the ray hit something
                if visualize_rays:
                    # Draw a line from the Lidar sensor to the hit position
                    p.addUserDebugLine(ray_from, result[3], [1, 0, 0])
                if visualize_hits:
                    # Draw a point at the hit position
                    new_hit_point = result[3]-0.00005*np.array(ray_directions[i*visualize_every])
                    p.addUserDebugText(".", new_hit_point, textColorRGB=[1, 1, 0], textSize=2)
            else:
                if visualize_rays:
                    # Draw a line from the Lidar sensor in the ray direction
                    p.addUserDebugLine(ray_from, ray_to[i*visualize_every], [0, 1, 0])

    return ray_results



def Cone_pcd_Detection(body_id, target_pcd, cond_viz, viz_dir, max_distance, fov_angle):

    # Get the current position and orientation of the body
    body_pos, body_rot = p.getBasePositionAndOrientation(body_id)
    rot_matrix = np.array(p.getMatrixFromQuaternion(body_rot)).reshape(3, 3)

    # set pcd state and extract points and normals
    points = np.array(target_pcd.points)
    normals = np.array(target_pcd.normals)

    # Cone parameters
    apex = body_pos+np.array([-0.20,0,0]) # Apex of the cone
    direction = rot_matrix[1] # Direction vector of the cone (should be normalized)
    radius = max_distance
    cone_angle_rad = fov_angle # Cone angle in radians
    normal_threshold_rad = np.radians(75) # Normal alignment threshold in radians

    # Filter points
    Landm = []
    M = len(points)
    for i in range(M):
        po = points[i]
        no = normals[i]
        v = is_point_valid(po, no, apex, direction, radius, cone_angle_rad, normal_threshold_rad)
        if v:
            Landm.append(po)
    Lc = np.mean(Landm, axis=0)


    # Visualize sensor direction
    if viz_dir and cond_viz:
        
        # Draw the sensor direction
        endpoint = np.array(apex) + 2 * direction
        line_color = [1, 1, 1]  # White line
        line_width = 3  # Adjust the line width as needed
        p.addUserDebugLine(np.array(apex), endpoint, line_color, lineWidth=line_width)

        # Draw the attitude
        C.plot_ort_rot(apex, rot_matrix, 0)

        # Draw the points sensed
        for i in range(len(Landm)):
            point = Landm[i]
            p.addUserDebugText(".", point, textColorRGB=[0, 1, 1], textSize=2)


    return Landm, Lc


def is_point_valid(point, normal, apex, direction, radius, cone_angle_rad, normal_threshold_rad):
    
    valid = False

    # Unit vector from apex to point
    LineOS = point - apex
    distance = np.linalg.norm(LineOS)
    LineOS_normalized = LineOS / distance

    # Check if the point is inside the cone
    inside_cone = False
    cos_angle = np.dot(LineOS_normalized, direction)
    if cos_angle >= np.cos(cone_angle_rad):
        inside_cone = True

    # Check if the normal is aligned with the LOS to the point
    normal_aligned = False
    normal_normalized = normal / np.linalg.norm(normal)
    cos_normal_angle = np.dot(normal_normalized, -LineOS_normalized)
    if cos_normal_angle >= np.cos(normal_threshold_rad):
        normal_aligned = True

    # Check if the point is within the specified distance from the apex
    within_distance = False
    if distance <= radius:
        within_distance = True

    if inside_cone and normal_aligned and within_distance:
        valid = True

    return valid