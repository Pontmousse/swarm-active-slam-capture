import pybullet as p
import numpy as np
import open3d as o3d
import copy
import random
from datetime import timedelta
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import shared_config

OBJECT_PARAMS = {
    "Turksat": {"ply_scale": 0.001, "urdf_scale": 2.0},
    "Nonconvex": {"ply_scale": 0.03, "urdf_scale": 2.0},
    "Orion_Capsule": {"ply_scale": 0.0025, "urdf_scale": 0.05 / 0.8850},
    "Motor": {"ply_scale": 0.030, "urdf_scale": 0.30},
    "Separation_Stage": {"ply_scale": 0.15, "urdf_scale": 1.0},
}

def load_target(target_position, target_orientation, texTar_id, object_name=None):

    ########################################################################################################################
    # Load PLY and URDF from: \Targets

    object_name = object_name or shared_config.object_name
    if object_name not in OBJECT_PARAMS:
        raise ValueError(f"Unsupported object_name '{object_name}'. Supported: {sorted(OBJECT_PARAMS)}")
    ply_scale = OBJECT_PARAMS[object_name]["ply_scale"]
    urdf_scale = OBJECT_PARAMS[object_name]["urdf_scale"]
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=40)

    ########################################################################################################################

    urdf_path = MODULE_DIR / "Targets" / object_name / f"{object_name}.urdf"
    target_body_id = p.loadURDF(str(urdf_path), basePosition=target_position, baseOrientation=target_orientation, globalScaling=urdf_scale)
    p.changeVisualShape(target_body_id, -1, rgbaColor=[1, 1, 1, 1])
    p.changeVisualShape(target_body_id, -1, textureUniqueId=texTar_id)

    # Get Z bounds for the URDF using getAABB
    urdf_aabb_min, urdf_aabb_max = p.getAABB(target_body_id)
    urdf_z_min, urdf_z_max = urdf_aabb_min[2], urdf_aabb_max[2]
    urdf_z_range = urdf_z_max - urdf_z_min

    ########################################################################################################################
    ply_path = MODULE_DIR / "Targets" / object_name / f"{object_name}.PLY"
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    mesh.compute_vertex_normals()
    mesh.scale(ply_scale, center=mesh.get_center())
    
    # Get Z bounds for the PLY model
    vertices = np.asarray(mesh.vertices)
    ply_z_min = np.min(vertices[:, 2])
    ply_z_max = np.max(vertices[:, 2])
    ply_z_range = ply_z_max - ply_z_min

    ########################################################################################################################
    # Calculate the scaling ratio
    scaling_ratio = urdf_z_range / ply_z_range

    print(f"URDF Z Range: {urdf_z_range:.4f}")
    print(f"PLY Z Range: {ply_z_range:.4f}")
    print(f"Scaling Ratio: {scaling_ratio:.4f}")

    ########################################################################################################################
    
    number_of_points = 3000
    target_pcd = mesh.sample_points_uniformly(number_of_points)
    target_pcd.estimate_normals(search_param)

    target_pcd.paint_uniform_color([0, 0, 0])

    o3d_update_geom(target_pcd, target_position, target_orientation)

    return target_body_id, target_pcd, object_name


def o3d_update_geom(pcd, pos, dq):

    R = np.array(p.getMatrixFromQuaternion(dq)).reshape(3, 3)
    pcd.rotate(R, center=pcd.get_center())
    pcd.translate(pos, relative=False)

    return


def plot_target_pybullet(pcd, voxel):
    
    downpcd = copy.deepcopy(pcd)
    downpcd = downpcd.voxel_down_sample(voxel_size=voxel)

    P = downpcd.points
    M = len(P)
    for i in range(M):
        point = P[i]
        p.addUserDebugText(".", point, textColorRGB=[0, 0, 0], textSize=2)

    return


##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

class Attachment_Point:
    def __init__(self, idx):
        self.idx = idx
        self.position = []
        self.velocity = []
        self.normal = []

    def add_iteration(self, target_state, target_pcd):
        if self.idx > len(target_pcd.points):
            raise SyntaxError(f'\nID point "{self.idx}" in file is higher than point cloud size ({len(target_pcd.points)} point)\n')

        # ap position
        pos = target_pcd.points[self.idx]
        self.position.append(np.array(pos))

        # ap velocity
        tar_com = target_state[:3]
        tar_vel = target_state[3:6]
        tar_angvel = target_state[10:13]
        self.velocity.append(np.cross(tar_angvel, pos - tar_com) + tar_vel)


        # Estimate normals using Open3D's built-in function
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=40)
        target_pcd.estimate_normals(search_param)
        
        # Retrieve the normal of the point at `self.idx`
        normals = np.asarray(target_pcd.normals)
        self.normal.append(normals[self.idx])


        return self

def read_coordinates_from_file(filename):
    coordinates = []  # This will hold our coordinates as tuples (x, y, z)
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line and not line.startswith("#"):  # Ignore empty lines and comments
                try:
                    # Split the line into components and convert to float
                    x, y, z = map(lambda v: round(float(v), 6), line.split(","))
                    coordinates.append((x, y, z))
                except ValueError:
                    # Handle the case where the line is not in the correct format
                    print(f"Skipping invalid line: {line}")
                    
    return coordinates

def find_closest_points_KDTree(coordinates, target_pcd):
    # Convert attachment points to numpy array
    attachment_points_np = coordinates
    
    # Build KDTree using target point cloud points
    kdtree = o3d.geometry.KDTreeFlann(target_pcd)
    
    closest_point_indices = []
    for point in attachment_points_np:
        [_, idx, _] = kdtree.search_knn_vector_3d(point, 1)  # Find the closest point
        closest_point_indices.append(idx[0])  # Take the first index (closest and only point that should be in the output)
    
    return closest_point_indices

def load_predefined_attachment_points(object_name, target_pcd):

    # get pre-defined target ids from file
    path_to_ap_coordinates = MODULE_DIR / "Targets" / object_name / f"{object_name}_keypoints_1.txt"
    
    attachment_point_coordinates = read_coordinates_from_file(str(path_to_ap_coordinates))
    
    # downsample attachment point (reduces computation time and makes it easy for agent to place bids)
    n = 30
    if n > len(attachment_point_coordinates):
        raise ValueError("n cannot be greater than the number of available attachment points")
    attachment_point_coordinates = random.sample(attachment_point_coordinates, n)


    attachment_point_ids = find_closest_points_KDTree(attachment_point_coordinates,
                                                      target_pcd)
    
    #Debug
    # print('target point cloud x,y,z ranges min-max')
    # print(np.min(np.asarray(target_pcd.points), axis=0))
    # print(np.max(np.asarray(target_pcd.points), axis=0))
    # print('\n')
    # print(attachment_point_coordinates)
    # print(attachment_point_ids)
    
    
    attachment_points = []
    for id in attachment_point_ids:
        attachment_point = Attachment_Point(id)
        attachment_point.idx = id
        attachment_points.append(attachment_point)

    return attachment_points


#### Testing functions
if __name__ == "__main__":
    
    physicsClient = p.connect(p.DIRECT)
    target_position = [0, 0, 0]
    target_orientation = p.getQuaternionFromEuler([0, 0, 0])
    texTar_id = p.loadTexture(str(MODULE_DIR / "Targets" / "Texture_Target.jpg"))
    target_body_id, target_pcd, object_name = load_target(target_position,target_orientation,texTar_id)


    target_state = [1,2,3,0.2,0.5,0.6,0.7,0.7,0,0,0.01,0.03,0.05]

    attachment_points = load_predefined_attachment_points(object_name, target_pcd)
    
    for ap in attachment_points:
        ap.add_iteration(target_state, target_pcd)




    iter = 0
    print('\n')
    print(attachment_points[8].normal[iter])
    print(np.linalg.norm(attachment_points[8].normal[iter]))


    print('\n')
    print('Function file test compiled without error')