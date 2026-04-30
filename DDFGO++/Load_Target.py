import pybullet as p
import numpy as np
import open3d as o3d
import copy
import time
from datetime import timedelta
from pathlib import Path
import sys

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import shared_config

OBJECT_PARAMS = {
    "Turksat": {"ply_scale": 0.001, "urdf_scale": 2.0},
    "Nonconvex": {"ply_scale": 0.03, "urdf_scale": 2.0},
    "Orion_Capsule": {"ply_scale": 0.001, "urdf_scale": 0.04},
    "Motor": {"ply_scale": 0.012, "urdf_scale": 0.12},
    "Separation_Stage": {"ply_scale": 0.025, "urdf_scale": 20.0},
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

    urdf_path = MODULE_DIR / 'Targets' / object_name / f'{object_name}.urdf'
    target_body_id = p.loadURDF(str(urdf_path), basePosition=target_position, baseOrientation=target_orientation, globalScaling=urdf_scale)
    p.changeVisualShape(target_body_id, -1, rgbaColor=[1, 1, 1, 1])
    p.changeVisualShape(target_body_id, -1, textureUniqueId=texTar_id)



    ########################################################################################################################
    ply_path = MODULE_DIR / 'Targets' / object_name / f'{object_name}.PLY'
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    mesh.compute_vertex_normals()
    mesh.scale(ply_scale, center=mesh.get_center())
    
    number_of_points = 2000
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