import pickle
import matplotlib.pyplot as plt
import numpy as np
import Plot_Telemetry_Func as Telemetry
import open3d as o3d
import Load_Target as lt
import pybullet as p
import pybullet_data
import time
import copy
import math as m
import glob
import os
from moviepy.editor import ImageSequenceClip
import config
import shared_config

#############################################################################################
# Visualization settings

show_target = False # show target real pcd
show_agents = True # show agents bodies
show_observations = False # show agent's of interest observed features at time step
show_map = True # show agent's of interest map estimate at time step

# Sparse map visualization (when show_map): two triangle-mesh sphere clouds —
#   * Own landmarks: MapNghSet entry equals focal agent id → warm orange, full radius.
#   * Neighbour-attributed landmarks: MapNghSet != id (features registered to another
#     agent but present in this agent's map) → cool blue, smaller radius.


#############################################################################################

# Kinem is controlled by config.
Kinem = config.Kinem

#############################################################################################
id = 2 # Agent ID to show (1 for first, 2 for second... until max = N)
D = config.D # Simulation duration

data_paths = config.get_data_paths()
results_paths = config.get_results_paths()
path_target_pcd = data_paths['target_pcd']
path_agents = results_paths['agents']
path_target = results_paths['target']


#############################################################################################

# Load Simulation History
Agents_History = Telemetry.load_variable_from_file(path_agents)
Target_History = Telemetry.load_variable_from_file(path_target)

# Truncate to valid iterations only (handles partially-saved histories)
valid_iter = Telemetry.find_valid_iterations(Agents_History)
if valid_iter < len(Agents_History):
    print(f'Warning: Truncating data from {len(Agents_History)} to {valid_iter} valid iterations')
    Agents_History = Agents_History[:valid_iter]
    Target_History = Target_History[:valid_iter]

# limit simulation to given period only
d = 38 # in seconds
num_iter = int(d/D*len(Agents_History))
Agents_History = Agents_History[:num_iter]



# Downsample History to speed up animation
num_iter = len(Agents_History)
step_size = 1
Agents_History = [Agents_History[i] for i in range(0, num_iter, step_size)]
Target_History = [Target_History[i] for i in range(0, num_iter, step_size)]


N = len(Agents_History[0]) # Number of agents
num_iter = len(Agents_History)

##############################################################################################################################
# Functions
##############################################################################################################################

def _to_points_array(x):
    x = np.asarray(x)
    if x.size == 0:
        return np.array([]).reshape(0, 3)
    if x.ndim == 1:
        return x.reshape(-1, 3)
    return x.reshape(-1, 3)

def _spheres_from_points(points, radius, resolution):
    pts = _to_points_array(points)
    if len(pts) == 0:
        return o3d.geometry.TriangleMesh()
    base = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    base.compute_vertex_normals()
    out = o3d.geometry.TriangleMesh()
    for p_xyz in pts:
        s = o3d.geometry.TriangleMesh(base)
        s.translate(p_xyz, relative=False)
        out += s
    out.compute_vertex_normals()
    return out


def animation_camera_look_at(k):
    """Match all-agent offscreen framing: fixed eye from shared_config, center on target."""
    camera_target = np.asarray(Target_History[k][:3], dtype=np.float64)
    camera_position = np.asarray(shared_config.animation_camera_eye_xyz, dtype=np.float64)
    camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return camera_target, camera_position, camera_up


def load_agent_pcd(id,i):
    Agents = Agents_History[i]
    Spacecraft = Agents[id-1]
    Feat = Spacecraft.get('FeatureSet', np.array([]).reshape(0, 3))
    Feat = _to_points_array(Feat)
    agent_pcd = o3d.geometry.PointCloud()
    agent_pcd.points = o3d.utility.Vector3dVector(Feat)
    return agent_pcd

def load_agent_map(id, frame_idx):
    Spacecraft = Agents_History[frame_idx][id - 1]
    Map = _to_points_array(Spacecraft.get('MapSet', np.array([]).reshape(0, 3)))
    MapNgh = np.asarray(Spacecraft.get('MapNghSet', [])).reshape(-1)

    if len(MapNgh) != len(Map):
        MapNgh = np.full(len(Map), id, dtype=int)

    indices2remove = []
    for j in range(len(MapNgh)):
        ngh = MapNgh[j]
        if ngh != id:
            indices2remove.append(j)

    Points = [Map[j] for j in range(len(Map)) if j not in indices2remove]
    agent_map = o3d.geometry.PointCloud()
    agent_map.points = o3d.utility.Vector3dVector(Points)
    return agent_map

def load_agent_neighbour_map(id, frame_idx):
    Spacecraft = Agents_History[frame_idx][id - 1]

    MapNgh = np.asarray(Spacecraft.get('MapNghSet', [])).reshape(-1)
    Map = _to_points_array(Spacecraft.get('MapSet', np.array([]).reshape(0, 3)))

    if len(MapNgh) != len(Map):
        MapNgh = np.full(len(Map), id, dtype=int)

    Points = []
    for j in range(len(MapNgh)):
        ngh = MapNgh[j]
        if ngh != id:
            Points.append(Map[j])

    agent_ngh_map = o3d.geometry.PointCloud()
    agent_ngh_map.points = o3d.utility.Vector3dVector(Points)
    return agent_ngh_map


def sparse_map_own_and_neighbour_counts(id_1based, frame_idx):
    """
    Count sparse map entries for focal agent id_1based at frame_idx:
    own (MapNgh == id) vs neighbour-attributed (MapNgh != id), plus per-neighbour ID counts.
    Neighbour IDs match MapNghSet semantics (agent index convention used in history).
    """
    Spacecraft = Agents_History[frame_idx][id_1based - 1]
    Map = _to_points_array(Spacecraft.get('MapSet', np.array([]).reshape(0, 3)))
    MapNgh = np.asarray(Spacecraft.get('MapNghSet', [])).reshape(-1)
    n_map = len(Map)
    if len(MapNgh) != len(Map):
        MapNgh = np.full(len(Map), id_1based, dtype=int)
    own_mask = MapNgh == id_1based
    n_own = int(np.sum(own_mask))
    n_neighbour = n_map - n_own
    per_neighbour = {}
    if n_neighbour > 0:
        neigh_ids = MapNgh[~own_mask]
        uniq, cnts = np.unique(neigh_ids, return_counts=True)
        per_neighbour = {int(u): int(c) for u, c in zip(uniq, cnts)}
    return n_own, n_neighbour, per_neighbour

def load_agent_merged_map(id, i):
    Agents = Agents_History[i]
    Spacecraft = Agents[id-1]
    pts = _to_points_array(Spacecraft.get('MergedMapSet', np.array([]).reshape(0, 3)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if len(pts) > 0 and config.animation_dense_voxel_size and config.animation_dense_voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(config.animation_dense_voxel_size))
    return pcd

##############################################################################################################################
# Generating geometries
##############################################################################################################################

# load global reference frame
grf = o3d.geometry.TriangleMesh.create_coordinate_frame()

# load initial target pcd
target_pcd = o3d.geometry.PointCloud()
Target_Point_Cloud_History = Telemetry.load_variable_from_file(path_target_pcd)
target_pcd.points = o3d.utility.Vector3dVector(Target_Point_Cloud_History[0])
target_pcd.paint_uniform_color([0.9, 0.9, 0.9])
search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=40)
if len(target_pcd.points) > 0:
    target_pcd.estimate_normals(search_param)

# downsample target pcd as needed
if config.animation_target_voxel_size and config.animation_target_voxel_size > 0:
    target_pcd = target_pcd.voxel_down_sample(voxel_size=float(config.animation_target_voxel_size))

# load initial agent observations
agent_pcd = load_agent_pcd(id,0)

# load initial agent maps (as point clouds for counting; sphere meshes used for rendering)
agent_map = load_agent_map(id,0)
agent_ngh_map = load_agent_neighbour_map(id,0)
agent_map_own_spheres = _spheres_from_points(
    np.asarray(agent_map.points),
    radius=config.animation_sparse_sphere_radius,
    resolution=config.animation_sparse_sphere_resolution,
)
agent_map_ngh_spheres = _spheres_from_points(
    np.asarray(agent_ngh_map.points),
    radius=max(0.5 * config.animation_sparse_sphere_radius, 1e-6),
    resolution=config.animation_sparse_sphere_resolution,
)

# load initial merged dense map
agent_merged_map = load_agent_merged_map(id, 0)


# load initial agents cubes
Agent_Geometries = []
for a in range(N):
    agent_box = o3d.geometry.TriangleMesh.create_box(
        width=float(config.animation_cubesat_size_m),
        height=float(config.animation_cubesat_size_m),
        depth=float(config.animation_cubesat_size_m),
    )
    agent_box.compute_vertex_normals()

    if a == (id-1):
        agent_box.paint_uniform_color([0, 0, 0])
    else:
        agent_box.paint_uniform_color([0.4, 0.9, 0.7])

    pos = Agents_History[0][a]['State'][:3]
    dq = Agents_History[0][a]['State'][6:10]
    lt.o3d_update_geom(agent_box, pos, dq)
    Agent_Geometries.append(agent_box)

##############################################################################################################################
# Main
##############################################################################################################################

# Set up paths
output_dir = config.MODULE_DIR / "Outputs"
frame_folder = str(output_dir / "Movie_frames_single_agent")
output_video_path = str(output_dir / "Mapping_Animation_Single_Agent.mp4")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(frame_folder, exist_ok=True)

# Animation parameters
width, height = 1920, 1080  # Image High resolution
# width, height = 800, 600  # Image Low resolution
fps = 30  # Frames per second for the final video

# grf_material definition
grf_material = o3d.visualization.rendering.MaterialRecord()
grf_material.base_color = [0.5, 0.5, 0.5, 1.0]  # RGB and Fourth input is opacity
grf_material.shader = "defaultLit"  # Use a basic shader that works without additional lighting

# target_pcd_material definition
target_pcd_material = o3d.visualization.rendering.MaterialRecord()
target_pcd_material.base_color = [0.9, 0.9, 0.9, 1.0]  # RGB and Fourth input is opacity
target_pcd_material.shader = "defaultLit"  # Use a basic shader that works without additional lighting

# agent_pcd_material definition
agent_pcd_material = o3d.visualization.rendering.MaterialRecord()
agent_pcd_material.base_color = [0.1, 1, 1, 1.0]  # RGB and Fourth input is opacity
agent_pcd_material.shader = "defaultLit"  # Use a basic shader that works without additional lighting

# agent_map_material definition
agent_map_material = o3d.visualization.rendering.MaterialRecord()
agent_map_material.base_color = [1.0, 0.55, 0.15, 1.0]  # warm orange
agent_map_material.shader = "defaultLit"

# agent_map2_material definition
agent_ngh_map_material = o3d.visualization.rendering.MaterialRecord()
agent_ngh_map_material.base_color = [0.35, 0.65, 1.0, 1.0]  # cool blue
agent_ngh_map_material.shader = "defaultLit"

# merged dense map material
agent_merged_map_material = o3d.visualization.rendering.MaterialRecord()
agent_merged_map_material.base_color = [0.55, 0.55, 0.85, 0.9]  # subdued violet
agent_merged_map_material.shader = "defaultUnlit"
agent_merged_map_material.point_size = float(config.animation_dense_point_size)

# agent_box_material definition
agent_box_material = o3d.visualization.rendering.MaterialRecord()
agent_box_material.base_color = [0.4, 0.9, 0.7, 1.0] # RGB and Fourth input is opacity
agent_box_material.shader = "defaultLit"  # Use a basic shader that works without additional lighting

# agent_box_material definition
agent2show_box_material = o3d.visualization.rendering.MaterialRecord()
agent2show_box_material.base_color = [0.4, 0.1, 0.7, 1.0]  # RGB and Fourth input is opacity
agent2show_box_material.shader = "defaultLit"  # Use a basic shader that works without additional lighting


# Set up the visualizer
vis = o3d.visualization.rendering.OffscreenRenderer(width, height)
# Set the background color (white or transparent)
if config.animation_dark_background:
    vis.scene.set_background(list(config.animation_background_rgba))
else:
    vis.scene.set_background([1.0, 1.0, 1.0, 1.0])  # White background


# Add the geometries
vis.scene.add_geometry('grf', grf, grf_material)
if show_target: vis.scene.add_geometry('target_pcd', target_pcd, target_pcd_material)
if show_map:
    vis.scene.add_geometry('agent_map_own_spheres', agent_map_own_spheres, agent_map_material)
    vis.scene.add_geometry('agent_map_ngh_spheres', agent_map_ngh_spheres, agent_ngh_map_material)
    if config.animation_show_dense_map:
        vis.scene.add_geometry('agent_merged_map', agent_merged_map, agent_merged_map_material)
if show_observations: vis.scene.add_geometry('agent_pcd', agent_pcd, agent_pcd_material)
for a in range(N):
    agent_name = 'agent'+str(a)+'_box'
    if show_agents:
        if a != id-1: vis.scene.add_geometry(agent_name, Agent_Geometries[a], agent_box_material)
        elif a == id-1: vis.scene.add_geometry(agent_name, Agent_Geometries[a], agent2show_box_material)

frac = 1
dt = 1/240
num_iter = int((len(Agents_History)))
duration = dt*num_iter
iter = int(m.floor(num_iter/frac))

# Loop through frames and call the callback
for k in range(iter):
    # Target
    pos = Target_History[k][:3]
    quat = Target_History[k][6:10]
    quat_previous = Target_History[k-1][6:10]
    dq = p.getDifferenceQuaternion(quat_previous,quat)
    lt.o3d_update_geom(target_pcd,pos,dq)

    #########################

    # Agent Landmark Observations
    new_agent_pcd = load_agent_pcd(id,k)
    agent_pcd.points = new_agent_pcd.points
    agent_pcd.paint_uniform_color([0.1, 1, 1])

    print('Number of current landmarks observed:  '+str(len(agent_pcd.points)))
    # print('Mean of current landmarks observed:  '+str(np.mean(agent_pcd.points, axis=0)[0]))
    
    #########################

    # Agent Landmark Map
    new_agent_map = load_agent_map(id,k)
    agent_map.points = new_agent_map.points
    # Agent Landmark Neighbour Map
    new_agent_ngh_map = load_agent_neighbour_map(id,k)
    agent_ngh_map.points = new_agent_ngh_map.points
    agent_map_own_spheres = _spheres_from_points(
        np.asarray(agent_map.points),
        radius=config.animation_sparse_sphere_radius,
        resolution=config.animation_sparse_sphere_resolution,
    )
    agent_map_ngh_spheres = _spheres_from_points(
        np.asarray(agent_ngh_map.points),
        radius=max(0.5 * config.animation_sparse_sphere_radius, 1e-6),
        resolution=config.animation_sparse_sphere_resolution,
    )

    # Agent merged dense map (optional)
    if config.animation_show_dense_map:
        agent_merged_map = load_agent_merged_map(id, k)

    n_own, n_ngh_sparse, ngh_by_agent = sparse_map_own_and_neighbour_counts(id, k)
    print(
        f'  [frame {k}] Sparse map spheres — own (orange): {n_own} | '
        f'neighbour-attributed (blue): {n_ngh_sparse}'
        + (f' | by neighbour agent id: {ngh_by_agent}' if ngh_by_agent else ' | (no neighbour-attributed landmarks; blue layer empty)')
    )

    #########################

    # Agents bodies
    for a in range(N):
        pos = Agents_History[k][a]['State'][:3]
        quat = Agents_History[k][a]['State'][6:10]
        quat_previous = Agents_History[k-1][a]['State'][6:10]
        dq = p.getDifferenceQuaternion(quat_previous,quat)
        lt.o3d_update_geom(Agent_Geometries[a],pos,dq)
    
    ############################################################################################################

    # Remove geometries
    vis.scene.remove_geometry("grf")
    if show_target: vis.scene.remove_geometry('target_pcd')
    if show_map:
        vis.scene.remove_geometry('agent_map_own_spheres')
        vis.scene.remove_geometry('agent_map_ngh_spheres')
        if config.animation_show_dense_map:
            vis.scene.remove_geometry('agent_merged_map')
    if show_observations: vis.scene.remove_geometry('agent_pcd')
    for a in range(N):
        agent_name = 'agent'+str(a)+'_box'
        if show_agents: vis.scene.remove_geometry(agent_name)

    # Add geometries
    vis.scene.add_geometry('grf', grf, grf_material)
    if show_target: vis.scene.add_geometry('target_pcd', target_pcd, target_pcd_material)
    if show_map:
        vis.scene.add_geometry('agent_map_own_spheres', agent_map_own_spheres, agent_map_material)
        vis.scene.add_geometry('agent_map_ngh_spheres', agent_map_ngh_spheres, agent_ngh_map_material)
        if config.animation_show_dense_map:
            vis.scene.add_geometry('agent_merged_map', agent_merged_map, agent_merged_map_material)
    if show_observations: vis.scene.add_geometry('agent_pcd', agent_pcd, agent_pcd_material)
    for a in range(N):
        agent_name = 'agent'+str(a)+'_box'
        if show_agents:
            if a != id-1: vis.scene.add_geometry(agent_name, Agent_Geometries[a], agent_box_material)
            elif a == id-1: vis.scene.add_geometry(agent_name, Agent_Geometries[a], agent2show_box_material)


    ############################################################################################################

    camera_target, camera_position, camera_up = animation_camera_look_at(k)
    vis.scene.camera.look_at(camera_target, camera_position, camera_up)

    # vis the scene
    image = vis.render_to_image()

    # Save the frame
    frame_path = f"{frame_folder}/frame_{k:04d}.png"
    o3d.io.write_image(frame_path, image)
    print(f"\nSaved frame {k + 1}/{iter}")


# Create video from saved frames using moviepy
image_files = [f"{frame_folder}/frame_{i:04d}.png" for i in range(iter)]
clip = ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(output_video_path, codec="libx264", preset="slow", bitrate="8000k")
print(f"Video saved as {output_video_path}")

print(f"Deleting frame images from {frame_folder}\n...")
# Delete all frame images in the frame_folder
frame_images = glob.glob(f"{frame_folder}/frame_*.png")  # Get all frame images in the folder
for image in frame_images:
    os.remove(image)  # Remove each image file
print("Frame images deleted.")





