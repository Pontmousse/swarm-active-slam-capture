import pickle
import matplotlib.pyplot as plt
import numpy as np
import Plot_Telemetry_Func as Telemetry
import open3d as o3d
import Load_Target as lt
import pybullet as p
import pybullet_data
import time
import random
import copy
import glob
import math as m
import os
from moviepy.editor import ImageSequenceClip
import config

#############################################################################################
# Visualization settings

show_target = True # show target real pcd
show_agents = True # show agents bodies
show_map = True # show agent's of interest map estimate at time step


voxel = 0.7 # agent map voxel size
target_voxel = 1.2 # target voxel size
search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=40)


#############################################################################################

# Kinem is controlled by config.
Kinem = config.Kinem

#############################################################################################
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
d = 6
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

def _desaturate_rgba(rgba, mix=0.6):
    r, g, b, a = rgba
    gray = (r + g + b) / 3.0
    r2 = (1.0 - mix) * r + mix * gray
    g2 = (1.0 - mix) * g + mix * gray
    b2 = (1.0 - mix) * b + mix * gray
    return [r2, g2, b2, a]

def load_agent_map(a,i):
    Agents = Agents_History[i]
    Spacecraft = Agents[a]
    Map = Spacecraft['MapSet']
    MapNgh = Spacecraft['MapNghSet']

    indices2remove = []
    for i in range(len(MapNgh)):
        ngh = MapNgh[i]
        if (ngh-1) != a:
            indices2remove.append(i)

    Points = [Map[j] for j in range(len(Map)) if j not in indices2remove]
    agent_map = o3d.geometry.PointCloud()
    agent_map.points = o3d.utility.Vector3dVector(Points)

    # downsample pcd as needed
    agent_map.estimate_normals(search_param)
    agent_map = agent_map.voxel_down_sample(voxel_size = voxel)

    return agent_map

def load_agent_merged_map(a, i):
    Spacecraft = Agents_History[i][a]
    pts = _to_points_array(Spacecraft.get('MergedMapSet', np.array([]).reshape(0, 3)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if len(pts) > 0 and config.animation_dense_voxel_size and config.animation_dense_voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(config.animation_dense_voxel_size))
    return pcd


def generate_matching_color_lists_rgba(N):
    if N > 10 or N < 1:
        raise ValueError("N must be between 1 and 10.")
    
    # Base list of 10 distinct colors in hexadecimal
    color_codes = [        
        "#FF0000",  # Red
        "#00FF00",  # Green
        "#0000FF",  # Blue
        "#FFFF00",  # Yellow
        "#FF00FF",  # Magenta
        "#00FFFF",  # Cyan
        "#800000",  # Maroon
        "#808000",  # Olive
        "#008080",  # Teal
        "#800080"   # Purple
    ]
    
    # Slightly modified colors to create matching pairs
    matching_colors = [        
        "#FF6666",  # Light Red
        "#66FF66",  # Light Green
        "#6666FF",  # Light Blue
        "#FFFF66",  # Light Yellow
        "#FF66FF",  # Light Magenta
        "#66FFFF",  # Light Cyan
        "#A05252",  # Lighter Maroon
        "#A0A052",  # Lighter Olive
        "#52A0A0",  # Lighter Teal
        "#A052A0"   # Lighter Purple
    ]
    
    def hex_to_rgba(hex_color, opacity=1.0):
        """Convert HEX color to RGBA."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        return [r, g, b, opacity]
    
    # Generate RGBA lists
    list_1 = [hex_to_rgba(color) for color in color_codes[:N]]
    list_2 = [hex_to_rgba(color) for color in matching_colors[:N]]
    
    return list_2, list_1

# Function to clip points based on the plane
def clip_point_cloud(pcd, plane_normal, plane_point):
    points = np.asarray(pcd.points)
    
    # Calculate signed distance to the plane (dot product with the normal)
    distances = np.dot(points - plane_point, plane_normal)
    
    # Keep only points where the signed distance is non-negative
    clipped_points = points[distances <= 0]
    
    # Create a new point cloud with the filtered points
    clipped_pcd = o3d.geometry.PointCloud()
    clipped_pcd.points = o3d.utility.Vector3dVector(clipped_points)
    
    return clipped_pcd

##############################################################################################################################
# Generating geometries
##############################################################################################################################

############ load global reference frame ############
grf = o3d.geometry.TriangleMesh.create_coordinate_frame()

############ load initial target pcd ############
target_pcd = o3d.geometry.PointCloud()
Target_Point_Cloud_History = Telemetry.load_variable_from_file(path_target_pcd)
target_pcd.points = o3d.utility.Vector3dVector(Target_Point_Cloud_History[0])
target_pcd.paint_uniform_color([0.9, 0.9, 0.9])
# downsample target pcd as needed
target_pcd.estimate_normals(search_param)
if config.animation_target_voxel_size and config.animation_target_voxel_size > 0:
    target_pcd = target_pcd.voxel_down_sample(voxel_size=float(config.animation_target_voxel_size))
else:
    target_pcd = target_pcd.voxel_down_sample(voxel_size = target_voxel)


############ load initial agents cubes and maps ############
Agent_Geometries = []
Agent_Map_Spheres = []
for a in range(N):
    # Agent box geometry
    agent_box = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
    agent_box.compute_vertex_normals()
    pos = Agents_History[0][a]['State'][:3]
    dq = Agents_History[0][a]['State'][6:10]
    lt.o3d_update_geom(agent_box, pos, dq)
    Agent_Geometries.append(agent_box)

    # Agent sparse map as sphere mesh
    agent_map = load_agent_map(a, 0)
    agent_spheres = _spheres_from_points(
        np.asarray(agent_map.points),
        radius=config.animation_sparse_sphere_radius,
        resolution=config.animation_sparse_sphere_resolution,
    )
    Agent_Map_Spheres.append(agent_spheres)

# Selected agent dense merged map (optional)
selected_agent_idx = int(getattr(config, "animation_selected_agent_id", 1)) - 1
selected_agent_idx = max(0, min(N - 1, selected_agent_idx))
Selected_Agent_Merged_Map = load_agent_merged_map(selected_agent_idx, 0)

##############################################################################################################################
# Main
##############################################################################################################################

# Set up paths
output_dir = config.MODULE_DIR / "Outputs"
frame_folder = str(output_dir / "Movie_frames_all_agents")
output_video_path = str(output_dir / "Mapping_Animation_All_Agents.mp4")
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

colors_maps, colors_robots = generate_matching_color_lists_rgba(N)
Agent_Map_Materials = []
Agent_Box_Materials = []
for a in range(N):
    # agent_map_material definition
    agent_map_material = o3d.visualization.rendering.MaterialRecord()
    agent_map_material.base_color = colors_maps[a]  # RGBA
    agent_map_material.shader = "defaultLit"

    # agent_box_material definition
    agent_box_material = o3d.visualization.rendering.MaterialRecord()
    agent_box_material.base_color = colors_robots[a] # RGB and Fourth input is opacity
    agent_box_material.shader = "defaultLit"  # Use a basic shader that works without additional lighting

    Agent_Map_Materials.append(agent_map_material)
    Agent_Box_Materials.append(agent_box_material)
    
selected_dense_material = o3d.visualization.rendering.MaterialRecord()
selected_dense_material.base_color = _desaturate_rgba(colors_maps[selected_agent_idx], mix=0.7)
selected_dense_material.base_color[3] = 0.9
selected_dense_material.shader = "defaultUnlit"
selected_dense_material.point_size = float(config.animation_dense_point_size)



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

for a in range(N):
    agent_name = 'agent'+str(a)+'_box'
    if show_agents: vis.scene.add_geometry(agent_name, Agent_Geometries[a], Agent_Box_Materials[a])
    
    agent_map_name = 'agent'+str(a)+'_map'
    if show_map: vis.scene.add_geometry(agent_map_name, Agent_Map_Spheres[a], Agent_Map_Materials[a])

if show_map and config.animation_show_dense_map:
    vis.scene.add_geometry('selected_agent_merged_map', Selected_Agent_Merged_Map, selected_dense_material)


frac = 1
dt = 1/240
num_iter = int((len(Agents_History)))
duration = dt*num_iter
iter = int(m.floor(num_iter/frac))

# Loop through frames
for k in range(iter):
    id = 1
    camera_target = Target_History[k][:3]


    ######## Uncomment to fix camera at agent
    # calculate target to one of the agents vector
    tar2agent_vec = np.array(camera_target) - np.array(Agents_History[k][id-1]['State'][:3])
    dis2agent = np.linalg.norm(tar2agent_vec)
    tar2agent_vec = tar2agent_vec/dis2agent
    camera_position = np.array(camera_target) + tar2agent_vec*1.3*dis2agent
    
    # Set up the camera parameters manually
    camera_position = [-12,3,3]
    

    # calculate camera to target vector
    cam2tar_vec = np.array(camera_target) - np.array(camera_position)
    dis2cam = np.linalg.norm(cam2tar_vec)

    # Define the clipping plane
    # depth = 1 is clipping exactly at the target center of mass (the target position)
    # Set to smaller for closer clipping distance, but not smaller than 0
    depth = 1
    plane_normal = cam2tar_vec/dis2cam
    plane_point = np.array(camera_position) + plane_normal*depth*dis2cam

    #########################

    # Target
    pos = Target_History[k][:3]
    quat = Target_History[k][6:10]
    quat_previous = Target_History[k-1][6:10]
    dq = p.getDifferenceQuaternion(quat_previous,quat)
    lt.o3d_update_geom(target_pcd,pos,dq)

    # # Uncomment to clip target
    # target_pcd = o3d.geometry.PointCloud()
    # target_pcd.points = o3d.utility.Vector3dVector(Target_Point_Cloud_History[k])
    # target_pcd.paint_uniform_color([0.9, 0.9, 0.9])
    # # downsample target pcd as needed
    # target_pcd.estimate_normals(search_param)
    # target_pcd = target_pcd.voxel_down_sample(voxel_size = target_voxel)
    # target_pcd = clip_point_cloud(target_pcd, plane_normal, plane_point)

    #########################

    # Agents bodies
    for a in range(N):
        ####################
        # Agent box geometry
        pos = Agents_History[k][a]['State'][:3]
        quat = Agents_History[k][a]['State'][6:10]
        quat_previous = Agents_History[k-1][a]['State'][6:10]
        dq = p.getDifferenceQuaternion(quat_previous,quat)
        lt.o3d_update_geom(Agent_Geometries[a],pos,dq)

        ####################
        # Agent map geometry
        new_agent_map = load_agent_map(a,k)

        # Clip the point cloud to show only the points in front of the camera
        clipped_pcd = clip_point_cloud(new_agent_map, plane_normal, plane_point)

        # Save sparse map as spheres (visual-only)
        Agent_Map_Spheres[a] = _spheres_from_points(
            np.asarray(clipped_pcd.points),
            radius=config.animation_sparse_sphere_radius,
            resolution=config.animation_sparse_sphere_resolution,
        )

        
    
    # Selected dense merged map
    if config.animation_show_dense_map:
        Selected_Agent_Merged_Map = load_agent_merged_map(selected_agent_idx, k)

    ############################################################################################################

    # Remove geometries
    vis.scene.remove_geometry("grf")
    if show_target: vis.scene.remove_geometry('target_pcd')
    for a in range(N):
        agent_name = 'agent'+str(a)+'_box'
        if show_agents: vis.scene.remove_geometry(agent_name)

        agent_map_name = 'agent'+str(a)+'_map'
        if show_map: vis.scene.remove_geometry(agent_map_name)

    if show_map and config.animation_show_dense_map:
        vis.scene.remove_geometry('selected_agent_merged_map')

    # Add geometries
    vis.scene.add_geometry('grf', grf, grf_material)
    if show_target: vis.scene.add_geometry('target_pcd', target_pcd, target_pcd_material)
    for a in range(N):
        agent_name = 'agent'+str(a)+'_box'
        if show_agents: vis.scene.add_geometry(agent_name, Agent_Geometries[a], Agent_Box_Materials[a])

        agent_map_name = 'agent'+str(a)+'_map'
        if show_map: vis.scene.add_geometry(agent_map_name, Agent_Map_Spheres[a], Agent_Map_Materials[a])

    if show_map and config.animation_show_dense_map:
        vis.scene.add_geometry('selected_agent_merged_map', Selected_Agent_Merged_Map, selected_dense_material)


    ############################################################################################################

    # Set the camera position, target, and up direction using look_at()
    view_ctl = vis.scene.camera
    camera_up = [0, 1, 0]
    view_ctl.look_at(camera_target, camera_position, camera_up)

    ###########################################################################################################

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
# clip.write_videofile(output_video_path, codec="libx264", preset="ultrafast", bitrate="20000k", ffmpeg_params=["-pix_fmt", "yuvj420p"])
print(f"Video saved as {output_video_path}\n")

print(f"Deleting frame images from {frame_folder}\n...")
# Delete all frame images in the frame_folder
frame_images = glob.glob(f"{frame_folder}/frame_*.png")  # Get all frame images in the folder
for image in frame_images:
    os.remove(image)  # Remove each image file
print("Frame images deleted.")