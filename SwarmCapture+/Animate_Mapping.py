import pickle
import matplotlib.pyplot as plt
import numpy as np
import Plot_Telemetry_Func as Telemetry
import open3d as o3d
import Load_Target as lt
import pybullet as p
import pybullet_data
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import shared_config
CUBESAT_SIZE_M = float(shared_config.VIS_CUBESAT_SIZE_M)

# Connect to PyBullet and set up the simulation
physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)

#############################################################################################
paths = shared_config.get_sim_data_paths()
tag = paths["tag"]
id = 1 # Agent ID to show

#############################################################################################

path_agents = paths["agents"]
path_target = paths["target"]

# Load Simulation History
Agents_History = Telemetry.load_variable_from_file(path_agents)
Target_History = Telemetry.load_variable_from_file(path_target)
N = len(Agents_History[0]) # Number of agents
num_iter = len(Agents_History)

##############################################################################################################################
# Functions
##############################################################################################################################

def load_agent_pcd(id,i):
    Landm = []
    Agents = Agents_History[i]
    Spacecraft = Agents[id-1]
    Landm = Spacecraft['FeatureSet']
    agent_pcd = o3d.geometry.PointCloud()
    agent_pcd.points = o3d.utility.Vector3dVector(Landm)
    return agent_pcd

def load_agent_map(id,i):
    Landm = []
    Agents = Agents_History[i]
    Spacecraft = Agents[id-1]
    Landm = Spacecraft['MapSet']
    agent_map = o3d.geometry.PointCloud()
    agent_map.points = o3d.utility.Vector3dVector(Landm)
    return agent_map

##############################################################################################################################
# Custom Animation Callback
##############################################################################################################################

def custom_animation_callback(vis):
    # Sleep run to slow down animation
    seconds = 0.02
    time.sleep(seconds)

    # Initialize index
    i = dict['current_index']
    i += 1
    k = i % num_iter

    # Target
    pos = Target_History[k][:3]
    quat = Target_History[k][6:10]
    quat_previous = Target_History[k-1][6:10]
    dq = p.getDifferenceQuaternion(quat_previous,quat)
    lt.o3d_update_geom(target_pcd,pos,dq)

    # Agent Landmark Observations
    new_agent_pcd = load_agent_pcd(id,k)
    agent_pcd.points = new_agent_pcd.points
    agent_pcd.paint_uniform_color([0, 0.6, 1])
    print('Number of current landmarks observed:  '+str(len(agent_pcd.points)))
    # print('Mean of current landmarks observed:  '+str(np.mean(agent_pcd.points, axis=0)[0]))
    
    # Agent Landmark Map
    new_agent_map = load_agent_map(id,k)
    agent_map.points = new_agent_map.points
    agent_map.paint_uniform_color([1, 0.6, 0])
    print('Number of map landmarks:  '+str(len(agent_map.points)))
    # print('Geometric center of landmarks map:  '+str(np.mean(agent_map.points, axis=0)[0]))

    # Agents bodies
    for a in range(N):
        pos = Agents_History[k][a]['State'][:3]
        quat = Agents_History[k][a]['State'][6:10]
        quat_previous = Agents_History[k-1][a]['State'][6:10]
        dq = p.getDifferenceQuaternion(quat_previous,quat)
        lt.o3d_update_geom(Agent_Geometries[a],pos,dq)
    
    # Update step
    dict['current_index'] = k

    # Update the point cloud in the visualization
    vis.update_geometry(target_pcd)
    vis.update_geometry(agent_pcd)
    vis.update_geometry(agent_map)
    vis.update_geometry(agent_box)
    for a in range(N):
        vis.update_geometry(Agent_Geometries[a])

    return False

##############################################################################################################################
# Main
##############################################################################################################################

# Set up initial dictionary with initial index
dict = {'current_target_state': Target_History,
        'current_agents_state': Agents_History,
        'current_index': 0}

################################################
################################################

# load global reference frame
grf = o3d.geometry.TriangleMesh.create_coordinate_frame()

# load initial target pcd
path_target_pcd = paths["target_pcd"]
target_pcd = o3d.geometry.PointCloud()
target_pcd.points = o3d.utility.Vector3dVector(Telemetry.load_variable_from_file(path_target_pcd)[0])
target_pcd.paint_uniform_color([0, 0, 0])
search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=40)
target_pcd.estimate_normals(search_param)

# load initial agent observations
agent_pcd = load_agent_pcd(id,0)
agent_pcd.paint_uniform_color([0, 0.6, 1])

# load initial agent map
agent_map = load_agent_map(id,0)
agent_map.paint_uniform_color([0.1, 0.2, 0.1])

# load initial agents cubes
Agent_Geometries = []
for a in range(N):
    agent_box = o3d.geometry.TriangleMesh.create_box(
        width=CUBESAT_SIZE_M,
        height=CUBESAT_SIZE_M,
        depth=CUBESAT_SIZE_M,
    )
    agent_box.compute_vertex_normals()

    if a == (id-1):
        agent_box.paint_uniform_color([0, 0, 1])
    else:
        agent_box.paint_uniform_color([0.9, 0.4, 0.4])

    pos = Agents_History[0][a]['State'][:3]
    dq = Agents_History[0][a]['State'][6:10]
    lt.o3d_update_geom(agent_box, pos, dq)
    Agent_Geometries.append(agent_box)

################################################
################################################

# Set up the visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()        
vis.get_view_control().convert_from_pinhole_camera_parameters(
    o3d.io.read_pinhole_camera_parameters("Targets/camera_params.json"))

vis.add_geometry(grf)
vis.add_geometry(target_pcd)
vis.add_geometry(agent_pcd)
vis.add_geometry(agent_map)
for a in range(N):
    vis.add_geometry(Agent_Geometries[a])

# Register the animation callback
vis.register_animation_callback(custom_animation_callback)

# Run the visualizer
vis.run()
vis.destroy_window()


print('\n')
print('ANIMATION FINISHED !')
print('\n')