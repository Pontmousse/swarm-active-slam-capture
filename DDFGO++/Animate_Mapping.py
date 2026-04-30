import pickle
import matplotlib.pyplot as plt
import numpy as np
import Plot_Telemetry_Func as Telemetry
import open3d as o3d
import Load_Target as lt
import pybullet as p
import pybullet_data
import time
import config

# Connect to PyBullet and set up the simulation
physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)

#############################################################################################
# Load configuration from config.py
#############################################################################################
DT = config.DT
N = config.N
D = config.D
object_name = config.object_name
Kinem = config.Kinem

# Get file paths from config
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

# Downsample History to speed up animation
num_iter = len(Agents_History)
step_size = config.animation_downsample_step
Agents_History = [Agents_History[i] for i in range(0, num_iter, step_size)]
Target_History = [Target_History[i] for i in range(0, num_iter, step_size)]



id = 1 # Agent ID to show
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

def _split_map_by_owner(spacecraft, owner_id):
    """
    Split MapSet into (own_points, neighbor_points) using MapNghSet if present.
    owner_id is 1-indexed to match stored neighbor IDs.
    """
    pts = _to_points_array(spacecraft.get('MapSet', []))
    ngh = spacecraft.get('MapNghSet', None)
    if ngh is None or len(pts) == 0:
        return pts, np.array([]).reshape(0, 3)
    ngh = np.asarray(ngh).reshape(-1)
    if len(ngh) != len(pts):
        # Fallback: cannot reliably split.
        return pts, np.array([]).reshape(0, 3)
    own_mask = (ngh == owner_id)
    return pts[own_mask], pts[~own_mask]

def _spheres_from_points(points, radius, resolution):
    """
    Build a single TriangleMesh containing a sphere per point.
    Intended for small landmark counts (e.g., sparse maps).
    """
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

def load_agent_pcd(id,i):
    Agents = Agents_History[i]
    Spacecraft = Agents[id-1]
    Landm = Spacecraft.get('FeatureSet', np.array([]).reshape(0, 3))
    Landm = _to_points_array(Landm)
    agent_pcd = o3d.geometry.PointCloud()
    agent_pcd.points = o3d.utility.Vector3dVector(Landm)
    return agent_pcd

def load_agent_map(id,i):
    Agents = Agents_History[i]
    Spacecraft = Agents[id-1]
    Landm = Spacecraft.get('MapSet', np.array([]).reshape(0, 3))
    Landm = _to_points_array(Landm)
    agent_map = o3d.geometry.PointCloud()
    agent_map.points = o3d.utility.Vector3dVector(Landm)
    return agent_map

def load_agent_merged_map(id,i):
    Agents = Agents_History[i]
    Spacecraft = Agents[id-1]
    if 'MergedMapSet' in Spacecraft:
        Landm = Spacecraft['MergedMapSet']
        Landm = _to_points_array(Landm)
    else:
        Landm = np.array([]).reshape(0, 3)
    agent_merged_map = o3d.geometry.PointCloud()
    agent_merged_map.points = o3d.utility.Vector3dVector(Landm)
    return agent_merged_map

##############################################################################################################################
# Custom Animation Callback
##############################################################################################################################

def custom_animation_callback(vis):
    # Sleep run to slow down animation
    seconds = 0.12
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
    agent_pcd.paint_uniform_color([0, 1, 1])
    print('Number of current landmarks observed:  '+str(len(agent_pcd.points)))
    # print('Mean of current landmarks observed:  '+str(np.mean(agent_pcd.points, axis=0)[0]))
    
    # Agent Landmark Map
    spacecraft = Agents_History[k][id-1]
    own_pts, ngh_pts = _split_map_by_owner(spacecraft, owner_id=id)

    # Replace sphere meshes each frame (robust across Open3D versions).
    vis.remove_geometry(geom_state['agent_map_own_spheres'], reset_bounding_box=False)
    vis.remove_geometry(geom_state['agent_map_ngh_spheres'], reset_bounding_box=False)

    geom_state['agent_map_own_spheres'] = _spheres_from_points(
        own_pts,
        radius=config.animation_sparse_sphere_radius,
        resolution=config.animation_sparse_sphere_resolution,
    )
    geom_state['agent_map_own_spheres'].paint_uniform_color([1.0, 0.35, 0.1])

    geom_state['agent_map_ngh_spheres'] = _spheres_from_points(
        ngh_pts,
        radius=max(0.5 * config.animation_sparse_sphere_radius, 1e-6),
        resolution=config.animation_sparse_sphere_resolution,
    )
    geom_state['agent_map_ngh_spheres'].paint_uniform_color([0.3, 0.6, 1.0])

    vis.add_geometry(geom_state['agent_map_own_spheres'], reset_bounding_box=False)
    vis.add_geometry(geom_state['agent_map_ngh_spheres'], reset_bounding_box=False)
    print('Number of own map landmarks:  '+str(len(own_pts)))
    print('Number of neighbor/shared map landmarks:  '+str(len(ngh_pts)))
    # print('Geometric center of landmarks map:  '+str(np.mean(agent_map.points, axis=0)[0]))
    
    # Agent Merged Dense Map
    new_agent_merged_map = load_agent_merged_map(id,k)
    dense_pts = np.asarray(new_agent_merged_map.points)
    if dense_pts.size > 0 and config.animation_dense_voxel_size and config.animation_dense_voxel_size > 0:
        tmp = o3d.geometry.PointCloud()
        tmp.points = o3d.utility.Vector3dVector(dense_pts)
        tmp = tmp.voxel_down_sample(voxel_size=float(config.animation_dense_voxel_size))
        agent_merged_map.points = tmp.points
    else:
        agent_merged_map.points = new_agent_merged_map.points
    agent_merged_map.paint_uniform_color([0.55, 0.55, 0.85])  # subdued violet
    print('Number of merged map points (viz):  '+str(len(agent_merged_map.points)))

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
    # (sphere meshes already refreshed above)
    vis.update_geometry(agent_merged_map)
    vis.update_geometry(agent_pcd)
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
target_pcd = o3d.geometry.PointCloud()
Target_Point_Cloud_History = Telemetry.load_variable_from_file(path_target_pcd)
target_pcd.points = o3d.utility.Vector3dVector(Target_Point_Cloud_History[0])
target_pcd.paint_uniform_color([0, 0, 0])

# load initial agent observations
agent_pcd = load_agent_pcd(id,0)
agent_pcd.paint_uniform_color([0, 0.1, 1])

# load initial agent map
spacecraft0 = Agents_History[0][id-1]
own0, ngh0 = _split_map_by_owner(spacecraft0, owner_id=id)
agent_map_own_spheres = _spheres_from_points(
    own0,
    radius=config.animation_sparse_sphere_radius,
    resolution=config.animation_sparse_sphere_resolution,
)
agent_map_own_spheres.paint_uniform_color([1.0, 0.35, 0.1])
agent_map_ngh_spheres = _spheres_from_points(
    ngh0,
    radius=max(0.5 * config.animation_sparse_sphere_radius, 1e-6),
    resolution=config.animation_sparse_sphere_resolution,
)
agent_map_ngh_spheres.paint_uniform_color([0.3, 0.6, 1.0])

# Mutable geometry references for callback updates
geom_state = {
    'agent_map_own_spheres': agent_map_own_spheres,
    'agent_map_ngh_spheres': agent_map_ngh_spheres,
}

# load initial agent merged map
agent_merged_map = load_agent_merged_map(id,0)
if config.animation_dense_voxel_size and config.animation_dense_voxel_size > 0:
    agent_merged_map = agent_merged_map.voxel_down_sample(voxel_size=float(config.animation_dense_voxel_size))
agent_merged_map.paint_uniform_color([0.55, 0.55, 0.85])  # subdued violet

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
        agent_box.paint_uniform_color([0.15, 0.05, 0.15])
    else:
        agent_box.paint_uniform_color([0.4, 0.2, 0])

    pos = Agents_History[0][a]['State'][:3]
    dq = Agents_History[0][a]['State'][6:10]
    lt.o3d_update_geom(agent_box, pos, dq)
    Agent_Geometries.append(agent_box)

################################################
################################################

# Set up the visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()        
camera_params_path = config.MODULE_DIR / "camera_params.json"
vis.get_view_control().convert_from_pinhole_camera_parameters(
    o3d.io.read_pinhole_camera_parameters(str(camera_params_path)))

opt = vis.get_render_option()
if config.animation_dark_background:
    opt.background_color = np.array(config.animation_background_rgba[:3], dtype=np.float64)
opt.point_size = float(config.animation_dense_point_size)

vis.add_geometry(grf)
vis.add_geometry(target_pcd)
vis.add_geometry(geom_state['agent_map_own_spheres'])
vis.add_geometry(geom_state['agent_map_ngh_spheres'])
vis.add_geometry(agent_merged_map)
vis.add_geometry(agent_pcd)
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