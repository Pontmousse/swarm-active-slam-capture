import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import Plot_Telemetry_Func as Telemetry
import pyvista as pv
from scipy.spatial.transform import Rotation as R
import pybullet as p

#############################################################################################
#############################################################################################
DT = 240
dt = 1/DT # time step
N = 7 # Number of Agents_Bodies
D = 15 # Simulation duration
object_name = 'Motor' # Target selected
#############################################################################################

tag = 'N'+str(N)+'_D'+str(D)+'_dt'+str(DT)
tag = tag+'_'+object_name


path_agents = 'Data/Agents_History_'+tag+'.pkl'
path_target = 'Data/Target_History_'+tag+'.pkl'
Agents_History = Telemetry.load_variable_from_file(path_agents)
Target_History = Telemetry.load_variable_from_file(path_target)

# Downsample History to speed up animation
num_iter = len(Agents_History)
step_size = 3
Agents_History = [Agents_History[i] for i in range(10, num_iter, step_size)]
num_iter = len(Agents_History)

############################################################################
############################################################################
#%% Prepare plot data and STL file
axes = pv.Axes(show_actor=True, actor_scale=2.0, line_width=5)
axes.origin = (0.0, 0.0, 0.0)


Time = []
Agents_Position = [[] for _ in range(N)]
Agents_Orientation = [[] for _ in range(N)]

# Extract positions and orientations for each agent
for i in range(num_iter):
    Time.append(i*step_size*dt)
    Agents = Agents_History[i]
    for a in range(N):
        Spacecraft = Agents[a]
        Agents_Position[a].append(Spacecraft['State'][:3])  # Position (x, y, z)
        Agents_Orientation[a].append(Spacecraft['LCD'])  # Orientation vector (LCD)


# Load the obj file
mesh = pv.read('Targets/' + object_name + '/' + object_name + '.obj')

# Scale the mesh
scale_factor = 0.30
mesh = mesh.scale(scale_factor)

# Translate (position) the mesh in the 3D space
translation_vector = np.array(Target_History[0][:3]) 
mesh = mesh.translate(translation_vector)

# create mesh for final step and rotate it
mesh_final = mesh
quat = Target_History[-1][6:10]
rotation_axis, rotation_angle = p.getAxisAngleFromQuaternion(quat)
mesh_final = mesh_final.rotate_vector(vector=np.array(rotation_axis),
                                      angle=np.degrees(rotation_angle),
                                      point=axes.origin)

# Create a plotter
plotter = pv.Plotter()

# Add the reference frame (axes) to the plot
plotter.add_axes()

# Add the STL mesh to the plot
plotter.add_mesh(mesh, color='lightgray', opacity = 0.00)
plotter.add_mesh(mesh_final, color='lightblue', opacity = 0.95)



colors = Telemetry.generate_color_lists_rgb(N)
# Add point trajectories to the plot
for a in range(N):
    points = np.array(Agents_Position[a])
    point_cloud = pv.PolyData(points)
    plotter.add_points(point_cloud, color=colors[a], point_size=5)



#%% Show the plot

camera = pv.Camera()
camera.position = (30.0, 30.0, 30.0)
camera.focal_point = (5.0, 5.0, 5.0)

# Show the plot
plotter.add_text("Mesh", font_size=24)
plotter.add_actor(axes.actor)
plotter.camera = camera
plotter.show()