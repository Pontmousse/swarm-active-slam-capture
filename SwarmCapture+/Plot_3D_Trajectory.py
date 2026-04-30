import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import Plot_Telemetry_Func as Telemetry
import trimesh
import pyvista as pv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import shared_config

# Object Selection
object_name = shared_config.object_name

# Load Simulation History
paths = shared_config.get_sim_data_paths()
Agents_History = Telemetry.load_variable_from_file(paths["agents"])
dt = 1 / shared_config.DT  # time step
N = len(Agents_History[0])  # Number of agents
num_iter = len(Agents_History)

# Downsample History to speed up animation
step_size = 10
Agents_History = [Agents_History[i] for i in range(10, num_iter, step_size)]
num_iter = len(Agents_History)

############################################################################
#%% Prepare plot data and STL file

Time = []
Agents_Position = [[] for _ in range(N)]
Agents_Orientation = [[] for _ in range(N)]

# Extract positions and orientations for each agent
for i in range(num_iter):
    Time.append(i*dt)
    Agents = Agents_History[i]
    for a in range(N):
        Spacecraft = Agents[a]
        Agents_Position[a].append(Spacecraft['State'][:3])  # Position (x, y, z)
        Agents_Orientation[a].append(Spacecraft['LCD'])  # Orientation vector (LCD)



# Load the STL file using trimesh
mesh = trimesh.load_mesh('Targets/'+object_name+'/'+object_name+'.stl')

# Extract the vertices and faces
vertices = mesh.vertices
faces = mesh.faces

############################################################################
#%% Plot trajectory

# Create the 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory of each agent
for a in range(N):
    positions = np.array(Agents_Position[a])
    orientations = np.array(Agents_Orientation[a])

    # Plot the trajectory of the agent
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=f'Agent {a+1}')

    # Plot orientation vectors along the trajectory
    for i in range(int(len(positions)/10)):
        pos = positions[i*10]
        orient = orientations[i*10]
        
        # Scale the orientation vector for visibility
        ax.quiver(pos[0], pos[1], pos[2], orient[0], orient[1], orient[2], length=1, normalize=True, color='r', alpha=0.6)


# Loop through faces of the STL and plot the triangles with transparency
# for face in faces:
#     v0, v1, v2 = vertices[face]
#     ax.plot_trisurf([v0[0], v1[0], v2[0]],
#                     [v0[1], v1[1], v2[1]],
#                     [v0[2], v1[2], v2[2]],
#                     color='cyan', edgecolor='black', alpha=0.4)  # Adjust alpha for transparency


# Customize the plot
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('3D Trajectories and Orientations of Agents')
ax.legend()

# Show the plot
plt.show()
