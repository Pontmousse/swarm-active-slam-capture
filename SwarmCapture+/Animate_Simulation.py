import numpy as np
import pybullet as p
import os
import pandas as pd
import math as m
import random
import time
import pybullet_data
import Observe_Target as rcl
import Spacecraft_Swarm as ss
import Neighborhood as nb
import Controllers as C
import Load_Target as lt
import Plot_Telemetry_Func as Telemetry
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import shared_config
CUBESAT_SIZE_M = float(shared_config.VIS_CUBESAT_SIZE_M)
CUBE_OBJ_BASE_SIZE_M = 0.2
CUBE_MESH_SCALE = CUBESAT_SIZE_M / CUBE_OBJ_BASE_SIZE_M

#############################################################################################
#############################################################################################
#############################################################################################
DT = shared_config.DT
dt = 1/DT # time step
N = shared_config.N # Number of Agents_Bodies
D = shared_config.D  # Simulation duration
object_name = shared_config.object_name # Target selected
#############################################################################################

paths = shared_config.get_sim_data_paths(n=N, d=D, dt=DT, name=object_name)
tag = paths["tag"]
path_agents = paths["agents"]
path_target = paths["target"]
path_attachment_points = paths["attachment_points"]

print('\nLoading agents history pickle file...')
Agents_History = Telemetry.load_variable_from_file(path_agents)
print('Loaded successfully')

print('\nLoading target history pickle file...')
Target_History = Telemetry.load_variable_from_file(path_target)
print('Loaded successfully')

print('\nLoading Excel file...')
input_file = paths["excel"]
if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"Missing simulation excel file: {input_file}\n"
        "Run the simulation first, or update shared_config.py to the dataset tag you want to animate."
    )
df = pd.read_excel(input_file)
print('Loaded successfully')

print('\nLoading attachment points history pickle file...')
attachment_points = Telemetry.load_variable_from_file(path_attachment_points)
print('Loaded successfully')

#############################################################################################
# Parameters

frac = 10 # Animation speed
stop = 1 # To animate only the first (total duration/stop) iterations of simulation

viz_target_pcd = False # Vizualize Target pcd or not
voxel = 3 # voxel size for downsampling of pcd for vizualization

viz_agent_pcd = False # Vizualize agent id landmarks
id = 1
ds = 5 # Downsample the landmarks to show by ds

#############################################################################################
#############################################################################################
#############################################################################################
#%% Connect to PyBullet and set up the simulation
physicsClient = p.connect(p.GUI, options="--mp4=swarm_capture.mp4")
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)

############################### Hyperparameters ##############################
Rcol = 5  # Collision radius
Rant = 40  # Antiflocking radius
Rdet = 10  # Detection radius
Rflk = 2 # DIstance to keep from landmarks
Rcom = 60  # Communication radius
delay = 1.5  # s - this is the frequency of communication. 0 for perfect comms, no delay.
Period = 80  # s - this is the period to wait before starting to chase an assigned target.


########################################################################################################################
# Set Camera view for GUI
camera_distance = 15  # Distance from the camera to the target point
camera_yaw = -24  # Yaw angle in degrees
camera_pitch = -31.0  # Pitch angle in degrees
camera_target_position = [-7, -1, 0]  # position that the camera is looking from
p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)


########################################################################################################################
# Generate the target
########################################################################################################################

target_position = Target_History[0][:3]
target_orientation = Target_History[0][6:10]
texTar_id = p.loadTexture("Targets/Texture_Target.jpg")
target_body_id, target_pcd, _ = lt.load_target(target_position,target_orientation,texTar_id)

# ########################################################################################################################
# Set target velocity
# p.resetBaseVelocity(target_body_id, [-15.2, 3.1, -4.5], [-60.9, -1.2, -20.4]) # Very Fast
# p.resetBaseVelocity(target_body_id, [-2.2, 2.1, -1.5], [2.9, -1.2, -2.4]) # Fast
# p.resetBaseVelocity(target_body_id, [-0.2, 0.8, -0.25], [1.1, -1, -0.8]) # Slow
p.resetBaseVelocity(target_body_id, [-0.05, 0.05, -0.05], [0.05, -0.03, 0.03]) # Very Slow
# p.resetBaseVelocity(target_body_id, [0, 0, 0], [0, 0, 0]) # Fixed




########################################################################################################################
# Generate the agents
########################################################################################################################

########################################################################################################################
# Create a agent's collision and visual shapes (same for all Agents_Bodies)
obj_file = "Cube_Blender/Cube.obj" # Cube is 20 cm x 20 cm x 20 cm (8U CubeSat)
agent_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                      fileName=obj_file,
                                      visualFramePosition=[0, 0, 0],
                                      meshScale=[CUBE_MESH_SCALE, CUBE_MESH_SCALE, CUBE_MESH_SCALE])
agent_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                            fileName=obj_file,
                                            collisionFramePosition=[0, 0, 0],
                                            meshScale=[CUBE_MESH_SCALE, CUBE_MESH_SCALE, CUBE_MESH_SCALE])
# Load the texture image
texture_file = "Cube_Blender/Texture_Cube.png"
TexCub_id = p.loadTexture(texture_file)

#######################################################################################################################
# Create the state of the agent
Agents_Bodies = []
Agents = []
for a in range(N):
    agent_body_id = p.createMultiBody(baseCollisionShapeIndex=agent_collision_shape_id, baseVisualShapeIndex=agent_visual_shape_id)
    Agents_Bodies.append(agent_body_id)


    agent_state = {
            "ID": a, # ID
            "TimeStep": 0,
            "State": Agents_History[0][a]['State'], # X = [r,v,q,w]
            "CommSet": [],
            "CollSet": [],
            "LandSet": [],
            "AntFlkSet": [],
            "DockSet": [],
            "LC": [],
            "LCD": [], # Unit vector showing the direction of Landmark Centroid in local frame
            "LCD_Frame": [], # Three orthonormed vectors forming a reference frame to align attitude
            "Mode": 's', # Begin in search mode
            "Mass": 4, # All satellites start with a 10 kg mass
            "Inertia": [1,1,1], # All satellite start are cubes with equal diagonal inertias
            "Target": [],
            "Odometry": [],
            "ChaseStability": [float('inf'), float('inf')], # Default Start [Initial Final]
        }
    Agents.append(agent_state)


    ########################################################################################################################
    # Create Agent body variable in simulation
    p.changeVisualShape(agent_body_id, -1, rgbaColor=[1, 1, 1, 1])
    p.changeVisualShape(agent_body_id, -1, textureUniqueId=TexCub_id)


    ########################################################################################################################
    # Set Agents_Bodies mass and state in simulation

    # Change mass of agent
    p.changeDynamics(agent_body_id,-1,mass=Agents[a]["Mass"])
    p.changeDynamics(agent_body_id,-1,localInertiaDiagonal=Agents[a]["Inertia"])
    # Define and Extract Initial agent state
    X = Agents[a]["State"]
    # rotate agent by 90 degrees around z axis
    p.resetBasePositionAndOrientation(agent_body_id, X[0:3], X[6:10])
    # reset agent velocity
    p.resetBaseVelocity(agent_body_id, X[3:6], X[10:13])


########################################################################################################################
########################################################################################################################
########################################################################################################################

def StripAndSplitStr(s):

    # Strip the parentheses
    s = s.strip('()')

    # Split the string by commas and convert each element to float
    numbers = [float(i) for i in s.split(",")]

    return numbers

########################################################################################################################
########################################################################################################################
########################################################################################################################


N += 1 # Add number of bodies making up the target
num_iter = len(Agents_History)/stop
duration = dt*num_iter

# Debug
# print(num_iter)


iter = int(m.floor(num_iter/frac))

for i in range(iter):
    j = frac*i
    p.removeAllUserDebugItems()
    
    
    
    cond_viz = False
    kappa = 0.5
    every = np.floor(kappa/dt)
    if i % every == 0: 
        cond_viz = True
        
    
    
    if cond_viz:
        for attachment_point in attachment_points:                    
            idx = attachment_point.idx          
            
            pos = attachment_point.position[j] + 0.05 * attachment_point.normal[j]
            p.addUserDebugText(".", pos, textColorRGB=[0.7, 0.4, 1], textSize=5)
            
            pos = attachment_point.position[j] + attachment_point.normal[j]
            p.addUserDebugText(str(idx), pos, textColorRGB=[0.7, 0.4, 1], textSize=1)


    for body_id in range(p.getNumBodies()):        
        k = (N*j)+body_id

        # Get the position and orientation data for the current body from the DataFrame
        body_data = df.iloc[k]
        position = StripAndSplitStr(body_data['positions'])
        orientation = StripAndSplitStr(body_data['orientations'])

        if body_id == 0: # If body id is 0, draw and update pos and orientation of target
            dq = p.getDifferenceQuaternion(Target_History[i-1][6:10],Target_History[i][6:10])
            lt.o3d_update_geom(target_pcd, position, dq)
            if viz_target_pcd and cond_viz:
                lt.plot_target_pybullet(target_pcd, voxel)
                
                
        if viz_agent_pcd and body_id == id and cond_viz:
            Landm = Agents_History[j][id-1]['LandSet']
            num = len(Landm) // ds
            Landm = random.sample(Landm, num)
            for q in range(len(Landm)):
                point = Landm[q]
                p.addUserDebugText(".", point, textColorRGB=[0, 1, 1], textSize=2)
        
    
        
        # Reset the position and orientation of the current body in the simulation
        p.resetBasePositionAndOrientation(body_id, position, orientation)
        
    p.addUserDebugText(f"Time step: {j*dt}", [5, 5, 10], textSize = 1)  # display the time step
    time.sleep(dt/10)
    

# Disconnect from the simulation
p.disconnect()

print('\n')
print('ANIMATION FINISHED !')
print('\n')
