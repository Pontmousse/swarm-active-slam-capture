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
from PIL import Image
from moviepy.editor import ImageSequenceClip
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import shared_config
CUBESAT_SIZE_M = float(shared_config.VIS_CUBESAT_SIZE_M)
CUBE_OBJ_BASE_SIZE_M = 0.2
CUBE_MESH_SCALE = CUBESAT_SIZE_M / CUBE_OBJ_BASE_SIZE_M

# Connect to PyBullet and set up the simulation
# physicsClient = p.connect(p.GUI, options="--mp4=swarm_capture.mp4")
physicsClient = p.connect(p.DIRECT)
# physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)

#############################################################################################
#############################################################################################
#############################################################################################
DT = shared_config.DT
dt = 1/DT # time step
N = shared_config.N # Number of Agents_Bodies
D = shared_config.D # Simulation duration
object_name = shared_config.object_name # Target selected
#############################################################################################

paths = shared_config.get_sim_data_paths(n=N, d=D, dt=DT, name=object_name)
tag = paths["tag"]
path_agents = paths["agents"]
path_target = paths["target"]
Agents_History = Telemetry.load_variable_from_file(path_agents)
Target_History = Telemetry.load_variable_from_file(path_target)

input_file = paths["excel"]
if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"Missing simulation excel file: {input_file}\n"
        "Run the simulation first, or update shared_config.py to the dataset tag you want to animate."
    )
df = pd.read_excel(input_file)


#############################################################################################
# Parameters

frac = 2 # Animation speed
stop = 1 # To animate only the first (total duration/stop) iterations of simulation

viz_target_pcd = False # Vizualize Target pcd or not
voxel = 1.5 # voxel size for downsampling of pcd for vizualization

viz_agent_pcd = False # Vizualize agent id landmarks
id = 1 # body id of agent of interest
ds = 5 # Downsample the landmarks to show by ds
#############################################################################################
#############################################################################################
#############################################################################################

############################### Hyperparameters ##############################
xbox = 20
ybox = 20
zbox = 20
Rcol = 5  # Collision radius
Rant = 40  # Antiflocking radius
Rdet = 10  # Detection radius
Rflk = 2 # DIstance to keep from landmarks
Rcom = 60  # Communication radius
delay = 1.5  # s - this is the frequency of communication. 0 for perfect comms, no delay.
Period = 80  # s - this is the period to wait before starting to chase an assigned target.


########################################################################################################################
# Set Camera view for GUI
camera_distance = 13.5  # Distance from the camera to the target point
camera_yaw = 24  # Yaw angle in degrees
camera_pitch = -31.0  # Pitch angle in degrees
camera_target_position = [7, -1, 0]  # position that the camera is looking from
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

def save_image(width, height, view_matrix, proj_matrix,i):
 
    # Capture an image
    img_arr = p.getCameraImage(width, height, view_matrix, proj_matrix)
    w, h, rgb, depth, seg = img_arr

    # Convert RGB data to a format that PIL can handle
    rgb = np.reshape(rgb, (h, w, 4))  # Reshape to (height, width, RGBA)
    rgb = rgb[:, :, :3]  # Drop the alpha channel

    path = 'Movie/frame_'+str(i)+'.png'
    image = Image.fromarray(rgb, 'RGB')
    image.save(path)

########################################################################################################################
########################################################################################################################
########################################################################################################################


N += 1 # Add number of bodies making up the target
num_iter = int((len(df))/N)/stop
duration = dt*num_iter

iter = int(m.floor(num_iter/frac))

for i in range(iter):
    p.removeAllUserDebugItems()
    j = frac*i
    for body_id in range(p.getNumBodies()):
        k = (N*j)+body_id

        # Get the position and orientation data for the current body from the DataFrame
        body_data = df.iloc[k]
        position = StripAndSplitStr(body_data['positions'])
        orientation = StripAndSplitStr(body_data['orientations'])

        ##############################################
        if body_id == 0: # If body id is the target body
            # draw and update pos and orientation of target
            dq = p.getDifferenceQuaternion(Target_History[i-1][6:10],Target_History[i][6:10])
            lt.o3d_update_geom(target_pcd, position, dq)
            if viz_target_pcd:
                lt.plot_target_pybullet(target_pcd, voxel)

        ##############################################""
        cond_viz = False
        kappa = 0.03
        every = np.floor(kappa/dt)
        if i % every == 0: 
            cond_viz = True

        if viz_agent_pcd and body_id == id and cond_viz:
            Landm = Agents_History[j][id-1]['LandSet']
            num = len(Landm) // ds
            Landm = random.sample(Landm, num)
            for q in range(len(Landm)):
                point = Landm[q]
                p.addUserDebugText(".", point, textColorRGB=[0, 1, 1], textSize=2)

        # Reset the position and orientation of the current body in the simulation
        p.resetBasePositionAndOrientation(body_id, position, orientation)
    
    #######################################################################################
    # Extract state of agent of interest
    camera_position = Agents_History[j][id-1]['State'][0:3]
    camera_orientation = Agents_History[j][id-1]['State'][6:10]

    # Rotation of target orientation
    angle = m.pi / 2.0
    axis_of_rotation = [0, 0, -1]  # normalized axis of rotation
    target_quaternion = p.getQuaternionFromAxisAngle(axis_of_rotation, angle)
    camera_orientation = p.multiplyTransforms([0, 0, 0], camera_orientation, [0, 0, 0], target_quaternion)[1]
    
    rotation_matrix = np.array(p.getMatrixFromQuaternion(camera_orientation)).reshape(3, 3).T

    # The forward direction is the third column of the rotation matrix
    forward_direction = np.array([rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2]])
    up_vector = np.array([rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2]])

    # Define the distance at which the camera is "looking" from the agent's position
    look_distance = 5.0  # Adjust this as necessary

    # Calculate the camera target position based on the agent's forward direction
    camera_target_position = camera_position + look_distance * forward_direction
    camera_position += 0.2 * forward_direction

    # Moving Camera
    # Set Camera view for video record
    width, height = 1920, 1080
    proj_matrix = p.computeProjectionMatrixFOV(fov = 45,
                                           aspect=float(width) / height,
                                           nearVal=0.1,
                                           farVal=100.0)
    view_matrix = p.computeViewMatrix(
                        cameraEyePosition = camera_position,      # Camera (agent's) position
                        cameraTargetPosition = camera_target_position,  # Where the agent is looking
                        cameraUpVector = up_vector                        # Up vector assuming z-axis is up
                    )

    ## Debug ######################################
    p.addUserDebugLine(camera_position, camera_target_position, lineColorRGB=[1, 1, 1], lineWidth=3) # Line of Sight
    # C.plot_ort_rot(camera_position, np.array(rotation_matrix), 0.2)
    # print('\n')
    # print(np.array(view_matrix).reshape(4,4).T)
    # print('\n')
    # print(np.array(proj_matrix).reshape(4,4))
    # print('\n')
    ## Debug ######################################

    save_image(width, height, view_matrix, proj_matrix, i)

    p.addUserDebugText(f"Time step: {j*dt}", [5, 5, 0.5], textSize = 1)  # display the time step
    time.sleep(dt/10)
    print(f'Animation / Simulation Step {i} / {int(iter-1)}')
    

# Disconnect from the simulation
p.disconnect()

##############################################################################
# Record fancy movie of the animation

# Define the path to your images and the output video
image_folder = 'Movie'  # Path to the folder containing your images
output_video = 'swarm_capture.mp4'  # Output video file

# Define the list of images
files = os.listdir(image_folder)
count = len([name for name in files if os.path.isfile(os.path.join(image_folder, name))])
image_files = [f'{image_folder}/frame_{i}.png' for i in range(0, count-1)]

# Define the frame rate (frames per second)
fps = 30  # Modify this according to your needs

# Load the images into a clip
clip = ImageSequenceClip(image_files, fps=fps)

# Optionally, you can resize the video
# clip = clip.resize(newsize=(width, height))  # newsize can be (width, height) or a scale factor

# Write the clip to a file
clip.write_videofile(output_video, codec='libx264', preset='slow', bitrate='8000k')  # You can change the codec if needed

##############################################################################

print('\n')
print('ANIMATION FINISHED AND MOVIE RECORDED !')
print('\n')