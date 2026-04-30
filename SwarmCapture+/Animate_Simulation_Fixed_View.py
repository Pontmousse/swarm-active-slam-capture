import numpy as np
import pybullet as p
import os
import pandas as pd
import math as m
import time
import pybullet_data
import Ray_Cast_Lidar as rcl
import Spacecraft_Swarm as ss
import Neighborhood as nb
import Controllers as C
import Load_Target as lt
import Plot_Telemetry_Func as Telemetry
from PIL import Image
from moviepy.editor import ImageSequenceClip
import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import shared_config
MODULE_DIR = Path(__file__).resolve().parent
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

print('\nLoading agents history pickle file...')
path_agents = paths["agents"]
Agents_History = Telemetry.load_variable_from_file(path_agents)
print('Loaded successfully')


print('\nLoading target history pickle file...')
path_target = paths["target"]
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
path_attachment_points = paths["attachment_points"]
attachment_points = Telemetry.load_variable_from_file(path_attachment_points)
print('Loaded successfully')



dt = 1/240 # time step
frac = 20 # Animation speed
slow = dt*1/10 # higher means slower
stop = 1 # To animate only 1/1th of the total duration of simulation

#############################################################################################
#############################################################################################
#############################################################################################
#%%

# Connect to PyBullet and set up the simulation
# physicsClient = p.connect(p.GUI, options="--mp4=swarm_capture.mp4")
# physicsClient = p.connect(p.GUI)
physicsClient = p.connect(p.DIRECT)


p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)


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
# Lidar sensor parameters
num_rays_theta = 25  # Number of rays in theta direction (horizontal)
num_rays_phi = 25  # Number of rays in phi direction (vertical)
max_distance = Rdet  # The maximum distance of the raycast
visualize_rays = False  # Whether or not to visualize the raycasts
visualize_hits = False  # Whether or not to visualize the raycast hits
visualize_fraction = 0.1  # Fraction of total rays to visualize
lidar_fov = np.pi/4  # Field of view (FOV) of the Lidar in radians (45 degrees in this case)



########################################################################################################################
# Set Camera view for GUI
camera_distance = 20  # Distance from the camera to the target point
camera_yaw = -40  # Yaw angle in degrees
camera_pitch = 51.0  # Pitch angle in degrees
camera_target_position = [0, 0, 0]  # position that the camera is looking at
p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)


# Set Camera view for video record
width, height = 1920, 1080
view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition = camera_target_position,
                                                  distance = camera_distance,
                                                  yaw = camera_yaw,
                                                  pitch = camera_pitch,
                                                  roll = 0,
                                                  upAxisIndex = 2)
proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                           aspect=float(width) / height,
                                           nearVal=0.1,
                                           farVal=100.0)

########################################################################################################################
# Generate the target
########################################################################################################################

target_position = [0, 0, 0]
target_orientation = p.getQuaternionFromEuler([0, 200, 0])
texTar_id = p.loadTexture(str(MODULE_DIR / "Targets" / "Texture_Target.jpg"))
target_body_id, target_pcd, _ = lt.load_target(target_position,target_orientation,texTar_id)

########################################################################################################################
# Set target velocity
# p.resetBaseVelocity(target_body_id, [-1.2, 0.1, 1.5], [4.2, -2.6, -1])
p.resetBaseVelocity(target_body_id, [-0.2, 0.1, 0], [0.1, -1, -0.3])
# p.resetBaseVelocity(target_body_id, [0.1, -0.05, 0], [0.01, -0.1, -0.05])




########################################################################################################################
# Generate the agents
########################################################################################################################

########################################################################################################################
# Create a agent's collision and visual shapes (same for all Agents_Bodies)
obj_file = str(MODULE_DIR / "Cube_Blender" / "Cube.obj") # Cube is 20 cm x 20 cm x 20 cm (8U CubeSat)
agent_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                      fileName=obj_file,
                                      visualFramePosition=[0, 0, 0],
                                      meshScale=[CUBE_MESH_SCALE, CUBE_MESH_SCALE, CUBE_MESH_SCALE])
agent_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                            fileName=obj_file,
                                            collisionFramePosition=[0, 0, 0],
                                            meshScale=[CUBE_MESH_SCALE, CUBE_MESH_SCALE, CUBE_MESH_SCALE])
# Load the texture image
texture_file = str(MODULE_DIR / "Cube_Blender" / "Texture_Cube.png")
TexCub_id = p.loadTexture(texture_file)

#######################################################################################################################
# Create the state of the agent
Agents_Bodies = []
Agents = []
for a in range(N):
    agent_body_id = p.createMultiBody(baseCollisionShapeIndex=agent_collision_shape_id, baseVisualShapeIndex=agent_visual_shape_id)
    Agents_Bodies.append(agent_body_id)


    agent_state = {
            "ID": a, # ID 'b' is for the target bodies
            "TimeStep": 0,
            "State": ss.Random_Agents(Agents), # X = [r,v,q,w]
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

    path = str(MODULE_DIR / "Movie" / f"frame_{i}.png")
    image = Image.fromarray(rgb, 'RGB')
    image.save(path)

########################################################################################################################
########################################################################################################################
########################################################################################################################


N += 1 # Add number of bodies making up the target
num_iter = int((len(df))/N)/stop
duration = dt*num_iter

# Debug
# print(num_iter)


iter = int(m.floor(num_iter/frac))

for i in range(iter):
    p.removeAllUserDebugItems()

    for body_id in range(p.getNumBodies()):
        j = frac*i
        k = (N*j)+body_id

        # Get the position and orientation data for the current body from the DataFrame
        body_data = df.iloc[k]
        position = StripAndSplitStr(body_data['positions'])
        orientation = StripAndSplitStr(body_data['orientations'])

        # Reset the position and orientation of the current body in the simulation
        p.resetBasePositionAndOrientation(body_id, position, orientation)

        
    p.stepSimulation()

    save_image(width, height, view_matrix, proj_matrix, i)

    p.addUserDebugText(f"Time step: {j*dt}", [5, 5, 0.5], textSize = 1)  # display the time step
    time.sleep(dt/10)

    time.sleep(slow)
    print(f'Animation / Simulation Step {i} / {int(iter-1)}')


##############################################################################
# Disconnect from the simulation
p.disconnect()



##############################################################################
# Record fancy movie of the animation

# Define the path to your images and the output video
image_folder = str(MODULE_DIR / "Movie")  # Path to the folder containing your images
output_video = str(MODULE_DIR / "swarm_capture.mp4")  # Output video file

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
print(f"Deleting frame images from {image_folder}\n...")
# Delete all frame images in the frame_folder
frame_images = glob.glob(f"{image_folder}/frame_*.png")  # Get all frame images in the folder
for image in frame_images:
    os.remove(image)  # Remove each image file
print(f"Frame images deleted.")

print('\n')
print('ANIMATION FINISHED AND MOVIE RECORDED !')
print('\n')