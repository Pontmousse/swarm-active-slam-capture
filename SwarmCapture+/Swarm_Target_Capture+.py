import numpy as np
import pybullet as p
import os
import pandas as pd
import math as m
import time
import random
from datetime import timedelta
import pybullet_data
import open3d as o3d
import Observe_Target as ot
import Spacecraft_Swarm as ss
import Neighborhood as nb
import Controllers as C
import Load_Target as lt
import Ray_Cast_Lidar as rcl
import copy
import pickle
import json

# Connect to PyBullet and set up the simulation
physicsClient = p.connect(p.DIRECT)
# physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)


########################## Tunable Hyperparameters ############################
# Instantiate the SimulationParameters class
sim_params = ss.SimulationParameters(
    AntFlk_Radius = 100,
    Flk_Radius = 4,
    
    Pointing_Proportional_Gain = 3,
    Pointing_Derivative_Gain = 0.5,
    
    Flk_Potential = 1,
    AntFlk_Potential = 7000,
    Encapsulate_Derivative_Gain = 200,
    
    Capture_Proportional_Gain = 5,
    Capture_Derivative_Gain = 100,
    Capture_Alignment_Gain = 5,
    
    Distance_bid_weight = 20,
    Velocity_bid_weight = 1,
    Normal_bid_weight = 1
)

Rant = sim_params.AntFlk_Radius
Rflk = sim_params.Flk_Radius

# thrust saturation
max_frc = 2
max_trq = 0.5

############################### Hyperparameters ##############################
N = 6 # Number of Agents_Bodies
duration = 150 # seconds of simulation

cancel_chw = False
altitude = 500 # in km
low_pass_filter_coeff_state = 0.1 # for filtering state estimation
low_pass_filter_coeff_control = 0.4 # for filtering control input command

Rcol = 5  # Collision radius
Rdet = 100  # Detection radius
Rdet_ap = Rflk*2 # Attachment Point Detection Radius
Rcom = 300  # Communication radius


delay = 0  # s - this is the frequency of communication. 0 for perfect comms, no delay.

IMU = True # If IMU is available (Allows better initialization of ICP)
std = 0 # (unit: m) Gaussian standard deviation for feature observation
max_lidar_pcd = 3 # (unit: m) Distance to check between URDF and PLY model for feature identification
max_angle_pcd = 0.3 # (unit: degrees) Angle that should be between normals of point cloud and line of sight
num_feat = 3 # Number of features to extract per observation instance


########################################################################################################################
# Set Camera view for GUI
camera_distance = 8.46  # Distance from the camera to the target point
camera_yaw = -50.0  # Yaw angle in degrees
camera_pitch = -25.0  # Pitch angle in degrees
camera_target_position = [-6.02, -3.66, 0.60]  # position that the camera is looking from
p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

########################################################################################################################
# Lidar sensor parameters
num_rays_theta = 20  # Number of rays in theta direction (horizontal)
num_rays_phi = 20  # Number of rays in phi direction (vertical)
max_distance = Rdet  # The maximum distance of the raycast
visualize_rays = False  # Whether or not to visualize the raycasts
visualize_hits = False  # Whether or not to visualize the raycast hits
visualize_fraction = 0.3  # Fraction of total rays to visualize
lidar_fov = np.pi/2  # Field of view (FOV) of the Lidar in radians

########################################################################################################################
# Point Cloud Vizualisation Parameters
viz_target_pcd = False # Vizualize Target pcd or not
voxel = 1 # voxel size for downsampling of pcd for vizualization
viz_agent_pcd = False # Vizualize the Landmark of agents
ds = 3 # Downsample the landmarks to show by ds

########################################################################################################################
# Generate the target
########################################################################################################################

target_position = [0, 0, 0]
target_orientation = p.getQuaternionFromEuler([0, 0, 0])
texTar_id = p.loadTexture("/home/elghali/Desktop/SwarmCapture+/Targets/Texture_Target.jpg")
target_body_id, target_pcd, object_name = lt.load_target(target_position,target_orientation,texTar_id)

Target_Point_Cloud_History = []

# Create name tag
dt = 1/240 # simulation time step
step = 1/dt
tag = f'_N{N:.0f}_D{duration:.0f}_dt{step:.0f}'
tag = tag+'_'+object_name

Target_Point_Cloud_History.append(np.array(target_pcd.points))

########################################################################################################################
# Set target velocity # fixed translational vel
# tar_vel, tar_angvel = [0.0, 0.0, 0.0], [-60.9, -1.2, -20.4] # Very Fast
# tar_vel, tar_angvel = [0.0, 0.0, 0.0], [2.9, -1.2, -2.4] # Fast
# tar_vel, tar_angvel = [0.0, 0.0, 0.0], [-0.03, 0.05, -0.1] # Faster than normal
tar_vel, tar_angvel = [0.0, 0.0, 0.0], [0.005, -0.01, 0.01] # Very slow for large satellite
# tar_vel, tar_angvel = [0, 0, 0], [0, 0, 0] # Fixed

p.resetBaseVelocity(target_body_id, tar_vel, tar_angvel)

pos, quat = p.getBasePositionAndOrientation(target_body_id)
vel, angvel = p.getBaseVelocity(target_body_id)
target_state = pos+vel+quat+angvel


########################################################################################################################
# Generate the attachment points
########################################################################################################################

attachment_points = lt.load_predefined_attachment_points(object_name, target_pcd)
# attachment_points = [] # No attachment points



# Add first iteration
for ap in attachment_points:
        ap.add_iteration(target_state, target_pcd)
        

########################################################################################################################
# Generate the agents
########################################################################################################################

agent_mass = 4 # kg
agent_size = 0.14 # meters # cubesat side ~3U cubesat with 15cm side
agent_inertia = 1/6*agent_mass*(agent_size)**2 #kg.m^2
thruster_Isp = 180 # 200s typical Isp for thrusters
g0 = 9.81 # Earth's gravitational coefficient

########################################################################################################################
# Create a agent's collision and visual shapes (same for all Agents_Bodies)
obj_file = "/home/elghali/Desktop/SwarmCapture+/Cube_Blender/Cube.obj" # Cube is 20 cm x 20 cm x 20 cm (8U CubeSat)
agent_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                      fileName=obj_file,
                                      visualFramePosition=[0, 0, 0],
                                      meshScale=[1, 1, 1])
agent_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                            fileName=obj_file,
                                            collisionFramePosition=[0, 0, 0],
                                            meshScale=[1, 1, 1])
# Load the texture image
texture_file = "/home/elghali/Desktop/SwarmCapture+/Cube_Blender/Texture_Cube.png"
TexCub_id = p.loadTexture(texture_file)

#######################################################################################################################
# Create the state of the agent
Agents_Bodies = []
Agents = []
for a in range(N):
    agent_body_id = p.createMultiBody(baseCollisionShapeIndex=agent_collision_shape_id, baseVisualShapeIndex=agent_visual_shape_id)
    Agents_Bodies.append(agent_body_id)
    
    Random_State = ss.Random_Agents(Agents) # X = [r,v,q,w]

    agent_state = {
            "ID": a,
            "TimeStep": 0,
            "Iteration": 0,

            "State": Random_State, 
            "Smooth_State": Random_State,
            
            "CommSet": [],
            "CollSet": [],
            "AntFlkSet": [],

            "LandSet": [],
            "FeatureSet": [],
            "FeatureIdxSet": [],
            "MapSet": [],
            "MapIdxSet": [],

            "LC": [],
            "LCD": [], # Unit vector showing the direction of Landmark Centroid in local frame
            
            # Three orthonormed vectors forming a reference frame to align attitude
            "Control_Frame": [],
            
            "Mode": 'e', # Begin in encapsulation phase
            "Mass": agent_mass, # All satellites start with a given mass in kg
            "Inertia": [agent_inertia, agent_inertia, agent_inertia], # All satellite start are cubes with equal diagonal inertias
            "Fuel_Consumed": 0,
            
            "CLH_Force": [],
            "Control_Force": [],
            "Smooth_Control_Force": [],
            
            "Control_Torque": [],
            "Smooth_Control_Torque": [],
            
            # APs: Attachement Points
            "APs": [],
            "APs_Bids": [],
            "Target": [],

            # Consensus time
            "ActionTime": random.sample(range(int(100*3*duration/10), int(100*4*duration/10)), 1)[0]/100,
            
            # For docking
            "DockConstraint": None, # Initially no constraint id is created for this specific body with the target
            "DockContactPoint": None, # Contact point at contact time in agent's world reference frame
            "DockPose": [], # stored dock agent pose relative to target at docking instant.
            "DockTime": None
            
        }
    Agents.append(agent_state)


    ########################################################################################################################
    # Create Agent body visual in simulation
    p.changeVisualShape(agent_body_id, -1, rgbaColor=[1, 1, 1, 1])
    p.changeVisualShape(agent_body_id, -1, textureUniqueId=TexCub_id)


    ########################################################################################################################
    # Set Agents_Bodies mass and state in simulation

    # Change mass of agent
    p.changeDynamics(agent_body_id,-1,mass=Agents[a]["Mass"])
    p.changeDynamics(agent_body_id,-1,localInertiaDiagonal=Agents[a]["Inertia"])
    p.changeDynamics(agent_body_id,-1, restitution=0,  lateralFriction=1,
                     rollingFriction=1, spinningFriction=1,
                     contactStiffness = 10, contactDamping = 500000)
    # Define and Extract Initial agent state
    X = Agents[a]["State"]
    # rotate agent by 90 degrees around z axis
    p.resetBasePositionAndOrientation(agent_body_id, X[0:3], X[6:10])
    # reset agent velocity
    p.resetBaseVelocity(agent_body_id, X[3:6], X[10:13])



########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
# Run the simulation for a few steps

# For saving in Excel file
positions = []
orientations = []

# Initialize Histories
Target_History = []
Target_History.append(target_state) # Add first iteration

Agents_History = []
Agents_History.append(Agents) # Add first iteration


p.setTimeStep(dt)
num_iter = m.floor(duration/dt)

# Record the start time for the entire loop
start_time = time.time()

for i in range(num_iter):
     
    rand_ord = np.random.permutation(N)

    # Enforce and Extract Target data at time step i
    p.resetBaseVelocity(target_body_id, tar_vel, tar_angvel)
    target_state = Target_History[i]

    # Calculate Target KDTree for this time iteration
    target_kdtree = o3d.geometry.KDTreeFlann(target_pcd)
    
    # Clear the debug lines from the previous frame
    p.removeAllUserDebugItems()

    # visualize every kappa seconds
    cond_viz = False
    kappa = 0.1
    every = np.floor(kappa/dt)
    if i % every == 0: 
        cond_viz = True

    # Start loop over agents
    for a in range(N):

        ######################################### Extract Data from Previous Iteration for Feedback
        # Extract Agents data
        Agents = Agents_History[i]
        r = rand_ord[a]
        Spacecraft = Agents[r]
        agent_body_id = Agents_Bodies[r]


        ########## Scan Lidar
        ray_results = rcl.lidar_sensor(agent_body_id, [-0.25,0,0], num_rays_theta, num_rays_phi, max_distance, 
                visualize_rays, visualize_hits, visualize_fraction, cond_viz, lidar_fov)
        
        
        # Landm, Lc = ot.Landmarks_Detected(target_pcd, Spacecraft['State'][:3], Rdet)
        Landm, Lc = ot.Set_Landmarks_Detected(ray_results)
        Spacecraft["LandSet"] = Landm
        Spacecraft["LC"] = Lc

        # Indices = ot.Set_Features_Indices_Using_KDTree(Landm, target_pcd, target_kdtree, Spacecraft["State"], max_lidar_pcd, max_angle_pcd, Rflk*1.1)
        # Spacecraft["FeatureIdxSet"] = ot.downsample_features(Indices, num_feat)
        # Spacecraft["FeatureSet"] = ot.Observe_Features_from_Indices(target_pcd, Spacecraft["FeatureIdxSet"], std)


        if viz_agent_pcd and cond_viz:
            num = len(Landm) // ds
            Landm = random.sample(Landm, num)
            for q in range(len(Landm)):
                point = Landm[q]
                p.addUserDebugText(".", point, textColorRGB=[0, 0.2, 0.9], textSize=2)

        


        ######################################### Communicate with neighboring agents
        Lcd, Lc = nb.Set_Landmark_Centroid_Direction2(Spacecraft, Agents)
        Spacecraft["LCD"] = Lcd
        Spacecraft["LC"] = Lc
        # DebugLine Landmark Centroid Direction
        # nb.plot_LCD(Spacecraft)
        # print(Spacecraft["LC"])
        

        ######################################### Perceive local environment
        ########## Scan for neighbors
        Spacecraft["CommSet"] = nb.Set_Neighborhood(Spacecraft, Agents, Rcom)
        Spacecraft["CollSet"] = nb.Set_Neighborhood(Spacecraft, Agents, Rcol)
        Spacecraft["AntFlkSet"] = nb.Set_Neighborhood(Spacecraft, Agents, Rant)

        
        # Scan for Auction-Consensus        
        Spacecraft['APs'] = ot.Set_Attachment_Points_Detected(attachment_points, i,
                                                              Spacecraft['State'][:3],
                                                              Spacecraft['LCD'],
                                                              Rdet_ap)

        Spacecraft['APs_Bids'] = ss.calculate_bid(Spacecraft, attachment_points, i, sim_params)

        # Select Neighbour to querry through network
        if N > 1:
            querried_agent = random.choice([k for k in range(N) if k != r])
        else:
            querried_agent = None

        # Calculate readiness signal
        if len(Spacecraft['LC']) != 0:
            dis = np.linalg.norm(np.array(Spacecraft['State'][:3]) - np.array(Spacecraft['LC']))
            if N > 1 and dis < Rflk*1.5:
                neighbour_weight = 0.5/(N*duration)
                ngh_action_time = Agents_History[i][querried_agent]['ActionTime']
                Spacecraft["ActionTime"] += neighbour_weight * (ngh_action_time - Spacecraft['ActionTime'])
    
        ######################################### Check and Switch Mode and put constraint if necessary for docking mode
        mode_prev = Spacecraft["Mode"]
        if i > 1:
            mode, constraint, DockPose, contact_point = ss.Check_Mode_Switch(Spacecraft, Agents_History[i-1][r], agent_body_id, target_body_id, Rflk)
            Spacecraft['DockConstraint'] = constraint
            Spacecraft["Mode"] = mode
            Spacecraft["DockPose"] = DockPose
            if contact_point != (None, None, None):
                Spacecraft["DockContactPoint"] = np.array(contact_point) - Spacecraft['State'][:3]
        
        if mode_prev == 'c' and mode == 'd':
            Spacecraft["DockTime"] = i
        
        #########################################
        # Calculate Attitude Lock Reference Frame
        if Spacecraft["Mode"] == 'e':
            Spacecraft["Control_Frame"] = nb.Set_LCD_Frame(Spacecraft)
        elif Spacecraft["Mode"] == 'c':
            Spacecraft["Control_Frame"] = nb.Set_Target_dir_Frame(Spacecraft, attachment_points)
            
            
        # Debug        
        # if len(Spacecraft["Control_Frame"]) != 0:
            #DDebug
            # print('\n')
            # print(f'LCD_Frame:{Spacecraft["Control_Frame"]}')
            # print(f'LCD_Frame"][:,0]:{Spacecraft["Control_Frame"][:,0]}')
            # print(f'LCD_Frame"][0]{Spacecraft["Control_Frame"][0]}')
            # print(f'LCD_Frame"][:,1]{Spacecraft["Control_Frame"][:,1]}')
            # print(f'LCD_Frame"][1]{Spacecraft["Control_Frame"][1]}')

            # Q = C.matrix_to_quaternion(Spacecraft["Control_Frame"])
            # position = Spacecraft["State"][0:3]
            # C.plot_ort_quat(position,Q,0)
            # C.plot_ort_rot(position,Spacecraft["Control_Frame"],0.4)

        
        
        ######################################### Store and Save
        Spacecraft["TimeStep"] += dt
        Spacecraft['Iteration'] += 1
        Agents[r] = Spacecraft

        ########################################## Calculate Control Force and Torque
        u, tar = ss.Spacecraft_OBC(Spacecraft,
                                   attachment_points, i,
                                   Agents, cond_viz,
                                   querried_agent, sim_params)
        Spacecraft['Target'] = tar
        

        if Spacecraft["Mode"] != 'd':
            # force/torque history update
            control_force = u[0:3]
            Spacecraft['Control_Force'] = control_force
            # Debug forces
            # print('\n')
            # print(f'Control forces         : {control_force}')
            if i == 0:
                control_force = ss.saturate(control_force, max_frc)
                Spacecraft['Smooth_Control_Force'] = control_force
            else:
                if Spacecraft["Mode"] == "e":
                    control_force = C.smooth_force(Spacecraft, Agents_History[i-1][r], low_pass_filter_coeff_control)
            
            control_force = ss.saturate(control_force, max_frc)
            Spacecraft['Smooth_Control_Force'] = control_force
            
            # chw force
            chw_force = C.chw_force(Spacecraft, altitude, cancel_chw)
            Spacecraft['CHW_Force'] = chw_force
            force = control_force + chw_force
            
            # Debug forces
            # print(f'Smoothed Control forces: {control_force}')
            
            
            # torque
            torque = u[3:7] # no disturbances
            Spacecraft['Control_Torque'] = torque
            if i == 0:
                torque = ss.saturate(torque, max_trq)
                Spacecraft['Smooth_Control_Torque'] = torque
            else:
                if Spacecraft["Mode"] == "e":
                    torque = C.smooth_torque(Spacecraft, Agents_History[i-1][r], low_pass_filter_coeff_control)
            torque = ss.saturate(torque, max_trq)
            Spacecraft['Smooth_Control_Torque'] = torque
            
            # apply force / torque to body
            p.applyExternalForce(agent_body_id, -1, force, Spacecraft["State"][0:3], p.WORLD_FRAME)
            p.applyExternalTorque(agent_body_id, -1, torque, p.WORLD_FRAME)
            
            
            
            
        else:
            DockPose = Spacecraft["DockPose"]
            
            # force/torque history update
            control_force = u[0:3]
            Spacecraft['Control_Force'] = control_force
            Spacecraft['Smooth_Control_Force'] = control_force
            chw_force = C.chw_force(Spacecraft, altitude, cancel_chw)
            Spacecraft['CHW_Force'] = chw_force
            torque = u[3:7] # no disturbances
            Spacecraft['Control_Torque'] = torque
            Spacecraft['Smooth_Control_Torque'] = torque
            

            # Position Update
            tar_pos, tar_quat = p.getBasePositionAndOrientation(target_body_id)
            tar_rot = np.array(p.getMatrixFromQuaternion(tar_quat)).reshape(3, 3).T
            rel_pos = np.dot(tar_rot.T, DockPose[:3])
            pos  = np.array(tar_pos) + rel_pos

            # Orientation Update
            age_rot = np.array(p.getMatrixFromQuaternion(DockPose[3:7])).reshape(3, 3)
            rel_rot = np.dot(tar_rot.T, age_rot)
            quat = C.matrix_to_quaternion(rel_rot)

            # Linear Velocity Update
            tar_vel, tar_angvel = p.getBaseVelocity(target_body_id)
            vel = tar_vel + np.cross(tar_angvel, rel_pos)

            # Reset Pose
            p.resetBasePositionAndOrientation(agent_body_id, pos, quat)
            p.resetBaseVelocity(agent_body_id, vel, tar_angvel)
        
        
        # Calculate Fuel Consumption
        force_thrust = np.linalg.norm(Spacecraft['Smooth_Control_Force'])*dt/(thruster_Isp*g0)
        torque_thrust = np.linalg.norm(Spacecraft['Smooth_Control_Torque'])*dt/(thruster_Isp*g0*(agent_size/2))
        Spacecraft["Fuel_Consumed"] += force_thrust + torque_thrust
        
        # save after maneuver
        Agents[r] = Spacecraft 
        
        

    p.stepSimulation()

    p.addUserDebugText(f"Time step: {i*dt}/{duration}", [5, 5, 10], textSize = 1)  # display the time step
    time.sleep(dt/10)


    ############################################### Save History
    #################### Save Agents
    for a in range(N):
        r = rand_ord[a]
        Spacecraft = Agents[r]
        agent_body_id = Agents_Bodies[r]

        pos, quat = p.getBasePositionAndOrientation(agent_body_id)
        vel, angvel = p.getBaseVelocity(agent_body_id)

        Spacecraft["State"] = pos+vel+quat+angvel
        Spacecraft["Smooth_State"] = C.smooth_state(Spacecraft,
                                                    Agents_History[i-1][r],
                                                    low_pass_filter_coeff_state)
        
        Agents[r] = Spacecraft

    Agents_History.append(copy.deepcopy(Agents))


    #################### Save Target
    pos, quat = p.getBasePositionAndOrientation(target_body_id)
    vel, angvel = p.getBaseVelocity(target_body_id)
    dq = p.getDifferenceQuaternion(target_state[6:10], quat)
    target_state = pos+vel+quat+angvel
    Target_History.append(target_state)

    if viz_target_pcd:
        lt.plot_target_pybullet(target_pcd, voxel)

    # Update Target point cloud locations for this time step
    lt.o3d_update_geom(target_pcd, pos, dq)
    Target_Point_Cloud_History.append(np.array(target_pcd.points))

    # update attachment points positions and velocities
    for ap in attachment_points:
        ap.add_iteration(target_state, target_pcd)

    #####################################################################################################
    # Calculate the elapsed time for this iteration
    elapsed_time = time.time() - start_time
    
    # Calculate the average time per iteration
    avg_time_per_iteration = elapsed_time / (i + 1)

    # Calculate the estimated time left for the remaining iterations
    iterations_left = num_iter - i - 1
    estimated_time_left = avg_time_per_iteration * iterations_left

    # Convert the estimated time left to a human-readable format
    step = i*dt
    estimated_time_left_str = str(timedelta(seconds=estimated_time_left))
    if cond_viz: print(f"Time step: {step:.5f}/{duration}   seconds - Estimated time left: {estimated_time_left_str}")


    # Saving in Excel File
    for body_id in range(p.getNumBodies()):
        position, orientation = p.getBasePositionAndOrientation(body_id)
        positions.append(position)
        orientations.append(orientation)

    # save every kappa seconds
    cond_save = False
    kappa = 5
    every = np.floor(kappa/dt)
    if i % every == 0: 
        cond_save = True

    if cond_save:
        ########################################################################################################################
        # Excel data saving
        ########################################################################################################################

        print('Saving Excel Data...')
        # Combine all the data into a dictionary
        data = {
            "positions": positions,
            "orientations": orientations
        }

        # Create a pandas DataFrame
        df = pd.DataFrame(data)

        # Save the DataFrame to an Excel file
        output_file = "/home/elghali/Desktop/SwarmCapture+/Data/simulation_data"+tag+".xlsx"  # Specify the desired name of the output Excel file

        # Create an ExcelWriter object
        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            # Write the DataFrame to the Excel file
            df.to_excel(writer, sheet_name="Sheet1", index=False)

        # The "with" block will automatically close the writer after saving the file.

        print('Excel Data saved')


        ########################################################################################################################
        # Save using Pickle module for Plotting and Printing in a separate code
        ########################################################################################################################

        print('\nSaving agents history pickle file...')
        with open('/home/elghali/Desktop/SwarmCapture+/Data/Agents_History'+tag+'.pkl', 'wb') as file:
            pickle.dump(Agents_History, file)
        print('Saved successfully')

        print('\nSaving target history pickle file...')
        with open('/home/elghali/Desktop/SwarmCapture+/Data/Target_History'+tag+'.pkl', 'wb') as file:
            pickle.dump(Target_History, file)
        print('Saved successfully')

        # print('\nSaving target PCD history pickle file...')
        # with open('/home/elghali/Desktop/SwarmCapture+/Data/Target_PointCloud'+tag+'.pkl', 'wb') as file:
        #     pickle.dump(Target_Point_Cloud_History, file)
        # print('Saved successfully')

        print('\nSaving attachment points history pickle file...')
        with open('/home/elghali/Desktop/SwarmCapture+/Data/Attachment_Points'+tag+'.pkl', 'wb') as file:
            pickle.dump(attachment_points, file)
        print('Saved successfully')

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

#########################################################################################################################
# Disconnect from the simulation
#########################################################################################################################

p.resetSimulation()
p.disconnect()


########################################################################################################################
# Calculate performance
########################################################################################################################

##################################################################
# Consuming less fuel
fuel_penality = 0
for Agent in Agents_History[-1]:
    fuel_penality += Agent["Fuel_Consumed"]

fuel_reward = 1/fuel_penality

##################################################################
# Achieving fast capture
docked = 0
num_iter = len(Agents_History)
total_steps_remained = 0

for a in range(N):
    if Agents_History[-1][a]['DockTime'] != None and Agents_History[-1][a]['Mode'] == 'd':
            total_steps_remained += num_iter - Agents_History[-1][a]['DockTime']
            docked += 1

agents_docked_reward = 100*docked # Reward for docking
agents_remained_time = total_steps_remained*dt # time_remained_to_simulation_end
dock_reward = agents_docked_reward + agents_remained_time


##################################################################
# Smooth force input
force_spike_penality = 0
for a in range(N):
    for i in range(1,num_iter):
        Spacecraft = Agents_History[i][a]
        Spacecraft_prev = Agents_History[i-1][a]
        force = np.linalg.norm(Spacecraft['Control_Force'])
        force_prev = np.linalg.norm(Spacecraft_prev['Control_Force'])
        force_spike_penality += abs(force - force_prev)/dt

force_spike_reward = N*num_iter/(force_spike_penality)

##################################################################
# contact velocity performance
contact_velocity = 0
for a in range(N):
    if Agents_History[-1][a]['DockTime'] != None:
        dock_iter = Agents_History[-1][a]['DockTime']
        target_index = Agents_History[dock_iter][a]['Target']

        attachment_point = C.extract_attachment_point(attachment_points, target_index)
        target_vel = np.linalg.norm(attachment_point.velocity[dock_iter])
        agent_vel = np.linalg.norm(Agents_History[dock_iter-5][a]['State'][3:6])
        contact_velocity += abs(target_vel - agent_vel)

if contact_velocity != 0:
    contact_velocity_reward = 1/contact_velocity
else:
    contact_velocity_reward = 0

##################################################################
# pointing performance
pointing_error = 0
for a in range(N):
    for i in range(num_iter):
        Spacecraft = Agents_History[i][a]
        if len(Spacecraft['Control_Frame']) != 0:
            state = 'Smooth_State'
            # Agent State extraction
            
            x = Spacecraft[state][0]
            y = Spacecraft[state][1]
            z = Spacecraft[state][2]
            r = np.array([x,y,z])

            q = Spacecraft[state][6:10]
            w1 = Spacecraft[state][10]
            w2 = Spacecraft[state][11]
            w3 = Spacecraft[state][12]
            ang_vel = np.array([w1,w2,w3])
            
            ort = Spacecraft["Control_Frame"]
            _, axis_error, ang_error = C.point_agent(r,q,ang_vel,ort, sim_params, cond_viz)
            
            pointing_error += np.linalg.norm(axis_error)*ang_error


pointing_reward = N * num_iter / pointing_error

##################################################################
# w1, w2, w3, w4 = 1, 1, 1, 1
w1, w2, w3, w4, w5 = 1, 1, 1000, 1, 10

performance = w1*dock_reward + w2*fuel_reward + w3*force_spike_reward + w4*contact_velocity_reward + w5 * pointing_reward

# Debug
print('\n')
print('\n')
print('Performance metrics with weights :')
print(f'Achieving capture reward        : {w1} * {agents_docked_reward}')
print(f'Times remaining reward          : {w1} * {agents_remained_time}')
print(f'Fuel consumption reward         : {w2} * {fuel_reward}')
print(f'Force smoothness reward         : {w3} * {force_spike_reward}')
print(f'Contact velocity reward         : {w4} * {contact_velocity_reward}')
print(f'Pointing reward                 : {w5} * {pointing_reward}')
print(f'Total performance               : {performance}')


# Write to a file
with open("/home/elghali/Desktop/SwarmCapture+/performance.json", "w") as f:
    json.dump(performance, f)


########################################################################################################################
# Print Output Log
########################################################################################################################

print('\n')
print('Simulation Finished')
end_time = time.time()
runtime = end_time-start_time
# Convert the runtime to hours, minutes, and seconds
hours, remainder = divmod(runtime, 3600)
minutes, seconds = divmod(remainder, 60)
# Print the end time
end_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
print(f"End time: {end_time_formatted}")
# Print the runtime
print(f"Runtime: {int(hours)} hours, {int(minutes)} minutes, {seconds} seconds")
print('\n')


########################################################################################################################
# Excel data saving
########################################################################################################################

print('Saving Excel Data...')
# Combine all the data into a dictionary
data = {
    "positions": positions,
    "orientations": orientations
}

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
output_file = "/home/elghali/Desktop/SwarmCapture+/Data/simulation_data"+tag+".xlsx"  # Specify the desired name of the output Excel file

# Create an ExcelWriter object
with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    # Write the DataFrame to the Excel file
    df.to_excel(writer, sheet_name="Sheet1", index=False)

# The "with" block will automatically close the writer after saving the file.

print('Excel Data saved')


########################################################################################################################
# Save using Pickle module for Plotting and Printing in a separate code
########################################################################################################################

print('\nSaving agents history pickle file...')
with open('/home/elghali/Desktop/SwarmCapture+/Data/Agents_History'+tag+'.pkl', 'wb') as file:
    pickle.dump(Agents_History, file)
print('Saved successfully')

print('\nSaving target history pickle file...')
with open('/home/elghali/Desktop/SwarmCapture+/Data/Target_History'+tag+'.pkl', 'wb') as file:
    pickle.dump(Target_History, file)
print('Saved successfully')

# print('\nSaving target PCD history pickle file...')
# with open('/home/elghali/Desktop/SwarmCapture+/Data/Target_PointCloud'+tag+'.pkl', 'wb') as file:
#     pickle.dump(Target_Point_Cloud_History, file)
# print('Saved successfully')

print('\nSaving attachment points history pickle file...')
with open('/home/elghali/Desktop/SwarmCapture+/Data/Attachment_Points'+tag+'.pkl', 'wb') as file:
    pickle.dump(attachment_points, file)
print('Saved successfully')

print('\n')
print('CODE COMPILED WITHOUT ERRORS')
print(f'Check files with names ending in "{tag}"')
print('\n')


