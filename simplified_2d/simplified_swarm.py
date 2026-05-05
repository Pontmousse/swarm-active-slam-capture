import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

import robomaster

import time
from random import randrange, sample
import copy

import sys
import time
from swarm_control import SubAruco, SubMM, Controller
from swarm_control import sub_simulation as ss


def upper_bound(array, bound):
    
    sign = np.copy(array)
    sign[sign>=0] = 1
    sign[sign<0] = -1

    n = len(array)
    bounded_vel = [0]*n
    for i in range(n):
        bounded_vel[i] = min(abs(array[i]), bound)

    return bounded_vel * sign



def loc_to_glob(v, theta):
    '''
    v = np.array of size (2,1)
    theta = a float
    '''

    temp = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coord_trans = np.linalg.inv(temp)
    result = np.matmul(coord_trans, v)
    
    return [result[0][0], result[1][0]]



def get_angle(vector):
    if vector.any():
        norm = np.sqrt(np.power(vector[0],2) + np.power(vector[1],2))
        vector = vector/norm
        angle = np.arccos(vector[0])
        if vector[1] >= 0:
            return angle
        else:
            return -angle
    else:
        return 0


def run_algorithm(id, s1_robot, s1_robots, s1_control, Land):
    s1_state = s1_robot['State']
    glob_pos = s1_state[:2]
    glob_vel = s1_state[2:4]
    ang_pos = s1_state[2]
    ang_vel = s1_state[5]
    
    
    # Set landmarks observed and landmark direction
    landmark_set, landmark_dir = ss.set_landmarks_detected(s1_robot, Land, Rdet = 0.7)
    s1_robot['LandSet'] = landmark_set
    s1_robot['LCD'] = landmark_dir
    
    print(s1_robot['LandSet'])
    print(s1_robot['LCD'])
       
    
    
       
    # Set neighbouring agents' positions.
    all_pos = s1_robots
       
       
    
    # Compute control action
    # force = s1_control.search(id, glob_pos, s1_robots, s1_state[3:5])
    # force = s1_control.constant()
    force = s1_control.explore2(id, glob_pos, glob_vel, all_pos, s1_robot['LandSet'])
    print("force:", force)
    
    if len(landmark_dir) != 0:
        torque = s1_control.pointing_behavior2(ang_vel, ang_pos, s1_robot['LCD'])
    else:
        torque = 0
    print("torque:", torque)
    
    # vel = upper_bound(loc_to_glob(force[:2].reshape(2, 1), s1_robot['State'][2]),0.3)
    # print("control_velocity:", vel)
    
    # Propagate Robot motion
    input = np.concatenate([force.flatten(), [torque]])
    s1_robot = ss.Propagate_Robot_Motion(s1_robot, input, Targ, dt)
    
    
    
    
    
if __name__ == "__main__":
    
    # setup the parameters for this process
    sn_list = ['159CG9V0050HED',  '159CKC50070ECX']
    # sn_list = ['159CG9V0050HED']
    agent_num = len(sn_list)
    time_out = 20 # simulation period in seconds
    dt = 0.05 # simulation time steps in seconds
    
    
    # initial states in the form [x y theta vx vy w]
    initial_states = [[-0.3, 0.25, 0, 0, 0, 0],
                      [0.5, 1.2, -90, 0, 0, 0]]
    
    
    ########################################################################
    # Target and Landmark generation
    omega = 0  # rad/s target tumbling rate
    Vt = np.array([0, 0])  # m/s target translation
    Center = np.array([0.5, 0.5])  # Target center position
    
    Target = [{'State': np.hstack((Center, [0], Vt, [omega])),
               'Mass': 0.5, 'Inertia': 0.05}]
    
    # Vertices for different polygon types
    # Square Target, Input corners in target reference frame.
    V = np.array([[-0.2, -0.2], [-0.2, 0.2], [0.2, 0.2], [0.2, -0.2]])
    # V = np.array([[-3, -2.5], [-2.5, 2.5], [0, 4.5], [1, 3.5], [2.5, -3.5]])
    
    # Generate polygonal landmarks based on the chosen vertices
    Target[0]['Landmarks'] = ss.Generate_Polygonal_Landmarks(Center, V, m = 7)
    ########################################################################
    
    # Robot generation
    s1_camera = []
    
    s1_robots = []
    s1_arucos = []
    for i in range(agent_num):
        s1_robot = [[{'ID': sn_list[i] , 'TimeStep': 0, 'State': initial_states[i],
                     'CommSet': [], 'CollSet': [], 'LandSet': [], 'AntFlkSet': [], 
                     'DockSet': [], 'LCD': [], 'Mode': 's', 'Mass': 4}]]
    
        s1_aruco = SubAruco(s1_camera, sn = sn_list[i], aruco_type="DICT_5X5_250", display=True)
        s1_robots.append(s1_robot[0])
        s1_arucos.append(s1_aruco)
        print(f"Generated Robot {id}")
    
    print("----------------- Generated All Robots ------------")
    
    
    
    # Initial plot
    # Plot_Robots([robot for robot in s1_robots], Target[0]['Landmarks'], False, False, xbox, ybox, Rcom, 0)
    
    # Propagation
    start_time = time.time()
    
    t = 0
    iter = 1
    
    while t <= time_out:
        for i in range(agent_num):
            # Monitoring output
            print(f'\nTime step {t} / {time_out} ||||| Agent {i+1} / {agent_num}')
            
            
            Targ = Target[-1]
            Land = Targ['Landmarks']
            
            s1_robot = copy.deepcopy(s1_robots[i][iter-1])           
            s1_state = s1_robot['State']
            
            
            ###########################################
    
            s1_control = Controller(agent_num=agent_num)
            run_algorithm(i, s1_robot, s1_robots, s1_control, Land)
            
    
            # Propagate target motion
            Targ = ss.Propagate_Target_Motion(Targ, dt)
            
            s1_robot['TimeStep'] += dt 
            
            s1_robots[i].append(s1_robot)
            Target.append(Targ)
    

        t += dt
        iter += 1


print(f'\nTotal number of iterations: {iter}\n')

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #

####################################################
####################################################
# %% Static Plot
####################################################
####################################################

# Create a new plot
# fig, ax = plt.subplots(figsize=(6, 6))

# # Plot robots as black dots
# for robot_H in s1_robots:
    
#     robot = robot_H[0]
#     x, y, theta = robot['State'][:3]
#     theta = np.deg2rad(theta)
#     ax.plot(x, y, 'ko', markersize=5)  # Plot robot position
#     ax.quiver(x, y, np.cos(theta), np.sin(theta), scale=10)  # Plot robot orientation

# # Plot landmarks as red crosses
# Land = Target[0]['Landmarks']
# plt.scatter(Land[:, 0], Land[:, 1], c='r')


# # Set plot limits
# ax.set_xlim([-1, 2])
# ax.set_ylim([-1, 2])

# # Add labels and title
# ax.set_title('Robots and Landmarks')
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# handles, labels = ax.get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# ax.legend(by_label.values(), by_label.keys())

# # Show the plot
# plt.show()


####################################################
####################################################
# %% Animated Plot
####################################################
####################################################
anim_robots = [[] for i in range(len(s1_robots))]
num_samples = len(s1_robots[0])
p = 25


for i in range(len(s1_robots)):
        anim_robots[i] = [s1_robots[i][j] for j in range(0, num_samples, p)]



# Function to update the animation
def update(frame, anim_robots, ax, Target):
    ax.clear()  # Clear the plot
    
    # Plot robots as black dots
    for robot_H in anim_robots:
        robot = robot_H[frame]
        x, y, theta = robot['State'][:3]
        theta = np.deg2rad(theta)
        ax.plot(x, y, 'ko', markersize=5)  # Plot robot position
        ax.quiver(x, y, np.cos(theta), np.sin(theta), scale=10)  # Plot robot orientation
        
        # Add a rectangle around the robot (as its body)
        robot_width = 0.15  # Width of the robot (adjust as needed)
        robot_length = 0.25  # Height of the robot (adjust as needed)

        # Create a rectangle centered at (x, y) with no rotation
        rect = patches.Rectangle(
            (-robot_length / 2, -robot_width / 2),  # Centered rectangle
            robot_length, robot_width,  # Width and height
            edgecolor='black', facecolor='blue'  # Red border, no fill
        )

        # Apply transformation to center the rectangle and rotate around the robot's center
        t = plt.matplotlib.transforms.Affine2D().rotate(theta).translate(x, y)
        rect.set_transform(t + ax.transData)  # Apply the transformation

        ax.add_patch(rect)  # Add the rectangle to the plot
        
    # Plot landmarks as red crosses
    Land = Target[0]['Landmarks']
    ax.scatter(Land[:, 0], Land[:, 1], c='r')

    # Set plot limits
    ax.set_xlim([-1, 3])
    ax.set_ylim([-1, 3])

    # Add labels and title
    time_step = robot['TimeStep']
    ax.set_title(f'Robots and Landmarks - {time_step} s')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

# Create a new plot
fig, ax = plt.subplots(figsize=(6, 6))


# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(anim_robots[0]), fargs=(anim_robots, ax, Target), repeat=False)

# Display the animation
plt.show()

# %%
# Define the writer (FFMpegWriter in this case)
Writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)

# Save the animation as a .mp4 video
ani.save('robots_animation.mp4', writer=Writer)

print("\nAnimation saved")




# %%
# Post-Processing and Mode Plotting
# Plot_Modes_History([agent for agent in s1_robots])

