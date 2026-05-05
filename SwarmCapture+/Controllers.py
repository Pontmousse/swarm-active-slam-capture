import numpy as np
import random
import math as m
import pybullet as p
import math

# Choose state feedback from simulation (noisy) or smoothed with low-pass-filter
# state = 'State'
state = 'Smooth_State'




# def Sheppard(C):
#     # C is a rotation matrix or an array of shape (3, 3)
#     # Q quaternion convention is: scalar at the end of vector

#     # First step
#     B0_2 = 0.25 * (1 + np.trace(C))  # Beta Zero Squared
#     B1_2 = 0.25 * (1 + 2 * C[0][0] - np.trace(C))  # Beta One Squared
#     B2_2 = 0.25 * (1 + 2 * C[1][1] - np.trace(C))  # Beta Two Squared
#     B3_2 = 0.25 * (1 + 2 * C[2][2] - np.trace(C))  # Beta Three Squared
#     B_2 = np.array([B0_2, B1_2, B2_2, B3_2])
#     k = np.argmax(B_2)

#     # Second step
#     if k == 0:
#         B0 = np.sqrt(B0_2)
#         B1 = (C[1][2] - C[2][1]) / (4 * B0)
#         B2 = (C[2][0] - C[0][2]) / (4 * B0)
#         B3 = (C[0][1] - C[1][0]) / (4 * B0)
#         Q = np.array([B1, B2, B3, B0])  # Reorder to [B1, B2, B3, B0]
#     elif k == 1:
#         B1 = np.sqrt(B1_2)
#         B0 = (C[1][2] - C[2][1]) / (4 * B1)
#         B2 = (C[0][1] + C[1][0]) / (4 * B1)
#         B3 = (C[2][0] + C[0][2]) / (4 * B1)
#         Q = np.array([B2, B3, B0, B1])  # Reorder to [B2, B3, B0, B1]
#     elif k == 2:
#         B2 = np.sqrt(B2_2)
#         B0 = (C[2][0] - C[0][2]) / (4 * B2)
#         B1 = (C[0][1] + C[1][0]) / (4 * B2)
#         B3 = (C[1][2] + C[2][1]) / (4 * B2)
#         Q = np.array([B3, B0, B1, B2])  # Reorder to [B3, B0, B1, B2]
#     elif k == 3:
#         B3 = np.sqrt(B3_2)
#         B0 = (C[0][1] - C[1][0]) / (4 * B3)
#         B2 = (C[1][2] + C[2][1]) / (4 * B3)
#         B1 = (C[2][0] + C[0][2]) / (4 * B3)
#         Q = np.array([B0, B1, B2, B3])  # Already in the desired order [B0, B1, B2, B3]

#     if Q[3] < 0:
#         Q = -Q

#     return Q

#####################################################################################################################
#####################################################################################################################

def matrix_to_quaternion(rotation_matrix):
    tr = rotation_matrix[0][0] + rotation_matrix[1][1] + rotation_matrix[2][2]

    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (rotation_matrix[2][1] - rotation_matrix[1][2]) / S
        qy = (rotation_matrix[0][2] - rotation_matrix[2][0]) / S
        qz = (rotation_matrix[1][0] - rotation_matrix[0][1]) / S
    elif (rotation_matrix[0][0] > rotation_matrix[1][1]) and (rotation_matrix[0][0] > rotation_matrix[2][2]):
        S = math.sqrt(1.0 + rotation_matrix[0][0] - rotation_matrix[1][1] - rotation_matrix[2][2]) * 2
        qw = (rotation_matrix[2][1] - rotation_matrix[1][2]) / S
        qx = 0.25 * S
        qy = (rotation_matrix[0][1] + rotation_matrix[1][0]) / S
        qz = (rotation_matrix[0][2] + rotation_matrix[2][0]) / S
    elif rotation_matrix[1][1] > rotation_matrix[2][2]:
        S = math.sqrt(1.0 + rotation_matrix[1][1] - rotation_matrix[0][0] - rotation_matrix[2][2]) * 2
        qw = (rotation_matrix[0][2] - rotation_matrix[2][0]) / S
        qx = (rotation_matrix[0][1] + rotation_matrix[1][0]) / S
        qy = 0.25 * S
        qz = (rotation_matrix[1][2] + rotation_matrix[2][1]) / S
    else:
        S = math.sqrt(1.0 + rotation_matrix[2][2] - rotation_matrix[0][0] - rotation_matrix[1][1]) * 2
        qw = (rotation_matrix[1][0] - rotation_matrix[0][1]) / S
        qx = (rotation_matrix[0][2] + rotation_matrix[2][0]) / S
        qy = (rotation_matrix[1][2] + rotation_matrix[2][1]) / S
        qz = 0.25 * S

    return [qx, qy, qz, qw]

#####################################################################################################################
#####################################################################################################################
def low_pass_filter(new_value, prev_value, alpha):
    """
    Apply a low-pass filter to smooth the new value.
    
    Args:
        new_value: Current value to filter.
        prev_value: Previous filtered value.
        alpha: Smoothing factor (0 < alpha < 1).
        
    Returns:
        Smoothed value.
    """
    return alpha * new_value + (1 - alpha) * prev_value


def smooth_state(Spacecraft, Spacecraft_prev, alpha):
    smoothed_state = []
    for s in range(len(Spacecraft['State'])):
        smoothed_state.append(low_pass_filter(Spacecraft['State'][s],
                                 Spacecraft_prev['Smooth_State'][s],
                                 alpha))
    return smoothed_state
    
def smooth_force(Spacecraft, Spacecraft_prev, alpha):
    smoothed_force = []
    for s in range(len(Spacecraft['Control_Force'])):
        smoothed_force.append(low_pass_filter(Spacecraft['Control_Force'][s],
                                 Spacecraft_prev['Smooth_Control_Force'][s],
                                 alpha))
    return smoothed_force


def smooth_torque(Spacecraft, Spacecraft_prev, alpha):
    smoothed_force = []
    for s in range(len(Spacecraft['Control_Torque'])):
        smoothed_force.append(low_pass_filter(Spacecraft['Control_Torque'][s],
                                 Spacecraft_prev['Smooth_Control_Torque'][s],
                                 alpha))
    return smoothed_force


#####################################################################################################################
#####################################################################################################################


def point_agent(r, q, ang_vel, ort, sim_gains, cond_viz):
    # Extract Controller gains
    Kp = sim_gains.Pointing_Proportional_Gain 
    Kd = sim_gains.Pointing_Derivative_Gain

    if len(ort) != 0: # ort is a rotation matrix (an array of shape 3x3)
        ort = matrix_to_quaternion(ort.T) # Convert to quaternion

        # Rotation of target orientation
        angle = math.pi / 2.0
        axis_of_rotation = [0, 0, 1]  # normalized axis of rotation
        target_quaternion = p.getQuaternionFromAxisAngle(axis_of_rotation, angle)
        ort = p.multiplyTransforms([0, 0, 0], ort, [0, 0, 0], target_quaternion)[1]
        
        # if cond_viz:
            # plot_ort_quat(r, ort, 0.4)

        quat_error = p.getDifferenceQuaternion(ort, q)
        axis_error, ang_error = p.getAxisAngleFromQuaternion(quat_error)
        
        # Compute the desired control torque
        Trq = -Kp*np.array(axis_error)*ang_error-Kd*np.array(ang_vel)
        # print(Trq)
    else:
        Trq = np.array([0,0,0])

    return Trq, axis_error, ang_error

#####################################################################################################################
#####################################################################################################################

def plot_ort_quat(position, quaternion, a):
    # Get body position and orientation
    origin = np.array(position)

    # Convert quaternion to rotation matrix
    rot_matrix = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    arrow_size = 2 # np.linalg.norm(np.array(p.getAABB(body_id)[1]) - np.array(p.getAABB(body_id)[0])) * 0.5

    # Plot X, Y, and Z axes using debug lines
    p.addUserDebugLine(origin, origin + arrow_size * rot_matrix[0], lineColorRGB=[1-a, a, a], lineWidth=3)
    p.addUserDebugLine(origin, origin + arrow_size * rot_matrix[1], lineColorRGB=[a, 1-a, a], lineWidth=3)
    p.addUserDebugLine(origin, origin + arrow_size * rot_matrix[2], lineColorRGB=[a, a, 1-a], lineWidth=3)

    # Optional: Display text labels for each axis
    p.addUserDebugText('X', origin + arrow_size * rot_matrix[0], textColorRGB=[1-a, a, a], textSize=1.5)
    p.addUserDebugText('Y', origin + arrow_size * rot_matrix[1], textColorRGB=[a, 1-a, a], textSize=1.5)
    p.addUserDebugText('Z', origin + arrow_size * rot_matrix[2], textColorRGB=[a, a, 1-a], textSize=1.5)

#####################################################################################################################
#####################################################################################################################

def plot_ort_rot(position, rot_matrix, a):
    # Get body position and orientation
    origin = np.array(position)

    arrow_size = 3 # np.linalg.norm(np.array(p.getAABB(body_id)[1]) - np.array(p.getAABB(body_id)[0])) * 0.5

    # Plot X, Y, and Z axes using debug lines
    p.addUserDebugLine(origin, origin + arrow_size * rot_matrix[0], lineColorRGB=[1-a, a, a], lineWidth=3)
    p.addUserDebugLine(origin, origin + arrow_size * rot_matrix[1], lineColorRGB=[a, 1-a, a], lineWidth=3)
    p.addUserDebugLine(origin, origin + arrow_size * rot_matrix[2], lineColorRGB=[a, a, 1-a], lineWidth=3)

    # Optional: Display text labels for each axis
    p.addUserDebugText('X', origin + arrow_size * rot_matrix[0], textColorRGB=[1-a, a, a], textSize=1.5)
    p.addUserDebugText('Y', origin + arrow_size * rot_matrix[1], textColorRGB=[a, 1-a, a], textSize=1.5)
    p.addUserDebugText('Z', origin + arrow_size * rot_matrix[2], textColorRGB=[a, a, 1-a], textSize=1.5)

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################


def Encapsulate(Spacecraft, Agents, sim_gains, cond_viz):    
    # Extract controller gains
    nu = sim_gains.Flk_Potential # Flocking Potential Gain
    ga = sim_gains.AntFlk_Potential # Antiflocking Potential Gain
    Kd = sim_gains.Encapsulate_Derivative_Gain # PID Gain
    Rflk = sim_gains.Flk_Radius
    Rant = sim_gains.AntFlk_Radius

    # State extraction
    x = Spacecraft[state][0]
    y = Spacecraft[state][1]
    z = Spacecraft[state][2]
    r = np.array([x,y,z])

    vx = Spacecraft[state][3]
    vy = Spacecraft[state][4]
    vz = Spacecraft[state][5]
    rdot = np.array([vx,vy,vz])

    q = Spacecraft[state][6:10]

    w1 = Spacecraft[state][10]
    w2 = Spacecraft[state][11]
    w3 = Spacecraft[state][12]
    ang_vel = np.array([w1,w2,w3])
    
    #####################################
    ######### Flocking Behavior #########
    L = Spacecraft['LandSet']
    # L = Spacecraft['FeatureSet']
    
    Nl = len(L)

    if Nl == 0:
        F_flk = np.array([0, 0, 0])
        print('No Landmarks !!')
    else:
        Fx = 0
        Fy = 0
        Fz = 0
        for i in range(Nl):
            xp = L[i][0]
            yp = L[i][1]
            zp = L[i][2]
            rp = np.array([xp, yp, zp])
            Nr = np.linalg.norm(r - rp)
            
            numer = nu * (Rflk - Nr)
            denom = Nr
            
            Fx += (numer / denom) * (x - xp)
            Fy += (numer / denom) * (y - yp)
            Fz += (numer / denom) * (z - zp)

        F_flk = np.array([Fx, Fy, Fz])

    #########################################
    ######### AntiFlocking Behavior #########

    Ngh = Spacecraft['AntFlkSet']
    Nc = len(Ngh)

    if Nc == 0:
        F_ant = np.array([0, 0, 0])
    else:
        Fx = 0
        Fy = 0
        Fz = 0
        for i in range(Nc):
            xp = Agents[Ngh[i]][state][0]
            yp = Agents[Ngh[i]][state][1]
            zp = Agents[Ngh[i]][state][2]
            rp = np.array([xp, yp, zp])
            Nr = np.linalg.norm(r - rp)

            numer = ga * (1/Rant - 1/Nr)
            denom = Nr ** 3
            
            Fx += (numer / denom) * (xp - x)
            Fy += (numer / denom) * (yp - y)
            Fz += (numer / denom) * (zp - z)

        F_ant = np.array([Fx, Fy, Fz])

    ##################################################
    ######### Derivative Feedback (Optional) #########
    F_v = -Kd * rdot
    
    ##################################################
    ####################### Total ####################
    
    Frc = F_flk + F_ant + F_v
    
    # Debug
    # print('\n')
    # print(f'Flocking Force: {F_flk}')
    # print(f'Antiflocking Force: {F_ant}')
    # print(f'Velocity feedback force: {F_v}')
    # print(f'Total force: {Frc}')
    

    ##################################################
    ################ Attitude Pointing ###############
    ort = Spacecraft["Control_Frame"]
    Trq, _, _ = point_agent(r,q,ang_vel,ort, sim_gains, cond_viz)

    
    u = np.concatenate((Frc,Trq))

    return u


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def extract_attachment_point(attachment_points, index):
    if not index:
        attachment_point = None
        return attachment_point
    
    
    found = False
    Nap = len(attachment_points)
    f = 1
    while not found:
        attachment_point = attachment_points[f-1]
        idx = attachment_point.idx
        if idx == index:
            found = True
            return attachment_point
        f += 1
        if f > Nap:
            attachment_point = None
            raise SyntaxError('Attachment point not found')
            break
        
    return attachment_point



def Capture_PID(Spacecraft, sim_gains, attachment_points, iter, cond_viz):
    # Extract controller gains    
    Kp = sim_gains.Capture_Proportional_Gain # proportial error feedback
    Kd = sim_gains.Capture_Derivative_Gain # derivative feedback for soft contact
    Kn = sim_gains.Capture_Alignment_Gain # alignment with normal: zero to cancel this term

    # Agent State extraction
    x = Spacecraft[state][0]
    y = Spacecraft[state][1]
    z = Spacecraft[state][2]
    r = np.array([x,y,z])

    vx = Spacecraft[state][3]
    vy = Spacecraft[state][4]
    vz = Spacecraft[state][5]
    rdot = np.array([vx,vy,vz])


    q = Spacecraft[state][6:10]
    w1 = Spacecraft[state][10]
    w2 = Spacecraft[state][11]
    w3 = Spacecraft[state][12]
    ang_vel = np.array([w1,w2,w3])
    
    # ############# Debug
    # print(Spacecraft['ID'])
    # print(Spacecraft['Target'])
    # print('\n')
    # print(Spacecraft['APs'])
    # print('\n')
    # print(Spacecraft['APs_Bids'])
    
    if Spacecraft['Target'] == []:
        Frc = np.array([0, 0, 0])
    else:
        # Target Attachment Point State extraction
        attachment_point = extract_attachment_point(attachment_points,
                                                    Spacecraft['Target'])
        
        # AP position
        xp = attachment_point.position[iter][0]
        yp = attachment_point.position[iter][1]
        zp = attachment_point.position[iter][2]
        rp = np.array([xp,yp,zp])
         
        # AP velocity
        vxp = attachment_point.velocity[iter][0]
        vyp = attachment_point.velocity[iter][1]
        vzp = attachment_point.velocity[iter][2]
        rpdot = np.array([vxp,vyp,vzp])
        
        # AP normal
        nxp = attachment_point.normal[iter][0]
        nyp = attachment_point.normal[iter][1]
        nzp = attachment_point.normal[iter][2]
        n = np.array([nxp,nyp,nzp])
        
        ##################################################
        ################ Proportional Feedback  ##########
        F_p = -Kp * (r - rp)
    
        ##################################################
        ######### Derivative Feedback  ###################
        F_v = -Kd * (rdot - rpdot)
        
        ##################################################
        ######### Normal Estimate Feedback  ##############
        dir = (r-rp) / np.linalg.norm(r-rp)
        mag = (1 - np.dot(n,dir))
        force_dir = dir - np.dot(dir, n)*n # the component of dir that is orthogonal to n. This ensures that the force applied will rotate the vectors toward alignment without affecting their length.
        F_n = -Kn * mag * force_dir
        
        ##################################################
        ###################### TOTAL #####################
        Frc = F_p + F_v + F_n

    ##################################################
    ################ Attitude Pointing ###############
    ort = Spacecraft["Control_Frame"]
    Trq, _, _ = point_agent(r,q,ang_vel,ort, sim_gains, cond_viz)

    
    u = np.concatenate((Frc,Trq))

    return u




def chw_force(Spacecraft, altitude, cancel_chw):
    # Calculate orbital mean motion
    G = 6.67430e-11 # m3 kg−1 s−2
    M = 5.972e24 # kg
    R = 6371 # km
    orbital_radius = (R+altitude)*1000 # in meters
    omega = np.sqrt((G*M)/(orbital_radius**3)) # in rad/s
    omega_dot = 0 #for circular orbits
    

    # State extraction
    x = Spacecraft[state][0]
    y = Spacecraft[state][1]
    z = Spacecraft[state][2]

    vx = Spacecraft[state][3]
    vy = Spacecraft[state][4]
    vz = Spacecraft[state][5]

    # CHW relative motion acceleration 
    Ax = 2*omega*vz + omega_dot*z
    Ay = -1*omega**2*y
    Az = 3*omega**2*z-2*omega*vx-omega_dot*x
    A_chw = np.array([Ax, Ay, Az])

    # extract spacecraft mass
    mass = Spacecraft['Mass']
    
    # CHW relative motion Force
    F_chw = mass*A_chw
    
    if cancel_chw:
        F_chw = F_chw*0
    
    return F_chw










    
    
   
# #####################################################################################################################
# #####################################################################################################################
# #####################################################################################################################
# #####################################################################################################################
# #####################################################################################################################
# #####################################################################################################################
# #####################################################################################################################
# #####################################################################################################################

# def Search(Spacecraft, Agents, xbox, ybox, zbox, Rcol, cond_viz):
#     # State extraction
#     x = Spacecraft["State"][0]
#     y = Spacecraft["State"][1]
#     z = Spacecraft["State"][2]
#     r = np.array([x,y,z])

#     vx = Spacecraft["State"][3]
#     vy = Spacecraft["State"][4]
#     vz = Spacecraft["State"][5]
#     rdot = np.array([vx,vy,vz])

#     q = Spacecraft["State"][6:10]

#     w1 = Spacecraft["State"][10]
#     w2 = Spacecraft["State"][11]
#     w3 = Spacecraft["State"][12]
#     ang_vel = np.array([w1,w2,w3])
    
#     #####################################
#     ######### Bouncing maneuver #########
#     # Hyperparameter
#     k = 500  # N of Force
#     def Frnd(k):
#         Fr = k/4 * random.uniform(-1,1)
#         return Fr
    
#     # Firing the bouncing thrusts for hitting x boundary
#     if x > xbox:
#         Fx = np.array([-k, Frnd(k), Frnd(k)])
#     elif x < -xbox:
#         Fx = np.array([k, Frnd(k), Frnd(k)])
#     else:
#         Fx = np.array([0, 0, 0])
    
#     # Firing the bouncing thrusts for hitting y boundary
#     if y > ybox:
#         Fy = np.array([Frnd(k), -k, Frnd(k)])
#     elif y < -ybox:
#         Fy = np.array([Frnd(k), k, Frnd(k)])
#     else:
#         Fy = np.array([0, 0, 0])

#     # Firing the bouncing thrusts for hitting z boundary
#     if z > zbox:
#         Fz = np.array([Frnd(k), Frnd(k), -k])
#     elif z < -zbox:
#         Fz = np.array([Frnd(k), Frnd(k), k])
#     else:
#         Fz = np.array([0, 0, 0])
    
#     F_bnc = Fx + Fy + Fz # Addition not Concatenation

#     ########################################
#     ######### Aggregation maneuver #########
#     dir = Spacecraft['LCD']
#     Kagg = 1000
#     vel_dir = rdot / np.linalg.norm(rdot)
    
#      # Debug
#     # print('\n')
#     # print("rdot:", rdot)
#     # print("vel_dir", vel_dir)
#     # print("LCD dir", dir)

#     if len(dir) == 0:
#         F_agg = np.array([0, 0, 0])
#     else:
#         F_agg = -Kagg * (vel_dir - dir)
    
#     # Debug
#     # print('\n')
#     # print("F_agg:", F_agg)

#     ######################################
#     ######### Collision maneuver #########
#     # Hyperparameter
#     ga = 10  # Collision avoidance strength
#     eps = 0.0001; # for numerical stability when dividing by 0 or "np.finfo(float).eps"
#     Ngh = Spacecraft['CollSet']
#     Nc = len(Ngh)
    
#     # No Neighbors
#     if Nc == 0:
#         F_col = np.array([0, 0, 0])
#     else:
#         Fx = 0
#         Fy = 0
#         Fz = 0
#         for i in range(Nc):
#             xp = Agents[Ngh[i]]["State"][0]
#             yp = Agents[Ngh[i]]["State"][1]
#             zp = Agents[Ngh[i]]["State"][2]
#             rp = np.array([xp, yp, zp])
#             Nr = np.linalg.norm(r - rp)
            
#             # X direction
#             Xd = (x ** 2) / (x + eps) - (xp ** 2) / (xp + eps)
#             numer = abs(x - xp) * (1 / Nr - 1 / Rcol) * (x - xp + Xd)
#             denom = Nr ** 3 * np.sqrt((x - xp) * Xd)+eps
#             Fx = Fx + (ga / 2) * (numer / denom)
            
#             # Y direction
#             Yd = (y ** 2) / (y + eps) - (yp ** 2) / (yp + eps)
#             numer = abs(y - yp) * (1 / Nr - 1 / Rcol) * (y - yp + Yd)
#             denom = Nr ** 3 * np.sqrt((y - yp) * Yd)+eps
#             Fy = Fy + (ga / 2) * (numer / denom)

#             # Z direction
#             Zd = (z ** 2) / (z + eps) - (zp ** 2) / (zp + eps)
#             numer = abs(z - zp) * (1 / Nr - 1 / Rcol) * (z - zp + Zd)
#             denom = Nr ** 3 * np.sqrt((z - zp) * Zd)+eps
#             Fz = Fz + (ga / 2) * (numer / denom)
        
#         F_col = np.array([Fx, Fy, Fz])
    
    
#     Frc = F_bnc + F_agg + F_col
    
#     #####################################
#     ######### Attitude Pointing #########
#     ort = Spacecraft["LCD_Frame"]
#     Trq = point_agent(r,q,ang_vel,ort, cond_viz)

    
#     u = np.concatenate((Frc,Trq))

#     # Debug
#     # print('\n')
#     # print("Force", F)
#     # print("Torque", Trq)
#     # print('u', u)
#     # print('\n')

#     return u
 
