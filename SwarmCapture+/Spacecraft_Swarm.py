import numpy as np
import math
import random
import Controllers as C
# import pyransac3d as pyrsc
import pybullet as p

class SimulationParameters:
    def __init__(self, 
                 AntFlk_Radius,  # Antiflocking radius
                 Flk_Radius,   # Distance to keep from landmarks
                 
                 Pointing_Proportional_Gain,
                 Pointing_Derivative_Gain,
                 
                 Flk_Potential,
                 AntFlk_Potential,
                 Encapsulate_Derivative_Gain,
                 
                 Capture_Proportional_Gain,
                 Capture_Derivative_Gain,
                 Capture_Alignment_Gain,
                 
                 Distance_bid_weight,
                 Velocity_bid_weight,
                 Normal_bid_weight):
        
        self.AntFlk_Radius = AntFlk_Radius
        self.Flk_Radius = Flk_Radius
        
        self.Pointing_Proportional_Gain = Pointing_Proportional_Gain
        self.Pointing_Derivative_Gain = Pointing_Derivative_Gain
        
        self.Flk_Potential = Flk_Potential
        self.AntFlk_Potential = AntFlk_Potential
        self.Encapsulate_Derivative_Gain = Encapsulate_Derivative_Gain
        
        self.Capture_Proportional_Gain = Capture_Proportional_Gain
        self.Capture_Derivative_Gain = Capture_Derivative_Gain
        self.Capture_Alignment_Gain = Capture_Alignment_Gain
        
        self.Distance_bid_weight = Distance_bid_weight
        self.Velocity_bid_weight = Velocity_bid_weight
        self.Normal_bid_weight = Normal_bid_weight

        
        
def Spacecraft_OBC(Spacecraft, attachment_points, iter, Agents,
                   cond_viz, querried_agent, sim_gains):
    # Agents is a list of all Spacecraft states at previous iteration
    # MIGHT INCLUDE INFORMATION THAT IS ONLY AVAILABLE IN SIMULATION !!
    # Filtering is needed

    # Spacecraft is the Spacecraft to be controlled.

    tar = Spacecraft["Target"]
    
    if Spacecraft['Mode'] == 'e':
        u = C.Encapsulate(Spacecraft, Agents, sim_gains, cond_viz)
        if querried_agent is not None:
            tar = assign_attachment_point_auction_bid(Spacecraft, Agents[querried_agent])
        else:
            tar = assign_attachment_point_highest_bid(Spacecraft)
                                          
    elif Spacecraft['Mode'] == 'c':
        u = C.Capture_PID(Spacecraft, sim_gains, attachment_points, iter, cond_viz)
        # u = C.Capture_MPC(Spacecraft, sim_gains, attachment_points, iter, cond_viz)
        
    elif Spacecraft['Mode'] == 'd':
        u = np.array([0,0,0,0,0,0])
        tar = Spacecraft['Target']
    else:
        raise ValueError(f"Invalid Spacecraft mode: {Spacecraft['Mode']}")


    return u, tar


def saturate(u, max_u):
    return np.clip(u, -max_u, max_u)

def saturate_frc(u, max_frc):    
    # Debug
    # print(f'Input saturation: {max_frc} N')

    u[0] = max_frc * np.tanh(u[0] / max_frc)
    u[1] = max_frc * np.tanh(u[1] / max_frc)
    u[2] = max_frc * np.tanh(u[2] / max_frc)

    return u

def saturate_trq(u, max_trq):
    
    # Debug
    # print(f'Input saturation: {max_trq} N.m')

    u[0] = max_trq * np.tanh(u[0] / max_trq)
    u[1] = max_trq * np.tanh(u[1] / max_trq)
    u[2] = max_trq * np.tanh(u[2] / max_trq)

    return u


#####################################################################################################################
#####################################################################################################################

def collision_condition(bodyA, bodyB):
    contact_points = p.getContactPoints(bodyA, bodyB)

    contact_point = (None, None, None)
    Contact = False
    if len(contact_points) > 0:
        Contact = True

        # pick first contact point
        contact_point = contact_points[0][6] # coordinate of the contact point on the agent's body in the world reference frame

        # pick deepest contact point
        # contact_point = min(contact_points, key=lambda c: c['contactDistance']) if contact_points else None


    return Contact, contact_point

def Dock_Relative_Pose(target_body_id, agent_body_id):

    # Extract absolute agent pose in Global Reference Frame (REF)
    age_pos, age_quat = p.getBasePositionAndOrientation(agent_body_id)

    # Extract absolute target pose in Global REF
    tar_pos, tar_quat = p.getBasePositionAndOrientation(target_body_id)

    # Calculate agent pose relative to target in global REF
    rel_pos = np.array(age_pos) - np.array(tar_pos)

    # Calculate rotation matrix to convert from global to target REF
    tar_rot = np.array(p.getMatrixFromQuaternion(tar_quat)).reshape(3, 3).T

    # Convert relative agent pose into target REF
    rel_pos = np.dot(tar_rot , np.array(rel_pos))

    ##############################################################################
    age_rot = np.array(p.getMatrixFromQuaternion(age_quat)).reshape(3, 3)
    rel_rot = np.dot(tar_rot, age_rot)
    rel_quat = C.matrix_to_quaternion(rel_rot)

    ##############################################################################
    RelPose = np.concatenate((rel_pos , rel_quat))

    return RelPose

def Check_Mode_Switch(Spacecraft, Spacecraft_prev, agent_body_id, target_body_id, Rflk):
    md = Spacecraft["Mode"]
    m = md
    DockPose = Spacecraft["DockPose"]
    Contact, contact_point = collision_condition(target_body_id, agent_body_id)
    constraint = Spacecraft['DockConstraint']

    if md == 'e':
        # Extract Readiness Signal
        At = Spacecraft['ActionTime']
        At_prev = Spacecraft_prev['ActionTime']
        current_time = Spacecraft['TimeStep']
        
        # Calculate distance to landmarks
        if len(Spacecraft['LC']) != 0:
            dis = np.linalg.norm(np.array(Spacecraft['State'][:3]) - np.array(Spacecraft['LC']))        
            if dis < Rflk*1.3 and abs(At - At_prev) < 1e-4 and current_time > At: # if readiness signal converged to consensus switch to capture mode
                m = 'c'
                print('\n')
                print(f'Agent with ID {agent_body_id} switched to capture')
                print('\n')
        else:
            m = 'e'

    elif md == 'c':
        if Contact: # If docking achieved by making contact, switch to docked mode.
            if constraint == None:
                constraint = True
                # constraint = create_bonding_constraint(target_body_id, agent_body_id)
                # print(f'Created constraint with id -> {constraint} between body {target_body_id} and body {agent_body_id}')

                print('\n')
                print(f'Agent with ID {agent_body_id} Touched !')

                DockPose = Dock_Relative_Pose(target_body_id, agent_body_id)
                print(f'Saved relative pose of agent {agent_body_id}') 

                p.setCollisionFilterGroupMask(agent_body_id,-1,0,0)
                print(f'Disabled agent {agent_body_id} from collision detection')
                print(f'Agent with ID {agent_body_id} Docked')
                print('\n')
                 
            m = 'd'

    elif md == 'd':
        m = 'd'

    return m, constraint, DockPose, contact_point



#####################################################################################################################
#####################################################################################################################

def calculate_distance(position1, position2):
    return math.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2 + (position1[2] - position2[2])**2)


def Random_Agents(Agents, ):
    min_distance = 2.5  # minimum spawn distance between agents

    valid_position = False
    iter = 0
    while not valid_position and iter < 100:
        iter += 1
        print('Unvalid random position generation for agent')
        far = random.choice([True, False])
        new_position = Random_State_Gen(far)
        
        if len(Agents) == 0:
            valid_position = True
            print('Valid Now')
        else:
            for agent in Agents:
                distance = calculate_distance(new_position, agent['State'][0:3])
                if distance < min_distance:
                    break
                else:
                    valid_position = True
                    print('Valid Now')
    
    if iter > 100:
        raise SyntaxError('Could not generate agents due to too distant spawning condition. Please reduce inter-agent spawn distance or decrease total agent number in swarm')
    
    return new_position


def Random_State_Gen(far):
    xmin = -12
    # if far: xmin = -30
    
    xmax = -10
    ymax = 2
    zmax = 2

    # Generate agents as being deployed by mothership
    x = random.uniform(xmin,xmax)
    y = random.uniform(-ymax,ymax)
    z = random.uniform(-zmax,zmax)
    r = [x,y,z]

    # Generate orientation
    q = [0,0,0.7071068,0.7071068]
    
    # Random velocities in the direction of the target
    Vmax = 0.01  # Maximum linear velocity
    wmax = 0.001  # Maximum angular velocity
    
    vx = random.uniform(Vmax/2,Vmax)
    vy = random.uniform(-Vmax/5,Vmax/5)
    vz = random.uniform(-Vmax/5,Vmax/5)
    v = [vx,vy,vz]

    wx = random.uniform(-wmax,wmax)
    wy = random.uniform(-wmax,wmax)
    wz = random.uniform(-wmax,wmax)
    w = [wx,wy,wz]

    state = r+v+q+w # concatenation
    
    return state



#####################################################################################################################
#####################################################################################################################


def create_bonding_constraint(bodyA, bodyB):
    """
    Creates a bonding constraint between two bodies in PyBullet upon contact.

    Parameters:
    bodyA (int): The unique ID of the first body.
    bodyB (int): The unique ID of the second body.
    """

    # Check for contact between the two bodies
    contact_points = p.getContactPoints(bodyA, bodyB)


    contact_info = get_contact_info(contact_points)

    posA, quatA = p.getBasePositionAndOrientation(bodyA)
    contact_position_on_A = np.array(posA) - np.array(contact_info[5]) # Position of contact on bodyA

    posB, quatB = p.getBasePositionAndOrientation(bodyB)
    contact_position_on_B = np.array(posB) - np.array(contact_info[6])  # Position of contact on bodyB

    contact_orientation = p.getDifferenceQuaternion(quatA,quatB)

    # Create a fixed constraint
    constraint_id = p.createConstraint(
            parentBodyUniqueId = bodyA,
            parentLinkIndex = -1,
            childBodyUniqueId = bodyB,
            childLinkIndex = -1,
            jointType = p.JOINT_FIXED,
            jointAxis = [0, 0, 0],
            parentFramePosition = contact_position_on_A,
            childFramePosition = contact_position_on_B,
            # parentFrameOrientation = p.getQuaternionFromEuler([90, 0, 0]),
            childFrameOrientation = contact_orientation
        )

    return constraint_id



def get_contact_info(contact_points):
    
    N = len(contact_points)
    s = []
    for i in range(N):
        s += np.array(contact_points[i][5])
    average = s/N

    M = []
    for i in range(N):
        M.append(np.linalg.norm(np.array(contact_points[i][5])-average))

    idx = np.argmin(M, axis=None, out=None)

    contact_info = contact_points[idx]

    return contact_info


#####################################################################################################################
#####################################################################################################################


def calculate_bid(Spacecraft, attachment_points, iter, sim_params):
    
    wd = sim_params.Distance_bid_weight # distance weight
    wv = sim_params.Velocity_bid_weight # velocity alignment weight
    wn = sim_params.Normal_bid_weight # normal alignment weight
    
    Bids = []
    APs = Spacecraft['APs']

    if len(APs) != 0:
        for ap_idx in APs:
            ap = C.extract_attachment_point(attachment_points, ap_idx)
            
            # Distance
            distance = np.linalg.norm(np.array(Spacecraft['State'][:3]) - ap.position[iter])
            dbid = 1 / distance if distance != 0 else float('inf') # Infinity bid for closest target
            
            # Velocity
            ap_vel = ap.velocity[iter]
            agent_vel = np.array(Spacecraft['State'][3:6])
            rel_vel = np.linalg.norm(ap_vel - agent_vel)
            cos_theta = np.dot(ap_vel, agent_vel)/(np.linalg.norm(ap_vel)*np.linalg.norm(agent_vel))
            vbid = cos_theta + 1/rel_vel
            
            # Normal  
            ap_normal = ap.normal[iter]
            ap2agent = np.array(Spacecraft['State'][:3]) - np.array(ap.position[iter])
            cos_theta = np.dot(ap_normal, ap2agent)/(np.linalg.norm(ap_normal)*np.linalg.norm(ap2agent))
            nbid = cos_theta
            
            # Debug
            # print('\n')
            # print(f'Bids of Spacecraft {Spacecraft["ID"]} -> Attachment Point {ap_idx}')
            # print(f"weighted distance bid: {wd*dbid}")
            # print(f"weighted velocity bid: {wv*vbid}")
            # print(f"weighted normal bid: {wn*nbid}")
            
            bid = wd*dbid + wv*vbid + wn*nbid       
            Bids.append(bid)

    return Bids

def assign_attachment_point_auction_bid(Spacecraft, Neighbour_Agent):
    target = Spacecraft['Target']

    # The current spacecraft's attachment points and bids
    self_APs = Spacecraft['APs']
    self_bids = Spacecraft['APs_Bids']
    
    if len(Spacecraft['APs']) != 0:
        # Assign target with the highest bid
        target = self_APs[self_bids.index(max(self_bids))]

        # The queried agent's attachment points and bids
        neighbour_APs = Neighbour_Agent['APs']
        neighbour_bids = Neighbour_Agent['APs_Bids']
        
        # Set to track APs that are common to both spacecraft and neighbor
        common_APs = set(self_APs).intersection(neighbour_APs)
        
        # Check if the queried neighbour has a higher bid on the target attachment point
        if target in common_APs:
            ngh_idx = neighbour_APs.index(target)
            self_idx = self_APs.index(target)
            
            # If the neighbor has a higher bid, we need to adjust the target
            if neighbour_bids[ngh_idx] > self_bids[self_idx]:
                # Find the next highest bid among the remaining attachment points
                remaining_APs = list(set(self_APs) - {target})
                remaining_bids = [self_bids[self_APs.index(ap)] for ap in remaining_APs]

                if remaining_APs:  # Ensure there are remaining APs
                    # Pick target with next highest bid    
                    target = remaining_APs[remaining_bids.index(max(remaining_bids))]
                else:
                    target = []  # If no valid next target, set to None

    return target


def assign_attachment_point_highest_bid(Spacecraft):
    target = Spacecraft['Target']

    # The current spacecraft's attachment points and bids
    self_APs = Spacecraft['APs']
    self_bids = Spacecraft['APs_Bids']
    
    if len(self_APs) != 0:
        # Assign target with the highest bid
        target = self_APs[self_bids.index(max(self_bids))]


    return target
