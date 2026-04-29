import numpy as np
import pybullet as p
import open3d as o3d
import Controllers as C

def Set_Neighborhood(Spacecraft, Agents, R):
    # Spacecraft is a dictionary representing the agent of interest
    # agents is a list of dictionaries containing information of all agents

    N = len(Agents)

    C = []
    for i in range(N):
        if Agents[i]['ID'] != Spacecraft['ID']:
            D = ((Spacecraft['State'][0] - Agents[i]['State'][0]) ** 2 +
                 (Spacecraft['State'][1] - Agents[i]['State'][1]) ** 2) ** 0.5
            if D <= R:
                C.append(Agents[i]['ID'])

    return C

#####################################################################################################################
#####################################################################################################################

def Set_Landmark_Centroid_Direction(Spacecraft, Agents):
    Lcd = Spacecraft['LCD']
    Lc = Spacecraft['LC']
    Comm = Spacecraft['CommSet']
    Landm = Spacecraft['FeatureSet']

    if len(Landm) == 0: # If there are no landmarks in sight.
        if len(Comm) != 0: # If there are neighboring agents.
            # Check their registered landmark centroids
            n = len(Comm)
            Comm_Rlcd = [] # initialize the set of landmark centroid directions if registered by neighboring agents.
            
            # Debug
            # print('\n')
            # print("CommSet:", Comm)
            # print("length CommSet:", n)
            # print('\n')
            # Debug

            for i in range(n):

                # Debug
                # print("i:", i)
                # print("CommSet:", Comm)
                # print("Neighboring Agent:", Comm[i])
                # print('\n]')
                # Debug

                V = Agents[Comm[i]-1]['LCD'] # -1  because python indexing starts with 0.
                #CommSet stores agent IDs that starts from 1 to N.
                # ID=0 is reserved for the target.
                if len(V) != 0:
                    Comm_Rlcd.append(V) # Collect their landmark centroids
                    # note if some of them don't have, they will not add up
                    # because it will be an empty vector.

            if len(Comm_Rlcd) != 0: # Do this only if there is landmark centroid registered in some of the neighboring agents
                Rlcd = np.mean(Comm_Rlcd, axis=0).T
                Lcd = Rlcd / np.linalg.norm(Rlcd)

    elif len(Landm) != 0: # If there are landmarks within detection range (within sight)
        Rlcd = np.mean(Landm, axis=0).T - Spacecraft['State'][0:3]
        Lcd = Rlcd / np.linalg.norm(Rlcd)

    return Lcd, Lc

#####################################################################################################################
#####################################################################################################################


def Set_Landmark_Centroid_Direction2(Spacecraft, Agents):
    Lcd = Spacecraft['LCD']
    Lc = Spacecraft['LC']
    Comm = Spacecraft['CommSet']

    if len(Lc) == 0: # If there is no landmark centroid.
        if len(Comm) != 0: # If there are neighboring agents.
            # Check their registered landmark centroids
            n = len(Comm)
            Comm_Rlc = [] # initialize the set of landmark centroid if registered by neighboring agents.
            
            # Debug
            # print('\n')
            # print("CommSet:", Comm)
            # print("length CommSet:", n)
            # print('\n')
            # Debug

            for i in range(n):

                # Debug
                # print("i:", i)
                # print("CommSet:", Comm)
                # print("Neighboring Agent:", Comm[i])
                # print('\n]')
                # Debug

                V = Agents[Comm[i]-1]['LC'] # -1  because python indexing starts with 0.
                #CommSet stores agent IDs that starts from 1 to N.
                # ID=0 is reserved for the target.
                if len(V) != 0:
                    Comm_Rlc.append(V) # Collect their landmark centroids
                    # note if some of them don't have, they will not add up
                    # because it will be an empty vector.

            if len(Comm_Rlc) != 0: # Do this only if there is landmark centroid registered in some of the neighboring agents
                Lc = np.mean(Comm_Rlc, axis=0).T    
                Rlcd = Lc - Spacecraft['State'][0:3]
                Lcd = Rlcd / np.linalg.norm(Rlcd)

    elif len(Lc) != 0: # If there is a landmark centroid (within sight)
        Rlcd = Lc - Spacecraft['State'][0:3]
        Lcd = Rlcd / np.linalg.norm(Rlcd)

    return Lcd, Lc


#####################################################################################################################
#####################################################################################################################

def plot_LCD(Spacecraft):
    dir = Spacecraft['LCD']

    if len(dir) != 0:
        position  = Spacecraft['State'][0:3]

        # Calculate the end point of the line (e.g., 0.5 units from the body's position)
        endpoint = np.array(position) + 2 * dir

        # Draw the line in PyBullet (you may want to use a unique color for visibility)
        line_color = [1, 1, 0]  # Yellow line
        line_width = 3  # Adjust the line width as needed
        p.addUserDebugLine(np.array(position), endpoint, line_color, lineWidth=line_width)

    return

def plot_LC(Spacecraft):
    Lc = Spacecraft['LC']

    if len(Lc) != 0:
        # Draw the line in PyBullet (you may want to use a unique color for visibility)
        p.addUserDebugText(".", Lc, textColorRGB=[1, 1, 1], textSize=4)

    return

#####################################################################################################################
#####################################################################################################################

def Set_LCD_Frame(Spacecraft):

    LCD_Frame = Spacecraft["Control_Frame"]
    dir = Spacecraft["LCD"]

    if len(dir) != 0:
        if len(LCD_Frame) == 0:
            xdir = dir
            ydir = generate_random_orthogonal_vector(dir)
            zdir = np.cross(xdir,ydir)
            zdir = zdir/np.linalg.norm(zdir)
        else:
            xdir = dir
            ydir = generate_close_orthogonal_vector(dir, LCD_Frame[1])
            zdir = np.cross(xdir,ydir)
            zdir = zdir/np.linalg.norm(zdir)
        LCD_Frame = np.array([xdir, ydir, zdir])
        
    return LCD_Frame



def Set_Target_dir_Frame(Spacecraft, attachment_points):
    target_dir_Frame = Spacecraft["Control_Frame"]
    
    if Spacecraft['Target'] != []:
        attachment_point = C.extract_attachment_point(attachment_points, Spacecraft['Target'])
        dir = attachment_point.normal[Spacecraft['Iteration']]
        dir = -dir # Target direction is the opposite of the attachment point normal vector
    else:
        dir = Spacecraft["LCD"]
 
    if len(dir) != 0:
        if len(target_dir_Frame) == 0:
            xdir = dir
            ydir = generate_random_orthogonal_vector(dir)
            zdir = np.cross(xdir,ydir)
            zdir = zdir/np.linalg.norm(zdir)
        else:
            xdir = dir
            ydir = generate_close_orthogonal_vector(dir, target_dir_Frame[1])
            zdir = np.cross(xdir,ydir)
            zdir = zdir/np.linalg.norm(zdir)
        target_dir_Frame = np.array([xdir, ydir, zdir])
        
    return target_dir_Frame

#####################################################################################################################
def generate_random_orthogonal_vector(unit_vector):
    # Step 1: Generate a random vector of the same dimension as the unit_vector
    random_vector = np.random.rand(*unit_vector.shape)
    
    # Step 2: Project the random vector onto the unit_vector to make it orthogonal
    orthogonal_vector = random_vector - np.dot(random_vector, unit_vector) * unit_vector
    
    # Step 3: Normalize the resulting vector to make it a unit vector
    orthogonal_unit_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)
    
    return orthogonal_unit_vector

#####################################################################################################################
def generate_close_orthogonal_vector(unit_vector, close_vector):    
    # Step 1: Project the random vector onto the unit_vector to make it orthogonal
    orthogonal_vector = close_vector - np.dot(close_vector, unit_vector) * unit_vector
    
    # Step 2: Normalize the resulting vector to make it a unit vector
    orthogonal_unit_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)
    
    return orthogonal_unit_vector

#####################################################################################################################
#####################################################################################################################

def lidar_odometry(Spacecraft0,Spacecraft1,IMU):
    # IMU = True if there is way for the agent to sense its own displacement that is independent of target
    # motion and use it as initialization for the ICP. If not available, use PCA-based initialization that
    # only needs inputs from the lidar (successive point clouds)


    Odometry = []
    Landm0  = Spacecraft0['LandSet']
    Landm1  = Spacecraft1['LandSet']

    if len(Landm0) != 0 and len(Landm1) != 0: # If there are landmarks within detection range (within sight): # If there exist landmarks
        pcd0 = o3d.geometry.PointCloud()
        pcd0.points = o3d.utility.Vector3dVector(np.asarray(Landm0))
        pcd0.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(np.asarray(Landm1))
        pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        if IMU: # If IMU = True # If IMU is available (Allows better initialization of ICP)
            p0 = Spacecraft0['State'][0:3]
            q0 = Spacecraft0['State'][6:10]
            p1 = Spacecraft1['State'][0:3]
            q1 = Spacecraft1['State'][6:10]
            dp = np.array(p1)-np.array(p0)
            dq = p.getDifferenceQuaternion(q1, q0)
            dR = p.getMatrixFromQuaternion(dq)
            trans_init = np.eye(4)
            trans_init[:3,:3] = np.array(dR).reshape(3, 3)
            trans_init[:3,3] = dp
        else:
            trans_init = icp_initialization(pcd0,pcd1)

        Odometry = perform_icp_alignment(pcd1, pcd0, trans_init)

        # DO NOT CONSIDER NON-CONVERGENT ICP SOLUTIONS
        trn = np.linalg.norm(Odometry[:3,3], ord=None, axis=None)
        rot = np.linalg.norm(Odometry[:3,:3]-np.eye(3), 'fro')
        if trn+rot >= 50:
            Odometry = []


    return Odometry


#####################################################################################################################

def perform_icp_alignment(source, target, trans_init):
    """
    Perform ICP (Iterative Closest Point) alignment between source and target point clouds.

    Args:
        source: Open3D point cloud representing the source cloud.
        target: Open3D point cloud representing the target cloud.

    Returns:
        transformation: The transformation matrix that aligns source to target.
    """
    # Set the ICP parameters
    threshold = 0.01
       
    # Perform ICP alignment
    reg_p2l = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria())

    return reg_p2l.transformation

#####################################################################################################################

def icp_initialization(source, target):
    """
    Generate an initial transformation matrix for ICP alignment.

    Args:
        source (o3d.geometry.PointCloud): The source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.

    Returns:
        np.array: A 4x4 initial transformation matrix.
    """

    # Ensure source and target are point clouds
    if not isinstance(source, o3d.geometry.PointCloud) or not isinstance(target, o3d.geometry.PointCloud):
        raise TypeError("Source and target must be open3d.geometry.PointCloud objects")

    # Compute the geometric centers of the point clouds
    source_center = np.mean(np.asarray(source.points), axis=0)
    target_center = np.mean(np.asarray(target.points), axis=0)

    # Compute translation
    translation = target_center - source_center

    # Compute PCA for rotation
    def pca_rotation(cloud):
        # Extracting points as numpy array
        points = np.asarray(cloud.points)
        # Centering points
        centered_points = points - np.mean(points, axis=0)
        # Computing covariance matrix and its eigenvectors
        H = np.dot(centered_points.T, centered_points)
        _, eigenvectors = np.linalg.eigh(H)
        return eigenvectors

    # Get rotation matrices from PCA (Principal Component Analysis)
    source_rot = pca_rotation(source)
    target_rot = pca_rotation(target)

    # Compute the rotation matrix
    rotation_matrix = np.dot(target_rot, source_rot.T)

    # Construct the transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation

    return transformation_matrix