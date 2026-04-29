import numpy as np
import gtsam
from gtsam import symbol_shorthand
from scipy.linalg import logm
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import savgol_filter


__all__ = ['varX', 'varL', 'varP', 'varV', 'varW',
           'add_noise_to_pose', 'true_pose',
           'measure_target_params', 'extract_agent_history',
           'squared_error_pose', 'squared_error_landmarks',
           'squared_error_visible_landmarks',
           'total_fgo_error', 'visible_fgo_error', 'iskeyingraph',
           'Cartesian2BearingRange3D', 'BearingRange2Cartesian3D',
           'add_noise_to_Unit3',
           'find_descriptor_in_keyframes', 'extract_keyframes',
           'forward_kinematics',
           'forward_covariance', 'Gaussian_Entropy',
           'calculate_com', 'calculate_v', 'calculate_v_savgol',
           'calculate_w1', 'calculate_w2',
           'calculate_w3', 'calculate_w4',
           'extract_landmarks', 'marginalize_factor_graph',
           'low_pass_filter', 'low_pass_filter_pose']


#############################################################################################################################
# Functions
##############################################################################################################################

def true_pose(Agents_History, a, i):
    pos  = Agents_History[i][a]['State'][:3]
    quat = Agents_History[i][a]['State'][6:10]
    rotation = gtsam.Rot3.Quaternion(quat[3],quat[0],quat[1],quat[2])
    translation = gtsam.Point3(pos[0], pos[1], pos[2])
    pose = gtsam.Pose3(rotation, translation)
    return pose

def add_noise_to_pose(pose, translation_noise_std, rotation_noise_std):
    """
    Add noise directly on the manifold to a pose.
    pose: gtsam.Pose3 object representing the original pose.
    translation_noise_std: Standard deviation of translational Gaussian noise.
    rotation_noise_std: Standard deviation of rotational Gaussian noise (in radians).
    """

    # Generate small translational noise
    translation_noise = np.random.normal(0, translation_noise_std, 3)
    delta_translation = gtsam.Point3(translation_noise[0],translation_noise[1],translation_noise[2])


    # Generate small rotational perturbation in tangent space (axis-angle form)
    rotation_noise = np.random.normal(0, np.deg2rad(rotation_noise_std), 3)
    rotation_noise_magnitude = np.linalg.norm(rotation_noise)
    if rotation_noise_magnitude > 0:
        # Convert axis-angle to quaternion
        delta_rotation = gtsam.Rot3.AxisAngle(rotation_noise / rotation_noise_magnitude, rotation_noise_magnitude)
    else:
        # No rotation
        delta_rotation = gtsam.Rot3.Quaternion(1, 0, 0, 0)

    # Apply noise to pose
    noise = gtsam.Pose3(delta_rotation, delta_translation)
    pose = pose.compose(noise)


    return pose

def extract_agent_history(a,Agents_History):
    Agent_History = []
    for i in range(len(Agents_History)):
        Agent = Agents_History[i][a]
        Agent_History.append(Agent)
    return Agent_History

def squared_error_pose(Agent):
    pos = np.array(Agent['State'][:3])
    pos_est = Agent['State_Estim'].translation()

    quat = np.array(Agent['State'][6:10])
    quat_est = np.array([Agent['State_Estim'].rotation().toQuaternion().x(),
                        Agent['State_Estim'].rotation().toQuaternion().y(),
                        Agent['State_Estim'].rotation().toQuaternion().z(),
                        Agent['State_Estim'].rotation().toQuaternion().w()])
    
    error_pos  = pos - pos_est
    error_quat = quat - quat_est
    e = np.hstack((error_pos, error_quat))

    return np.linalg.norm(e)

def squared_error_landmarks(Agent, TargetPC):
    Map = Agent['MapSet']
    MapIdx = Agent['MapIdxSet']

    e = 0
    for l in range(len(Map)):
        predicted = Map[l]
        descriptor = MapIdx[l]
        true = TargetPC[descriptor]
        error = predicted - true
        e += np.linalg.norm(error)

    return e

def squared_error_visible_landmarks(Agent, TargetPC):
    Features = Agent['FeatureSet']
    FeaturesIdx = Agent['FeatureIdxSet']

    e = 0
    for l in range(len(Features)):
        predicted = Features[l]
        descriptor = FeaturesIdx[l]
        true = TargetPC[descriptor]
        error = predicted - true
        e += np.linalg.norm(error)

    return e

def total_fgo_error(Agent, TargetPC):
    error_pose = squared_error_pose(Agent)
    error_landmarks = squared_error_landmarks(Agent, TargetPC)
    error = error_pose + error_landmarks
    return error

def visible_fgo_error(Agent, TargetPC):
    error_pose = squared_error_pose(Agent)
    error_landmarks = squared_error_visible_landmarks(Agent, TargetPC)
    error = error_pose + error_landmarks
    return error

def iskeyingraph(graph, key):
    """
    Check if a key exists in the graph as a variable.

    Parameters:
        graph (gtsam.NonlinearFactorGraph): The factor graph.
        key: The key to check for existence.

    Returns:
        bool: True if the key exists in the graph as a variable, False otherwise.
    """
    # Iterate through all factors in the graph
    for i in range(graph.size()):
        # Get the keys associated with this factor
        keys = graph.at(i).keys()
        
        # Check if the key exists in the keys associated with this factor
        if key in keys:
            return True
    
    # If the key is not found in any factor, return False
    return False

def Cartesian2BearingRange3D(state_obs, feature):
    pos = state_obs.translation()
    rel = feature - pos
    range_obs = float(np.linalg.norm(rel))

    # extract rotation matrix of agent (or extract the three orthonormal orthogonal axis - x-y-z)
    rot_matrix = state_obs.rotation().matrix()

    # rotate relative position from global to local reference frame
    rel = np.dot(rot_matrix.T, rel)

    # Normalize to get unit vector
    norm = np.linalg.norm(rel)
    bearing_obs = gtsam.Unit3(gtsam.Point3(rel/norm))

    return bearing_obs, range_obs

def BearingRange2Cartesian3D(state_obs, bearing_obs, range_obs):
    # # Convert Unit3 bearing to a vector in the local reference frame
    bearing_vector = bearing_obs.point3()
   
    # Scale by range to get the relative position in the local frame
    rel_pos_local = bearing_vector * range_obs
   
    # Rotate relative position from local to global reference frame
    rot_matrix = state_obs.rotation().matrix()
    rel_pos_global = np.dot(rot_matrix, rel_pos_local)
   
    # Add the observer's position to get the global position of the feature
    pos = state_obs.translation()
    feature_global = pos + rel_pos_global

    return feature_global

def add_noise_to_Unit3(bearing, bearing_sigma):
    # Convert noise from degrees to radians
    bearing_sigma = np.radians(bearing_sigma)
   
    # Generate random noise in three dimensions
    noise = np.random.normal(0, bearing_sigma, 3)
   
    # Retrieve the original bearing vector
    original_vector = bearing.point3()
   
    # Create a small rotation matrix from the noise
    noise_rotation = gtsam.Rot3.Rodrigues(noise[0], noise[1], noise[2])
   
    # Apply the rotation to the original vector
    noisy_vector = noise_rotation.rotate(original_vector)
   
    # Normalize and convert back to gtsam.Unit3
    noisy_bearing = gtsam.Unit3(noisy_vector)
   
    return noisy_bearing


def extract_keyframes(Frames_Hist, a): # For ISAM2 solver incremental run    
    Frames = []
    for frame in Frames_Hist:
        Frames.append(frame[a])
    return Frames


def find_descriptor_in_keyframes(descriptor, KeyFramesIdx):
    # Check condition for feature in map to exist in keyframes
    exist = False
    m = None
    for j in range(len(KeyFramesIdx)-1, -1, -1): # reverse sweeping
        frameIdx = KeyFramesIdx[j]
        if len(frameIdx) != 0:
            for m in range(len(frameIdx)):
                if descriptor in frameIdx:
                    exist = True
                    break

        if exist: break
    
    return exist, j, m

def forward_kinematics(pos, num_steps, step_size, com, vel, ang_vel):
    dt = 1/240
    DeltaT = dt*step_size*num_steps
    
    pos -= com
    pos += ( np.cross(ang_vel, pos) + vel ) * DeltaT
    pos += com

    return pos


def measure_target_params(Target, target_noise_std, target_noise_bias):
    
    # Extract true target kinematic parameters
    com = Target[:3]
    vel = Target[3:6]
    ang_vel = Target[10:13]

    # Generate noisy variations
    com_noise = np.random.normal(target_noise_bias[0], target_noise_std[:3], 3)
    vel_noise = np.random.normal(target_noise_bias[1], target_noise_std[3:6], 3)
    ang_vel_noise = np.random.normal(target_noise_bias[2], target_noise_std[6:9], 3)
    
    # Add the biased noise to the estimates
    com += com_noise
    vel += vel_noise
    ang_vel += ang_vel_noise

    return np.concatenate((com,vel,ang_vel))


def forward_covariance(Sigma_0, num_steps, delta_t, omega, noise):
    """
    Propagates the covariance matrix over multiple time steps using a constant velocity rigid body model.
    
    Parameters:
        Sigma_0: Initial covariance matrix (3x3 numpy array).
        omega: Angular velocity vector (3x1 numpy array).
        v: Translational velocity vector (3x1 numpy array).
        delta_t: Time step size (float).
        Q: noise magnitude (float) for process noise covariance matrix (3x3 numpy array).
        num_steps: Number of time steps to propagate.
        
    Returns:
        Sigma_t: The propagated covariance matrix after num_steps.
    """
    # Identity matrix for the position part
    I = np.eye(3)
    
    # Jacobian of the cross product term (rotation part)
    def jacobian_cross_product(omega):
        J_w_cross = np.array([[0, -omega[2], omega[1]],
                              [omega[2], 0, -omega[0]],
                              [-omega[1], omega[0], 0]])
        return J_w_cross
    
    # Initialize the covariance matrix
    Sigma_t = Sigma_0
    
    # Propagate over multiple steps
    for _ in range(num_steps):
        # Jacobian of the system at the current step
        J_omega_cross = jacobian_cross_product(omega)
        
        # Full Jacobian for the rigid body motion
        F = I + J_omega_cross * delta_t
        
        
        
        # Translational process noise covariance (uncertainty in velocity in meters per second)
        trans_noise_std = noise  # Standard deviation of translational noise (in meters/second)
        Q_trans = np.diag([trans_noise_std**2, trans_noise_std**2, trans_noise_std**2])  # Variance in each direction
        
        # Rotational process noise covariance (uncertainty in angular velocity in radians per second)
        rot_noise_std = noise/5  # Standard deviation of rotational noise (in radians/second)
        Q_rot = np.diag([rot_noise_std**2, rot_noise_std**2, rot_noise_std**2])  # Variance in each direction
        
        # Total process noise covariance matrix
        Q = Q_trans + Q_rot

        # Covariance propagation equation
        Sigma_t = F @ Sigma_t @ F.T + Q
    
    return Sigma_t

    
    
    
def Gaussian_Entropy(Sigma):
    """
    Calculates the entropy of a Gaussian distribution with covariance matrix Sigma.
    
    Parameters:
        Sigma: Covariance matrix (3x3 numpy array or any NxN array).
        
    Returns:
        Entropy (float): The entropy of the Gaussian distribution.
    """
    d = Sigma.shape[0]  # Dimensionality of the distribution
    
    Sigma_reg = Sigma + 5e-2 * np.eye(Sigma.shape[0])
    
    det_Sigma = np.linalg.det(Sigma)  # Determinant of the covariance matrix
    
    # Calculate the entropy using the Gaussian entropy formula
    entropy = 0.5 * np.log(det_Sigma * (2 * np.pi * np.e) ** d)
    
    return entropy

def calculate_com(Points):
    """
    Calculates the center of mass (COM) given Points as a np.array of shape (n, d),
    where n is the number of points and d is the dimension.
    
    Args:
    Points (np.array): Array of shape (n, d) representing positions of points.
    
    Returns:
    np.array: Center of mass as a 1D array of shape (d,).
    """
    
    p = np.mean(Points, axis=0)
    
    return p

def calculate_v(Points, Points_previous, delta_t):
    """
    Calculates the average velocity between two sets of points (current and previous).
    
    Args:
    Points (np.array): Current positions of points, array of shape (n, d).
    Points_previous (np.array): Previous positions of points, array of shape (n, d).
    delta_t (float): Time step between current and previous positions.
    
    Returns:
    np.array: Average velocity as a 1D array of shape (d,).
    """    
    com_current = calculate_com(Points)
    # print(f'com_current {com_current}')
    
    com_previous = calculate_com(Points_previous)
    # print(f'com_previous {com_previous}')
    
    velocity = (com_current - com_previous) / delta_t
    # print(f'velocity {velocity}')
    
    return velocity


def calculate_v_savgol(Agents_History, time_step=0.1, window_length=5, polyorder=2):
    """
    Computes velocity using the Savitzky-Golay filter.

    :param past_positions: List or numpy array of past position estimates (1D or 2D for multiple dimensions)
    :param time_step: Time step between successive position estimates (default: 0.1)
    :param window_length: Number of points used for local polynomial fitting (should be odd)
    :param polyorder: Polynomial order for fitting (should be < window_length)
    :return: Estimated velocity at the latest time step
    """

    past_positions = [Agent['Target_COM'] for Agent in Agents_History]
    past_positions = np.asarray(past_positions)

    if past_positions.shape[0] < window_length:
        return None  # Not enough data to compute velocity

    if window_length % 2 == 0:
        window_length += 1  # Ensure window_length is odd

    # Apply Savitzky-Golay filter independently to each dimension
    velocity = np.apply_along_axis(
        lambda x: savgol_filter(x, window_length, polyorder, deriv=1, delta=time_step),
        axis=0,
        arr=past_positions
    )

    return velocity[-1]  # Return the most recent velocity estimate


def calculate_w1(Points, Points_previous, delta_t):
    """
    Calculates the rotational component (angular velocity-like measure).
    
    Args:
    Points (np.array): Current positions of points, array of shape (n, d).
    Points_previous (np.array): Previous positions of points, array of shape (n, d).
    delta_t (float): Time step between current and previous positions.
    
    Returns:
    np.array: Rotational component as a 1D array of shape (d,).
    """
    n = Points.shape[0]
    com_current = calculate_com(Points)
    com_previous = calculate_com(Points_previous)
    
    relative_current = Points - com_current
    relative_previous = Points_previous - com_previous
    change_in_relative = (relative_current - relative_previous) / delta_t
    
    rotational_component = np.mean(np.cross(relative_previous, change_in_relative), axis=0)
    return rotational_component


def calculate_w2(Points, Points_previous, delta_t):
    """
    Calculates the angular velocity based on point positions over time using
    least square method
    
    Args:
    Points (np.array): Current positions of points, array of shape (n, d).
    Points_previous (np.array): Previous positions of points, array of shape (n, d).
    delta_t (float): Time step between current and previous positions.
    
    Returns:
    np.array: Angular velocity as a 1D array of shape (d,).
    """
    # Calculate the center of mass for current and previous points
    R_COM = np.mean(Points, axis=0)
    R_COM_previous = np.mean(Points_previous, axis=0)

    # Relative positions of points to the center of mass at current and previous times
    r_prime = Points - R_COM
    r_prime_previous = Points_previous - R_COM_previous

    # Calculate the velocity of each landmark relative to the COM
    v_prime = (r_prime - r_prime_previous) / delta_t

    # Set up matrix A and vector b for least-squares estimation
    A = []
    b = []

    for i in range(Points.shape[0]):
        # Cross product matrix for r_prime
        C_i = np.array([
            [0, -r_prime[i, 2], r_prime[i, 1]],
            [r_prime[i, 2], 0, -r_prime[i, 0]],
            [-r_prime[i, 1], r_prime[i, 0], 0]
        ])

        A.append(C_i)
        b.append(v_prime[i])

    # Convert A and b to arrays
    A = np.vstack(A)
    b = np.hstack(b)

    # Solve for angular velocity using least-squares
    omega, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    return omega


def calculate_w3(Points, Points_previous, delta_t):
    """
    Calculates the angular velocity using an analytical approach based on point positions over time.
    
    Args:
    Points (np.array): Current positions of points, array of shape (n, d).
    Points_previous (np.array): Previous positions of points, array of shape (n, d).
    delta_t (float): Time step between current and previous positions.
    
    Returns:
    np.array: Angular velocity as a 1D array of shape (d,).
    """
    # Calculate the center of mass for current and previous points
    R_COM = np.mean(Points, axis=0)
    R_COM_previous = np.mean(Points_previous, axis=0)

    # Relative positions of points to the center of mass at current and previous times
    r_prime = Points - R_COM
    r_prime_previous = Points_previous - R_COM_previous

    # Calculate the velocity of each landmark relative to the COM
    v_prime = (r_prime - r_prime_previous) / delta_t

    # Compute the numerator (sum of cross products)
    cross_sum = np.sum(np.cross(r_prime, v_prime), axis=0)

    # Compute the denominator (sum of squared magnitudes of r_prime)
    magnitude_sum = np.sum(np.linalg.norm(r_prime, axis=1)**2)

    # Calculate the angular velocity
    omega = cross_sum / magnitude_sum

    return omega


def calculate_w4(Points, Points_previous, delta_t):
    """
    Calculates the angular velocity using a hybrid analytical-numerical approach
    based on orientation matrices and singular value decomposition.
    
    Args:
    Points (np.array): Current positions of points, array of shape (n, d).
    Points_previous (np.array): Previous positions of points, array of shape (n, d).
    delta_t (float): Time step between current and previous positions.
    
    Returns:
    np.array: Angular velocity as a 1D array of shape (d,).
    """
    # Center the points by subtracting their respective center of mass
    R_COM = np.mean(Points, axis=0)
    R_COM_previous = np.mean(Points_previous, axis=0)

    Points_centered = Points - R_COM
    Points_previous_centered = Points_previous - R_COM_previous
    
    # Debugging output to confirm shapes
    # print('\n')
    # print("Points_centered shape:", Points_centered.shape)
    # print("Points_previous_centered shape:", Points_previous_centered.shape)
    
    # Compute the cross-dispersion matrix
    H = Points_previous_centered.T @ Points_centered

    # Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(H)

    # Calculate the rotation matrix R
    R = Vt.T @ U.T

    # Ensure the rotation matrix is proper (determinant = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Calculate the matrix logarithm of R
    R_log = logm(R)

    # Extract the skew-symmetric part
    omega_skew = (R_log - R_log.T) / 2

    # Extract the angular velocity vector from the skew-symmetric matrix
    omega = np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]]) / delta_t

    return omega


    
def extract_landmarks(Map, MapIdx, Map_previous, MapIdx_previous):
    # Create sets of descriptors for easy lookup
    idx_set = set(MapIdx)
    idx_prev_set = set(MapIdx_previous)

    # Find the intersection of the descriptors
    common_indices = idx_set.intersection(idx_prev_set)

    # Extract the corresponding points from Map and Map_previous
    overlapping_points = [Map[i] for i, idx in enumerate(MapIdx) if idx in common_indices]
    overlapping_points_previous = [Map_previous[i] for i, idx in enumerate(MapIdx_previous) if idx in common_indices]

    return overlapping_points, overlapping_points_previous

    # # Example usage
    # Map = [(1, 2), (3, 4), (5, 6), (7, 8)]
    # MapIdx = [101, 102, 103, 104]
    # Map_previous = [(2, 3), (4, 5), (6, 7), (8, 9)]
    # MapIdx_previous = [103, 104, 105, 106]

    # result, result_previous = extract_landmarks(Map, MapIdx, Map_previous, MapIdx_previous)
    # print(result)
    # print(result_previous)



#########################################################################
#########################################################################
#########################################################################
#########################################################################


from gtsam import NonlinearFactorGraph, Values, Marginals, Symbol

def varX(X, a, i):
    a += 1
    return X(int(f"{a:02}{i:05}"))

def varL(L, descriptor, i):
    return L(int(f"{descriptor:05}{i:05}"))

def varP(P, a, i):
    a += 1
    return P(int(f"{a:02}{i:05}"))

def varV(V, a, i):
    a += 1
    return V(int(f"{a:02}{i:05}"))

def varW(W, a, i):
    a += 1
    return W(int(f"{a:02}{i:05}"))

def get_time_step_from_key(key):
    """
    Extracts the time step from a Symbol key that encodes both the agent ID and time step.
    """
    symbol = gtsam.Symbol(key)
    return symbol.index() % 100000  # Assumes the last 5 digits represent the time step

def marginalize_factor_graph(graph, values, sw, i):
    """
    Marginalizes out variables in the factor graph and values object that have indices beyond the sliding window (sw).

    Parameters:
    - graph (NonlinearFactorGraph): The factor graph to be modified.
    - values (Values): The values object to be modified.
    - sw (int): The sliding window threshold index.

    Returns:
    - updated_graph (NonlinearFactorGraph): The updated factor graph after marginalization.
    - updated_values (Values): The updated values object after marginalization.
    """
    keys_to_marginalize = []
    for key in values.keys():
        time_step = get_time_step_from_key(key)
        # print(f'key: {key}')
        # print(f'time_step: {time_step}')
        # print(f'i - sw: {i} - {sw} = {i - sw}')
        # print('\n')
        if time_step < i - sw:  # Check if the time step is beyond the sliding window    
            keys_to_marginalize.append(key)

    if not keys_to_marginalize:
        print('No keys to marginalize')  # No variables to marginalize
        return graph, values

    # Create a new factor graph and add the existing factors that do not reference the marginalized variables
    updated_graph = gtsam.NonlinearFactorGraph()
    for idx in range(graph.size()):
        factor = graph.at(idx)
        if not any(key in factor.keys() for key in keys_to_marginalize):
            updated_graph.add(factor)

    # Create a new values object with variables within the sliding window only
    updated_values = gtsam.Values()
    for key in values.keys():
        if key not in keys_to_marginalize:
            # Check the type of the value and use the appropriate method
            if values.exists(key):  # Ensure the key exists before accessing it
                try:
                    value = values.atPose3(key)
                    updated_values.insert(key, value)
                except RuntimeError:
                    try:
                        value = values.atPoint3(key)
                        updated_values.insert(key, value)
                    except RuntimeError:
                        print(f"Key {key} has an unsupported type or cannot be accessed.")
                        # Handle other types or log an error if necessary
    
    return updated_graph, updated_values



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

    

def low_pass_filter_pose(prev_pose: gtsam.Pose3, new_pose: gtsam.Pose3, alpha: float) -> gtsam.Pose3:
    """
    Applies a low-pass filter using LERP for translation and SLERP for rotation.
    """
    if prev_pose is None:
        return new_pose  # No filtering if there is no previous estimate

    # --- 1. Interpolate Translation (LERP) ---
    prev_translation = prev_pose.translation()
    new_translation = new_pose.translation()
    filtered_translation = (1 - alpha) * prev_translation + alpha * new_translation

    # --- 2. Interpolate Rotation (SLERP) ---
    # Extract quaternion components into a NumPy array
    prev_quat = np.array([
        prev_pose.rotation().toQuaternion().x(),
        prev_pose.rotation().toQuaternion().y(),
        prev_pose.rotation().toQuaternion().z(),
        prev_pose.rotation().toQuaternion().w()
    ])
    
    new_quat = np.array([
        new_pose.rotation().toQuaternion().x(),
        new_pose.rotation().toQuaternion().y(),
        new_pose.rotation().toQuaternion().z(),
        new_pose.rotation().toQuaternion().w()
    ])

    # Convert to Scipy Rotation object
    prev_rot = Rotation.from_quat(prev_quat)  # Scipy uses (x, y, z, w) order
    new_rot = Rotation.from_quat(new_quat)

    # Use Scipy's Slerp
    slerp = Slerp([0, 1], Rotation.from_quat([prev_quat, new_quat]))  # Define interpolation
    filtered_rot = slerp(alpha)  # Get interpolated rotation

    # Convert back to GTSAM Rot3
    filtered_quat = filtered_rot.as_quat()  # Returns (x, y, z, w)
    filtered_rotation = gtsam.Rot3.Quaternion(
        filtered_quat[3], filtered_quat[0], filtered_quat[1], filtered_quat[2]  # Convert back to (w, x, y, z) for GTSAM
    )

    # --- 3. Construct the new Pose3 ---
    filtered_pose = gtsam.Pose3(filtered_rotation, filtered_translation)
    return filtered_pose
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
