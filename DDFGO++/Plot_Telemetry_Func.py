import numpy as np
import pickle
import gtsam
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d
import os
from pathlib import Path

#############################################################################
# Low-pass filter for smoothing plot data
#############################################################################

def smooth(data, sigma=2):
    """
    Apply a Gaussian low-pass filter to smooth noisy data.
    
    Args:
        data: List or numpy array of values (1D or 2D)
        sigma: Standard deviation of Gaussian kernel (default 2). 
               Higher values = more smoothing. 
               Use None or 0 to disable smoothing and return plain data.
    
    Returns:
        Smoothed numpy array of same shape as input.
        For 2D arrays (N x M), smoothing is applied along axis=0 (by columns).
    
    Usage:
        plt.plot(Time, smooth(noisy_data))          # Default sigma=2
        plt.plot(Time, smooth(noisy_data, 5))       # More smoothing
        plt.plot(Time, smooth(noisy_data, None))    # No smoothing
    """
    # No smoothing if sigma is None or <= 0
    if sigma is None or sigma <= 0:
        return np.array(data) if not isinstance(data, np.ndarray) else data
    
    # Convert to numpy array
    try:
        data = np.array(data, dtype=float)
    except (ValueError, TypeError):
        # Data contains non-numeric or nested objects, return as-is
        try:
            return np.array(data)
        except:
            return data
    
    # Handle based on dimensionality
    if data.ndim == 1:
        if len(data) < 3:
            return data
        # Apply Gaussian filter (handles edges automatically with mode='nearest')
        return gaussian_filter1d(data, sigma=sigma, mode='nearest')
    
    elif data.ndim == 2:
        # Smooth each column independently (along axis=0, i.e., time axis)
        if data.shape[0] < 3:
            return data
        smoothed = np.zeros_like(data)
        for col in range(data.shape[1]):
            smoothed[:, col] = gaussian_filter1d(data[:, col], sigma=sigma, mode='nearest')
        return smoothed
    
    else:
        # Higher dimensional arrays - return as-is
        return data

#############################################################################

def load_variable_from_file(filename):
    try:
        with open(filename, 'rb') as file:
            variable = pickle.load(file)
        return variable
    except FileNotFoundError:
        # Prefer config-driven Results dir when available, fallback to cwd-relative.
        results_dir = Path("./Results")
        try:
            import config

            configured_results_dir = getattr(config, "RESULTS_DIR", None)
            if configured_results_dir:
                results_dir = Path(configured_results_dir)
        except Exception:
            pass
        try:
            available_files = sorted(
                [p.name for p in results_dir.iterdir() if p.is_file()]
            )
        except Exception:
            available_files = []
        print(f"\n\nOutput file '{filename}' not found. \nAvailable files in the Results directory:\n")
        for file in available_files:
            print(file)
        print("\n")
        return None
        
#############################################################################

def find_valid_iterations(Agents_History):
    """Find the number of valid iterations (where State_Estim is not empty)."""
    for i in range(len(Agents_History)):
        state_estim = Agents_History[i][0].get('State_Estim')
        if isinstance(state_estim, list) and len(state_estim) == 0:
            return i
    return len(Agents_History)

def minmax_range(my_array):

    min_value = min(my_array)
    max_value = max(my_array)

    # Step 2: Calculate the range
    data_range = max_value - min_value

    # Step 3: Add a 10% margin
    margin_percent = 0.010
    margin = data_range * margin_percent

    # Step 4: Determine the new minimum and maximum values with the margin
    new_min = min_value - margin
    new_max = max_value + margin

    return new_min, new_max

############################################################################

def call_parameter(Agent, case):
    a = Agent["ID"]+1
    scatter = False

    if case == 1:
        p = Agent["State"][0:3]
        title = 'True Position'+' for agent '+str(a)

    ###########################################################

    elif case == 2:
        state_estim = Agent["State_Estim"]
        p = state_estim.translation()
        title = 'Estimated Position'+' for agent '+str(a)

    ###########################################################
        
    elif case == 3:
        gps_measurements = Agent["State_Obs"]
        p = gps_measurements.translation()[0]
        title = 'Observed x position'
        scatter = True

    elif case == 4:
        gps_measurements = Agent["State_Obs"]
        p = gps_measurements.translation()[1]
        title = 'Observed y position'
        scatter = True

    elif case == 5:
        gps_measurements = Agent["State_Obs"]
        p = gps_measurements.translation()[2]
        title = 'Observed z position'
        scatter = True

    ###########################################################

    elif case == 6:
        gps_measurements = Agent["State_Obs"]
        p = gps_measurements.rotation().toQuaternion().x()
        title = 'Observed QuaternionX'+' for agent '+str(a)
        scatter = True

    elif case == 7:
        gps_measurements = Agent["State_Obs"]
        p = gps_measurements.rotation().toQuaternion().y()
        title = 'Observed QuaternionY'+' for agent '+str(a)
        scatter = True

    elif case == 8:
        gps_measurements = Agent["State_Obs"]
        p = gps_measurements.rotation().toQuaternion().z()
        title = 'Observed QuaternionZ'+' for agent '+str(a)
        scatter = True

    elif case == 9:
        gps_measurements = Agent["State_Obs"]
        p = gps_measurements.rotation().toQuaternion().w()
        title = 'Observed QuaternionW'+' for agent '+str(a)
        scatter = True
        

    ###########################################################
        
    elif case == 10:
        # Legacy: raw sim quaternion vector (may confuse plots expecting scalars).
        p = Agent["State"][6:10]
        title = 'True Quaternion (raw State[6:10] vector)'+' for agent '+str(a)

    elif case == 11:
        # Legacy: packed 4-vector per timestep (not comparable to scalar Case 6–9 series).
        state_estim = Agent["State_Estim"]
        qx = state_estim.rotation().toQuaternion().x()
        qy = state_estim.rotation().toQuaternion().y()
        qz = state_estim.rotation().toQuaternion().z()
        qw = state_estim.rotation().toQuaternion().w()
        p = [qx, qy, qz, qw]
        title = 'Estimated Quaternion (packed xyzw)'+' for agent '+str(a)

    # --- Quaternion scalars: same convention as helper.true_pose (w,x,y,z from State[6:10]=qx,qy,qz,qw) ---
    elif case == 35:
        quat = Agent["State"][6:10]
        r = gtsam.Rot3.Quaternion(float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))
        p = r.toQuaternion().x()
        title = 'True Quaternion x (GTSAM from State)'+' for agent '+str(a)

    elif case == 36:
        quat = Agent["State"][6:10]
        r = gtsam.Rot3.Quaternion(float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))
        p = r.toQuaternion().y()
        title = 'True Quaternion y (GTSAM from State)'+' for agent '+str(a)

    elif case == 37:
        quat = Agent["State"][6:10]
        r = gtsam.Rot3.Quaternion(float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))
        p = r.toQuaternion().z()
        title = 'True Quaternion z (GTSAM from State)'+' for agent '+str(a)

    elif case == 38:
        quat = Agent["State"][6:10]
        r = gtsam.Rot3.Quaternion(float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]))
        p = r.toQuaternion().w()
        title = 'True Quaternion w (GTSAM from State)'+' for agent '+str(a)

    elif case == 39:
        state_estim = Agent["State_Estim"]
        p = state_estim.rotation().toQuaternion().x()
        title = 'Estimated Quaternion x'+' for agent '+str(a)

    elif case == 40:
        state_estim = Agent["State_Estim"]
        p = state_estim.rotation().toQuaternion().y()
        title = 'Estimated Quaternion y'+' for agent '+str(a)

    elif case == 41:
        state_estim = Agent["State_Estim"]
        p = state_estim.rotation().toQuaternion().z()
        title = 'Estimated Quaternion z'+' for agent '+str(a)

    elif case == 42:
        state_estim = Agent["State_Estim"]
        p = state_estim.rotation().toQuaternion().w()
        title = 'Estimated Quaternion w'+' for agent '+str(a)

    elif case == 12:
        p = Agent['Target_Estim'][6:9]
        title = 'Estimated Target Parameter'+' for agent '+str(a)

    elif case == 13:
        p = Agent['Total_FGO_Error']
        title = 'Total FGO Error'+' for agent '+str(a)

    elif case == 14:
        p = Agent['Visible_FGO_Error']
        title = 'Visible FGO Error'+' for agent '+str(a)
        
    elif case == 15:
        p = len(Agent['MapSet'])
        title = 'Map size - Number of features'+' for agent '+str(a)

    elif case == 16:
        map_size = len(Agent['MapSet'])
        p = Agent['Total_FGO_Error'] / map_size if map_size > 0 else 0.0
        title = 'Error-to-Map Size Ratio'+' for agent '+str(a)
    
    elif case == 23:
        merged_map_error = Agent.get('MergedMap_Error', {}) or {}
        p = merged_map_error.get('chamfer_distance', 0.0)
        title = 'Merged Map Chamfer Distance'+' for agent '+str(a)
    
    elif case == 24:
        merged_map_error = Agent.get('MergedMap_Error', {}) or {}
        p = merged_map_error.get('rmse_est_to_gt', 0.0)
        title = 'Merged Map RMSE (est->gt)'+' for agent '+str(a)
    
    elif case == 25:
        merged_map_error = Agent.get('MergedMap_Error', {}) or {}
        p = merged_map_error.get('inlier_ratio', 0.0)
        title = 'Merged Map Inlier Ratio'+' for agent '+str(a)
    
    elif case == 26:
        p = len(Agent.get('MergedMapSet', []))
        title = 'Merged Map size - Number of points'+' for agent '+str(a)

    ###########################################################
    # Loop-closure diagnostics
    ###########################################################

    elif case == 27:
        p = Agent.get('KinLoopClosuresAdded', 0)
        title = 'Kinematic Loop Closures Added (count)'+' for agent '+str(a)

    elif case == 28:
        p = Agent.get('KinLoopClosuresUnique', 0)
        title = 'Kinematic Loop Closures (unique landmark IDs)'+' for agent '+str(a)

    elif case == 29:
        p = Agent.get('ReobsCount', 0)
        title = 'Re-observations in selected features'+' for agent '+str(a)
    
    ###########################################################

    elif case == 17:
        state_estim = Agent["State_Estim"]
        p = state_estim.translation()[0]
        title = 'Estimated x position'
    
    elif case == 18:
        state_estim = Agent["State_Estim"]
        p = state_estim.translation()[1]
        title = 'Estimated y position'
    
    elif case == 19:
        state_estim = Agent["State_Estim"]
        p = state_estim.translation()[2]
        title = 'Estimated z position'
        
    ###########################################################

    elif case == 20:
        p = Agent["State"][0]
        title = 'True x position'
    
    elif case == 21:
        p = Agent["State"][1]
        title = 'True y position'
    
    elif case == 22:
        p = Agent["State"][2]
        title = 'True z position'
        
        
    else:
        raise ValueError(f"Invalid case for parameter call")

    return p, title, scatter



def plot_kinematic_factor_comparison(fig_name, plot_title, Case, A, Agents_History_nkf, Agents_History_1k, Agents_History_nk, num_iter, step_size):

    # Set plotting cases
    c = len(Case)
    Observations = []
    Legend = []
    Scat = []

    ############################################################################
    ############################################################################
    ############################################################################

    for i in range(c):
        Obs = []
        a = A[i]-1
        case = Case[i]
        for j in range(num_iter):
            p , _ , scatter = call_parameter(Agents_History_nkf[j][a], case)
            Obs.append(p)

        Legend.append('No Kinematic Factor')
        Observations.append(Obs)
        Scat.append(scatter)

    ############################################################################

    for i in range(c):
        Obs = []
        a = A[i]-1
        case = Case[i]
        for j in range(num_iter):
            p , _ , scatter = call_parameter(Agents_History_1k[j][a], case)
            Obs.append(p)

        Legend.append('1-step loop closing kinematic factor')
        Observations.append(Obs)
        Scat.append(scatter)

    ############################################################################

    for i in range(c):
        Obs = []
        a = A[i]-1
        case = Case[i]
        for j in range(num_iter):
            p , _ , scatter = call_parameter(Agents_History_nk[j][a], case)
            Obs.append(p)

        Legend.append('N-step loop closing kinematic factor')
        Observations.append(Obs)
        Scat.append(scatter)

    ############################################################################
    ############################################################################
    ############################################################################
    dt = 1/240
    Time = []
    for j in range(num_iter):
        Time.append(j*dt*step_size)

    ############################################################################
    # Plot Data

    plt.figure(fig_name)
    plt.title(plot_title)

    # Set labels for x and y axis
    plt.xlabel("Time")

    # Use a for loop to plot each array in Observations and add legend names
    for i, observation_array in enumerate(Observations):
        if Scat[i]:
            plt.scatter(Time, observation_array, label=Legend[i], s = 3, color = 'k', marker = '.')
        else:
            plt.plot(Time, observation_array, label=Legend[i])

    # Show the legend in the plot
    plt.legend()

    # Show the plot
    plt.show()


def plot_kinematic_factor_comparison_4(fig_name, plot_title, Case, A, Agents_History_nkf, Agents_History_1k, Agents_History_mk, Agents_History_nk, num_iter, step_size):

    # Set plotting cases
    c = len(Case)
    Observations = []
    Legend = []
    Scat = []

    ############################################################################
    ############################################################################
    ############################################################################

    for i in range(c):
        Obs = []
        a = A[i]-1
        case = Case[i]
        for j in range(num_iter):
            p , _ , scatter = call_parameter(Agents_History_nkf[j][a], case)
            Obs.append(p)

        Legend.append('No Kinematic Factor')
        Observations.append(Obs)
        Scat.append(scatter)

    ############################################################################

    for i in range(c):
        Obs = []
        a = A[i]-1
        case = Case[i]
        for j in range(num_iter):
            p , _ , scatter = call_parameter(Agents_History_1k[j][a], case)
            Obs.append(p)

        Legend.append('n = 1')
        Observations.append(Obs)
        Scat.append(scatter)

    ############################################################################

    for i in range(c):
        Obs = []
        a = A[i]-1
        case = Case[i]
        for j in range(num_iter):
            p , _ , scatter = call_parameter(Agents_History_mk[j][a], case)
            Obs.append(p)

        Legend.append('n = 10')
        Observations.append(Obs)
        Scat.append(scatter)
    
    ############################################################################

    for i in range(c):
        Obs = []
        a = A[i]-1
        case = Case[i]
        for j in range(num_iter):
            p , _ , scatter = call_parameter(Agents_History_nk[j][a], case)
            Obs.append(p)

        Legend.append('n = 20')
        Observations.append(Obs)
        Scat.append(scatter)

    ############################################################################
    ############################################################################
    ############################################################################
    dt = 1/240
    Time = []
    for j in range(num_iter):
        Time.append(j*dt*step_size)

    ############################################################################
    # Plot Data

    plt.figure(fig_name)
    plt.title(plot_title, fontweight='bold', fontsize=20)

    # Set labels for x and y axis
    plt.xlabel("Time (s)", fontweight='bold', fontsize=12)
    plt.ylabel("Error", fontweight='bold', fontsize=12)

    # Use a for loop to plot each array in Observations and add legend names
    for i, observation_array in enumerate(Observations):
        if Scat[i]:
            plt.scatter(Time, observation_array, label=Legend[i], s = 3, color = 'k', marker = '.')
        else:
            plt.plot(Time, observation_array, label=Legend[i])

    # Show the legend in the plot
    plt.legend(loc = 'best', fontsize=15)

    # Show the plot
    plt.show()


def plot_pose_graph_optimization(fig_name, Title, Case, A, Agents_History, num_iter, step_size, sigma=2):
    """
    Plot pose graph optimization results.
    
    Args:
        sigma: Gaussian filter sigma (default 2). Use 0 to disable smoothing.
        For quaternion component series, prefer sigma=0: smoothing each component
        independently is not a valid rotation filter and can mimic large observation error.
    """
    # Set plotting cases
    c = len(Case)
    Observations = []
    Legend = []
    Scat = []

    for i in range(c):
        Obs = []
        a = A[i]-1
        case = Case[i]
        for j in range(num_iter):
            p , _ , scatter = call_parameter(Agents_History[j][a], case)
            Obs.append(p)
            _ , legend_title , _ = call_parameter(Agents_History[j][a], case)
        
                
        Legend.append(legend_title)
        Observations.append(Obs)
        Scat.append(scatter)
    
    
    
    ###############
    
    
    ############################################################################
    dt = 1/240
    Time = []
    for j in range(num_iter):
        Time.append(j*dt*step_size)

    ############################################################################
    # Plot Data

    plt.figure(fig_name)
    plt.title(Title, fontweight='bold', fontsize=20)

    # Set labels for x and y axis
    plt.xlabel("Time (s)", fontweight='bold', fontsize=12)
    plt.ylabel("Quaternion", fontweight='bold', fontsize=12)
    
    # Use a for loop to plot each array in Observations and add legend names
    for i, observation_array in enumerate(Observations):
        if Scat[i]:
            plt.scatter(Time, observation_array, label=Legend[i], s = 3, color = 'k', marker = '.')
        else:
            # Apply Gaussian smoothing filter
            smoothed_obs = smooth(observation_array, sigma)
            # Choose color by legend text, with safe fallback
            color = None
            if 'True' in Legend[i]:
                color = '#1f77b4'
            elif 'Estimated' in Legend[i]:
                color = '#FF4500'
            if color is not None:
                plt.plot(Time, smoothed_obs, label=Legend[i], color=color)
            else:
                plt.plot(Time, smoothed_obs, label=Legend[i])
        
        
        
    ################# Show fewer legends in the legend in the plot
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', lw=2, label='True'),
        Line2D([0], [0], color='#FF4500', lw=2, label='Estimated'),
        Line2D([0], [0], marker='o', color='k', markerfacecolor='k', lw=0, markersize=3, label='Observed')  # Point for scatter
        ]
    
    ################# Add the manually defined legend
    plt.legend(handles=legend_elements, loc = 'best')
    
    
    
    # show legends for all plot entries
    # plt.legend()
    
    # Show the plot
    plt.show()




