import numpy as np
import Plot_Telemetry_Func as Telemetry
import open3d as o3d
import Load_Target as lt
import matplotlib.pyplot as plt
import time
import gtsam
from gtsam import symbol_shorthand
import pickle
import config

# Select the solver to plot results for:

#############################################################################################
# Load configuration from config.py
#############################################################################################
DT = config.DT
N = config.N
D = config.D
object_name = config.object_name
Kinem = config.Kinem

# Get results path from config
results_paths = config.get_results_paths()
path_agents_nk = results_paths['agents']

# Load Simulation History
# Note: To compare different Kinem settings, you can temporarily override Kinem in config.py
# or load multiple files manually:
# path_agents_nkf = 'Results/Agents_History_'+config.get_tag()+'_No_Kinem.pkl'
# path_agents_1k = 'Results/Agents_History_'+config.get_tag()+'_1_step_Kinem.pkl'
# path_agents_mk = 'Results/Agents_History_'+config.get_tag()+'_m_step_Kinem.pkl'
# Agents_History_nkf = Telemetry.load_variable_from_file(path_agents_nkf)
# Agents_History_1k = Telemetry.load_variable_from_file(path_agents_1k)
# Agents_History_mk = Telemetry.load_variable_from_file(path_agents_mk)
Agents_History_nk = Telemetry.load_variable_from_file(path_agents_nk)

# Truncate to valid iterations only (handles partially-saved histories)
valid_iter = Telemetry.find_valid_iterations(Agents_History_nk)
if valid_iter < len(Agents_History_nk):
    print(f'Warning: Truncating data from {len(Agents_History_nk)} to {valid_iter} valid iterations')
    Agents_History_nk = Agents_History_nk[:valid_iter]

N = len(Agents_History_nk[0]) # Number of agents
num_iter = len(Agents_History_nk)

print(f'\nNumber of agents: {N}')
print(f'Number of iterations: {num_iter}')

# Retrieve step size used for downsampling the data from 1/240 time step
step_size = (D*DT)/(num_iter-1)
print(f'Step_Size: {round(step_size)}\n')


##############################################################################################################################
# Plot
##############################################################################################################################
# Select Agent to show results for: Agent 1 for first agent
a = 1
fig_name = "Batch Factor Graph Estimation"

# Enable focused sparse-vs-dense diagnostics for one agent.
enable_sparse_dense_diagnostics = True


#######################################################################################################################
#######################################################################################################################
# %%
# # Position
Case = [20,21,22,3,4,5,17,18,19]
A = a*np.array([1,1,1,1,1,1,1,1,1])
plot_title = 'Pose Evolution: Position'
Telemetry.plot_pose_graph_optimization(fig_name, plot_title, Case, A, Agents_History_nk, num_iter, step_size)

# %%
# # Orientation
Case = [6,7,8,9,10,11]
A = a*np.array([1,1,1,1,1,1])
plot_title = 'Pose Evolution: Orientation'
Telemetry.plot_pose_graph_optimization(fig_name, plot_title, Case, A, Agents_History_nk, num_iter, step_size)


# %% Target Estimation
Case = [12]
A = a*np.array([1])
plot_title = 'Target Parameter Estimation'
Telemetry.plot_pose_graph_optimization(fig_name, plot_title, Case, A, Agents_History_nk, num_iter, step_size)


# # %% Error Evolution
# Case = 13 * np.ones(N)
# A = [a for a in range(1, N + 1)]
# plot_title = 'Total FGO Error'
# Telemetry.plot_pose_graph_optimization(fig_name, plot_title, Case, A, Agents_History_nk, num_iter, step_size)


# %% Map Size Evolution
Case = 15 * np.ones(N)
A = [a for a in range(1, N + 1)]
plot_title = 'Map Size Evolution'
Telemetry.plot_pose_graph_optimization(fig_name, plot_title, Case, A, Agents_History_nk, num_iter, step_size)


# %% Merged Map Size Evolution
Case = 26 * np.ones(N)
A = [a for a in range(1, N + 1)]
plot_title = 'Merged Map Size Evolution'
Telemetry.plot_pose_graph_optimization(fig_name, plot_title, Case, A, Agents_History_nk, num_iter, step_size)



# %% Error to Map size Evolution
# Case = 16 * np.ones(N)
# A = [a for a in range(1, N + 1)]
# plot_title = 'Total FGO Error'
# Telemetry.plot_pose_graph_optimization(fig_name, plot_title, Case, A, Agents_History_nk, num_iter, step_size)


# # %% Error Evolution
# Case = 14 * np.ones(N)
# A = [a for a in range(1, N + 1)]
# plot_title = 'Visible FGO Error'
# Telemetry.plot_pose_graph_optimization(fig_name, plot_title, Case, A, Agents_History_nk, num_iter, step_size)


# %% Merged Map Error Evolution - Chamfer Distance
Case = 23 * np.ones(N)
A = [a for a in range(1, N + 1)]
plot_title = 'Merged Map Chamfer Distance'
Telemetry.plot_pose_graph_optimization(fig_name, plot_title, Case, A, Agents_History_nk, num_iter, step_size)


# %% Merged Map Error Evolution - RMSE
Case = 24 * np.ones(N)
A = [a for a in range(1, N + 1)]
plot_title = 'Merged Map RMSE (est->gt)'
Telemetry.plot_pose_graph_optimization(fig_name, plot_title, Case, A, Agents_History_nk, num_iter, step_size)


# %% Merged Map Error Evolution - Inlier Ratio
Case = 25 * np.ones(N)
A = [a for a in range(1, N + 1)]
plot_title = 'Merged Map Inlier Ratio'
Telemetry.plot_pose_graph_optimization(fig_name, plot_title, Case, A, Agents_History_nk, num_iter, step_size)


# %% Loop-closure diagnostics - Kinematic factors added
Case = 27 * np.ones(N)
A = [a for a in range(1, N + 1)]
plot_title = 'Kinematic Loop Closures Added (count)'
Telemetry.plot_pose_graph_optimization(fig_name, plot_title, Case, A, Agents_History_nk, num_iter, step_size)


# %% Loop-closure diagnostics - Unique landmark IDs in kinematic closures
Case = 28 * np.ones(N)
A = [a for a in range(1, N + 1)]
plot_title = 'Kinematic Loop Closures (unique landmark IDs)'
Telemetry.plot_pose_graph_optimization(fig_name, plot_title, Case, A, Agents_History_nk, num_iter, step_size)


#######################################################################################################################
# Sparse-vs-dense diagnostics (why spheres disappear while dense map exists)
#######################################################################################################################
if enable_sparse_dense_diagnostics:
    agent_idx = int(a) - 1
    t = np.arange(num_iter) * step_size / DT

    feature_counts = []
    map_counts = []
    map_own_counts = []
    map_ngh_counts = []
    merged_counts = []
    reobs_counts = []
    outlier_dist_counts = []
    outlier_stale_counts = []
    map_growth = []

    for k in range(num_iter):
        ag = Agents_History_nk[k][agent_idx]

        feature_set = np.asarray(ag.get('FeatureSet', []))
        map_set = np.asarray(ag.get('MapSet', []))
        map_ngh = np.asarray(ag.get('MapNghSet', []))
        merged_set = np.asarray(ag.get('MergedMapSet', []))

        f_count = len(feature_set)
        m_count = len(map_set)
        mm_count = len(merged_set)

        if len(map_ngh) == m_count and m_count > 0:
            own_count = int(np.sum(map_ngh == a))
            ngh_count = int(np.sum(map_ngh != a))
        else:
            own_count = m_count
            ngh_count = 0

        feature_counts.append(f_count)
        map_counts.append(m_count)
        map_own_counts.append(own_count)
        map_ngh_counts.append(ngh_count)
        merged_counts.append(mm_count)
        reobs_counts.append(int(ag.get('ReobsCount', 0)))
        outlier_dist_counts.append(int(ag.get('OutliersRemovedDistance', 0)))
        outlier_stale_counts.append(int(ag.get('OutliersRemovedStale', 0)))
        map_growth.append(int(ag.get('MapGrowth', 0)))

    # Figure 1: direct count comparison (most useful for your current issue)
    plt.figure("Sparse vs Dense Count Diagnostics", figsize=(12, 6))
    plt.plot(t, feature_counts, label='FeatureSet (current observations)', linewidth=1.8, color='c')
    plt.plot(t, map_own_counts, label='MapSet own (spheres in orange)', linewidth=2.0, color='tab:orange')
    plt.plot(t, map_ngh_counts, label='MapSet neighbor/shared (spheres in blue)', linewidth=2.0, color='tab:blue')
    plt.plot(t, map_counts, label='MapSet total', linewidth=1.2, linestyle='--', color='tab:red')
    plt.plot(t, merged_counts, label='MergedMapSet (dense)', linewidth=1.4, color='tab:purple')
    plt.xlabel('Time [s]')
    plt.ylabel('Count')
    plt.title(f'Agent {a}: FeatureSet vs Sparse MapSet vs MergedMapSet')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Figure 2: map maintenance diagnostics (do we add/remove sparse landmarks?)
    plt.figure("Sparse Map Maintenance Diagnostics", figsize=(12, 6))
    plt.plot(t, map_growth, label='MapGrowth', linewidth=1.5, color='tab:green')
    plt.plot(t, reobs_counts, label='ReobsCount', linewidth=1.5, color='tab:brown')
    plt.plot(t, outlier_dist_counts, label='OutliersRemovedDistance', linewidth=1.5, color='tab:gray')
    plt.plot(t, outlier_stale_counts, label='OutliersRemovedStale', linewidth=1.5, color='k')
    plt.xlabel('Time [s]')
    plt.ylabel('Count per frame')
    plt.title(f'Agent {a}: Sparse Map Update/Filtering Diagnostics')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Console summary for quick verification before checking figures.
    print("\n=== Sparse-vs-dense diagnostics summary ===")
    print(f"Agent {a}")
    print(f"FeatureSet count range: min={np.min(feature_counts)}, max={np.max(feature_counts)}")
    print(f"MapSet total count range: min={np.min(map_counts)}, max={np.max(map_counts)}")
    print(f"MapSet own count range: min={np.min(map_own_counts)}, max={np.max(map_own_counts)}")
    print(f"MapSet neighbor count range: min={np.min(map_ngh_counts)}, max={np.max(map_ngh_counts)}")
    print(f"MergedMapSet count range: min={np.min(merged_counts)}, max={np.max(merged_counts)}")
    print(f"Max OutliersRemovedDistance: {np.max(outlier_dist_counts)}")
    print(f"Max OutliersRemovedStale: {np.max(outlier_stale_counts)}")

    # Show timestamps where observations exist but sparse map is empty.
    suspect_idx = [i for i in range(num_iter) if feature_counts[i] > 0 and map_counts[i] == 0]
    if len(suspect_idx) > 0:
        print(f"Frames with FeatureSet>0 but MapSet==0: {len(suspect_idx)}")
        print(f"First 10 suspect times [s]: {[round(t[i], 3) for i in suspect_idx[:10]]}")
    else:
        print("No frames found with FeatureSet>0 and MapSet==0.")


# %% Loop-closure diagnostics - Re-observations in selected features
Case = 29 * np.ones(N)
A = [a for a in range(1, N + 1)]
plot_title = 'Re-observations in selected features'
Telemetry.plot_pose_graph_optimization(fig_name, plot_title, Case, A, Agents_History_nk, num_iter, step_size)


#######################################################################################################################
#######################################################################################################################


# # TOtal Error Evolution
# Case = [13]
# A = a*np.array([1])
# plot_title = 'Total FGO Error'
# Telemetry.plot_kinematic_factor_comparison(fig_name, plot_title, Case, A, Agents_History_nkf, Agents_History_1k, Agents_History_nk, num_iter, step_size)


# # Error to Map size ratio Evolution
# Case = [16]
# A = a*np.array([1])
# plot_title = 'Error-to-Map Size Ratio'
# Telemetry.plot_kinematic_factor_comparison(fig_name, plot_title, Case, A, Agents_History_nkf, Agents_History_1k, Agents_History_nk, num_iter, step_size)


#######################################################################################################################
#######################################################################################################################
# FOUR CASES
#######################################################################################################################
#######################################################################################################################
# %%
# Total Error Evolution
# Case = [13]
# A = a*np.array([1])
# plot_title = 'Total FGO Error'
# Telemetry.plot_kinematic_factor_comparison_4(fig_name, plot_title, Case, A,
#                                            Agents_History_nkf,
#                                            Agents_History_1k,
#                                            Agents_History_mk,
#                                            Agents_History_nk,
#                                            num_iter, step_size)

# # %%
# # Error to Map size ratio Evolution
# Case = [16]
# A = a*np.array([1])
# plot_title = 'Error-to-Map Size Ratio'
# Telemetry.plot_kinematic_factor_comparison_4(fig_name, plot_title, Case, A,
#                                            Agents_History_nkf,
#                                            Agents_History_1k,
#                                            Agents_History_mk,
#                                            Agents_History_nk,
#                                            num_iter, step_size)
