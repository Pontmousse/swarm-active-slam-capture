import pickle
import matplotlib.pyplot as plt
import numpy as np
import Plot_Telemetry_Func as Telemetry
import Load_Target as lt
import pybullet as p
import pybullet_data
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import shared_config

def merge_lists(list1, list2):
    # Combine the lists and convert to a set to remove duplicates
    merged_set = set(list1 + list2)
    # Convert the set back to a list
    merged_list = list(merged_set)
    return merged_list

def find_redundancies(lst):
    frequency = {}
    redundancies = []

    for item in lst:
        # Increment the count for each item
        if item in frequency:
            frequency[item] += 1
        else:
            frequency[item] = 1

    # Identify elements that appeared more than once
    for item, count in frequency.items():
        if count > 1:
            redundancies.append(item)

    return redundancies

#############################################################################################
#############################################################################################
#############################################################################################

# Connect to PyBullet and set up the simulation
physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)

#############################################################################################
object_name = shared_config.object_name
dt_ = shared_config.DT
paths = shared_config.get_sim_data_paths(
    n=shared_config.N,
    d=shared_config.D,
    dt=dt_,
    name=object_name,
)
tag = paths["tag"]

#############################################################################################

path_agents = paths["agents"]
path_target = paths["target"]

# Load Simulation History
Agents_History = Telemetry.load_variable_from_file(path_agents)
Target_History = Telemetry.load_variable_from_file(path_target)
N = len(Agents_History[0]) # Number of agents
num_iter = len(Agents_History)

#############################################################################################
# load initial target pcd
target_position = Target_History[0][:3]
target_orientation = Target_History[0][6:10]
texTar_id = p.loadTexture("Targets/Texture_Target.jpg")
_, target_pcd, _ = lt.load_target(target_position,target_orientation,texTar_id)

M = len(target_pcd.points)

#############################################################################################
#############################################################################################
#############################################################################################

# Fraction of Total Features Observed in the Swarm

dt = 1/dt_
Time = []
FeatureFrac = []
for i in range(num_iter):
    sum = 0
    Features = []
    for a in range(N):
        feat = Agents_History[i][a]['FeatureIdxSet']
        Features = merge_lists(Features, feat)
    sum = len(Features)  
    FeatureFrac.append(sum/M)
    Time.append(i*dt)

############################################################################
# Plot Data

plt.figure()

# Set labels for x and y axis
plt.xlabel("Time")

# Use a for loop to plot each array in Observations and add legend names
Legend = 'Fraction of Total Features Observed in the Swarm'
plt.plot(Time, FeatureFrac, label=Legend)

# Show the legend in the plot
plt.legend()

# Show the plot
plt.show()


#############################################################################################
#############################################################################################
#############################################################################################












