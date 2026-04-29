import pickle
import matplotlib.pyplot as plt
import numpy as np
import Plot_Telemetry_Func as Telemetry
import Load_Target as lt
import pybullet as p
import pybullet_data

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
object_name = 'Orion_Capsule' # Target selected
dt_ = 240
tag = 'N10_D3_dt'+str(dt_)
tag = tag+'_'+object_name

#############################################################################################

path_agents = 'Data/Agents_History_'+tag+'.pkl'
path_target = 'Data/Target_History_'+tag+'.pkl'

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












