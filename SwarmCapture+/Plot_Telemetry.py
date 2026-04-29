import pickle
import matplotlib.pyplot as plt
import numpy as np
import Plot_Telemetry_Func as Telemetry
import Controllers as C

#############################################################################################
#############################################################################################
#############################################################################################

DT = 240
dt = 1/DT # time step
N = 6 # Number of Agents_Bodies
D = 150  # Simulation duration



# object_name = 'Motor' # Target selected
object_name = 'Orion_Capsule' # Target selected




tag = 'N'+str(N)+'_D'+str(D)+'_dt'+str(DT)
tag = tag+'_'+object_name

#############################################################################################
#############################################################################################
#############################################################################################
# Load Simulation History

path_agents = '/home/elghali/Desktop/SwarmCapture+/Data/Agents_History_'+tag+'.pkl'
print('\nLoading agents history pickle file...')
Agents_History = Telemetry.load_variable_from_file(path_agents)
print('Loaded successfully')


path_attachment_points = '/home/elghali/Desktop/SwarmCapture+/Data/Attachment_Points_'+tag+'.pkl'
print('\nLoading attachment points history pickle file...')
attachment_points = Telemetry.load_variable_from_file(path_attachment_points)
print('Loaded successfully')

dt = 1/240 # time step
N = len(Agents_History[0]) # Number of agents
num_iter = len(Agents_History)



# Downsample History to speed up animation
step_size = 1
Agents_History = [Agents_History[i] for i in range(10, num_iter, step_size)]
num_iter = len(Agents_History)

print(num_iter)

#%%
############################################################################
# Build Plotting Vectors
############################################################################

############################################################################
# # Telemetry ALL cases
Case = 2*np.ones(N)
# # # Agent ID: index 1 for first agent
A = np.arange(1, N+1)

############################################################################
Set_Colors = True
# Set_Colors = False

# # Telemetry multiple cases
# Case = 1*np.array([15,18]) # capture velocity
# Case = 1*np.array([1,17]) # capture position

# # Agent ID: index 1 for first agent
# A = 2*np.array([1,1])

############################################################################
# Telemetry single case
# Case = [10]
# Agent ID: index 1 for first agent
# A = [1]


############################################################################
# Plot graph
############################################################################
if Set_Colors:
    Color = Telemetry.generate_color_lists_rgb(N)


c = len(Case)
Observations = []
Legend = []
CLR = []

special_inputs = {}

for i in range(c):
    Obs = []
    a = A[i]-1
    case = Case[i]
    if Set_Colors: clr = Color[i]
    
    # Special Inputs
    # dock_iter = Agents_History[-1][a]['DockTime']
    # attachment_point = C.extract_attachment_point(attachment_points, Agents_History[dock_iter][a]['Target'])
    # dock_shift = Agents_History[dock_iter][a]['State'][:3] - attachment_point.position[Agents_History[dock_iter][a]['Iteration']]
    # special_inputs['dock_shift'] = dock_shift
    # special_inputs['attachment_point'] = attachment_point


    for j in range(num_iter):
        p , _ = Telemetry.call_parameter(Agents_History[j][a], case, attachment_points, special_inputs)
        Obs.append(p)
    _ , title = Telemetry.call_parameter(Agents_History[j][a], case, attachment_points, special_inputs)

    Legend.append(title)
    Observations.append(Obs)
    if Set_Colors: CLR.append(clr)

############################################################################

Time = []
for j in range(num_iter):
    Time.append(j*step_size*dt)

############################################################################
# Plot Data

plt.figure(figsize=(12,12), dpi=300)

# Set labels for x and y axis
plt.xlabel("Time (s)")

substring = ' for agent ' + str(a)
text = title.split(substring)[0]
plt.ylabel(text)


############################################################################
############################################################################
# plt.ylim(-0.1,0.1)

# plt.xlim(0,65.60) # encapsulation phase
# plt.xlim(45,55) # capture phase


# plt.ylim(-12,12) # zoom in
# plt.xlim(20,25) # 5s zoom in encapsulation phase
# ax.set_aspect(aspect=0.08)  # Elongate x-axis


# plt.xlim(22,22.5) # 0.2s zoom in encapsulation phase
# ax.set_aspect(aspect=0.005)  # Elongate x-axis
############################################################################
############################################################################


# Use a for loop to plot each array in Observations and add legend names
for i, observation_array in enumerate(Observations):
    if Set_Colors:
        plt.plot(Time, observation_array, label=Legend[i], color = CLR[i])
    else:
        plt.plot(Time, observation_array, label=Legend[i])

# Show the legend in the plot
plt.legend()


save_path = "/home/elghali/Desktop/SwarmCapture+/Data/N"+str(N)+"_D"+str(D)+"_"+text+".png"
plt.savefig(save_path, dpi=300)

# Show the plot
plt.show()