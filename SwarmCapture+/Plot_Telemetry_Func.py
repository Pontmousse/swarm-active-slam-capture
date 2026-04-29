import numpy as np
import pickle
import pybullet
import Controllers as C

dt = 1/240
s = 1

def load_variable_from_file(filename):
    with open(filename, 'rb') as file:
        variable = pickle.load(file)
    return variable

#############################################################################

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

def call_parameter(Agent, case, attachment_points, special_inputs):
    a = Agent["ID"]

    if case == 1:
        p = Agent["State"][0:3]
        title = 'Position'+' for agent '+str(a)
        ##########################################

    elif case == 2:
        m = Agent["Mode"]

        if m == 'e':
            p = 1
        elif m == 'c':
            p = 2
        else:
            p = 3

        title = 'Mode'+' for agent '+str(a)
        ##########################################

    elif case == 3:
        p = Agent["State"][10:13]
        title = 'Ang Vel'+' for agent '+str(a)
        ##########################################

    elif case == 4:
        p = np.linalg.norm(Agent["LCD"])
        title = 'LCD'+' for agent '+str(a)
        ##########################################

    elif case == 5:
        p1 = Agent["LCD"]
        p2 = Agent["LCD_Frame"]

        if len(p1) == 0:
            p1 = np.array([0,0,0])

        if len(p2) == 0:
            p2 = np.array([0,0,0])
        else:
            p2 = Agent["LCD_Frame"][0]

        p  = np.dot(p1,p2)

        title = 'LCD pointing'+' for agent '+str(a)
        ##########################################

    elif case == 6:
        Landm = Agent['LandSet']
        p  = len(Landm)
        title = 'No of Landmarks'+' for agent '+str(a)
        ##########################################

    elif case == 7:
        Comm = Agent['CommSet']
        p  = len(Comm)
        title = 'No of Comms'+' for agent '+str(a)
        ##########################################

    elif case == 8:
        Landm = Agent['FeatureSet']
        p  = len(Landm)
        title = 'No of Features'+' for agent '+str(a)
        
    elif case == 9:
        p = None
        if Agent['ActionTime']:
            p = Agent['ActionTime']
        title = 'ActionTime'+' for agent '+str(a)
    
    elif case == 10:
        p = Agent['CHW_Force']
        title = 'CHW_Force'+' for agent '+str(a)
    
    elif case == 11:
        p = Agent['Control_Force']
        title = 'Control_Force'+' for agent '+str(a)
      
    elif case == 12:
        p = Agent['Control_Torque']
        title = 'Control_Torque'+' for agent '+str(a)
    
    elif case == 13:
        p = Agent['Fuel_Consumed']
        title = 'Fuel consumed'+' for agent '+str(a)
    
    elif case == 14:
        p = Agent["State"][6:10]
        title = 'Orientation'+' for agent '+str(a)
        
    elif case == 15:
        p = Agent["Smooth_State"][3:6]
        title = 'Velocity'+' for agent '+str(a)
        
    elif case == 16:
        p = None
        if Agent['Target'] != []:
            p = Agent['Target']
        title = 'Target attachment point ID'+' for agent '+str(a)
    
    elif case == 17:
        p = (None, None, None)
        if Agent['Target'] != []:
            k = Agent['Iteration']
            attachment_point = C.extract_attachment_point(attachment_points, Agent['Target'])
            attachment_point = special_inputs['attachment_point']
            p = attachment_point.position[k] + special_inputs['dock_shift']
        title = 'Target attachment point position'+' for agent '+str(a)
    
    elif case == 18:
        p = (None, None, None)
        if Agent['Target'] != []:
            k = Agent['Iteration']
            attachment_point = C.extract_attachment_point(attachment_points, Agent['Target'])
            attachment_point = special_inputs['attachment_point']
            p = attachment_point.velocity[k]
        title = 'Target attachment point velocity'+' for agent '+str(a)
    
    
    elif case == 19:
        p = Agent['Smooth_Control_Force']
        title = 'Smooth_Control_Force'+' for agent '+str(a)
      
    elif case == 20:
        p = Agent['Smooth_Control_Torque']
        title = 'Smooth_Control_Torque'+' for agent '+str(a)
        
    else:
        raise ValueError(f"Invalid case for parameter call")

    return p, title

def generate_color_lists_rgb(N):
    if N > 10 or N < 1:
        raise ValueError("N must be between 1 and 20.")
    
    # Base list of 10 distinct colors in hexadecimal
    color_codes = [        
        "#FF0000",  # Red        
        "#0000FF",  # Blue        
        "#FF00FF",  # Magenta        
        "#800000",  # Maroon
        "#808000",  # Olive
        "#008080",  # Teal
        "#800080"   # Purple
        "#FFFF00",  # Yellow
        "#00FFFF",  # Cyan
        "#00FF00",  # Green
    ]
    
    def hex_to_rgba(hex_color, opacity=1.0):
        """Convert HEX color to RGBA."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        return [r, g, b, opacity]
    
    # Generate RGBA lists
    list = [color for color in color_codes[:N]]
    
    return list

def generate_light_color_lists_rgb(N):
    if N > 10 or N < 1:
        raise ValueError("N must be between 1 and 20.")
    
    # Base list of 10 distinct colors in hexadecimal
    color_codes = [              
        "#FF6666",  # Light Red
        "#6666FF",  # Light Blue
        "#FF66FF",  # Light Magenta
        "#A05252",  # Lighter Maroon
        "#A0A052",  # Lighter Olive
        "#52A0A0",  # Lighter Teal
        "#A052A0"   # Lighter Purple
        "#FFFF66",  # Light Yellow        
        "#66FFFF",  # Light Cyan
        "#66FF66",  # Light Green
        
    ]
    
    def hex_to_rgba(hex_color, opacity=1.0):
        """Convert HEX color to RGBA."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        return [r, g, b, opacity]
    
    # Generate RGBA lists
    list = [color for color in color_codes[:N]]
    
    return list