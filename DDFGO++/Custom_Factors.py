import numpy as np
import gtsam
from typing import List, Optional
from sympy import * # import everything
from helper import *

__all__ = ['jacobian', 'descriptor2symbols',
           'landmark_factor_error', 'kinem_error',
           'center_of_mass_factor_error', 'com_error',
           'target_velocity_factor_error', 'vel_error',
           'ang_vel_error']

###############################################################################
###############################################################################
# LANDMARK KINEMATIC FACTOR SYMBOLIC EQUATIONS & FUNCTIONS
###############################################################################
###############################################################################

def jacobian(error, Lvar):
    # Inputs:
    # error is a vector of symbolic expressions defining the error to be differentiated
    # Lvar is the list of variables to be differentiated with to calculate the Jacobian

    # Outputs a matrix with symbolic expressions for the Jacobian to be evaluated during constraints calculations
    
    nvar = len(Lvar)
    pdim  = len(error)

    J = Matrix.ones(pdim, nvar)
    for i in range(pdim):
        for j in range(nvar):
            errori = error[i]
            J[i,j] = errori.diff(Lvar[j])

    return J

def landmark_factor_error():
    # Outputs a vector with symbolic expressions

    x1, y1, z1 = symbols('x1 y1 z1')
    x2, y2, z2 = symbols('x2 y2 z2')
    Lvar = [x1, y1, z1, x2, y2, z2] # List of variables

    w1, w2, w3 = symbols('w1 w2 w3')
    v1, v2, v3 = symbols('v1 v2 v3')
    p1, p2, p3 = symbols('p1 p2 p3')
    dt = symbols('dt')
    Lpar = [w1, w2, w3, v1, v2, v3, p1, p2, p3, dt] # List of parameters (constants, not to be differentiated)


    o1, o2, o3 = symbols('o1 o2 o3')
    Lobs = [o1, o2, o3] # List of observations (constants, not to be differentiated)


    # r2 - r1 - obs
    term1 = Matrix([[x2-x1],
                    [y2-y1],
                    [z2-z1]])
    
    # (w x r1) * dt
    term2 = Matrix([[(w2 * (z1 - p3) - w3 * (y1 - p2)) * dt],
                    [(w3 * (x1 - p1) - w1 * (z1 - p3)) * dt],
                    [(w1 * (y1 - p2) - w2 * (x1 - p1)) * dt]])
    
    # v * dt
    term3 = Matrix([[v1 * dt],
                    [v2 * dt],
                    [v3 * dt]])
    
    # obs
    term4 = Matrix([[o1],
                    [o2],
                    [o3]])
    
    error = term1 - term2 - term3 - term4

    return error, Lvar, Lobs, Lpar

def kinem_error(measurement: np.ndarray, w, v, p, dt, lf_error, lf_jacobian, Lvar, Lobs, Lpar,
                this: gtsam.CustomFactor,
                values: gtsam.Values,
                jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    """Landmark Kinematic Factor error function
    :param measurement: landmark position difference measurement, to be filled with `partial`
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the error
    """


    # extract keys (or nodes or variables)
    key1 = this.keys()[0]
    key2 = this.keys()[1]

    # DEBUG
    # print('Symbols')
    # print(gtsam.Symbol(key1))
    # print(gtsam.Symbol(key2))    

    # extract values and evaluate on error
    r1 = list(values.atPoint3(key1))
    r2 = list(values.atPoint3(key2))

    obs = [measurement[0], measurement[1], measurement[2]]
    par = [w[0], w[1], w[2], v[0], v[1], v[2], p[0], p[1], p[2], dt]

    val = r1+r2+obs+par

    L = Lvar+Lobs+Lpar
    values = {key: val[s] for s, key in enumerate(L)}

    error = np.array(lf_error.evalf(subs=values)).astype(np.float64)


    # Calculate Jacobians
    if jacobians is not None:
        J = np.array(lf_jacobian.evalf(subs=values)).astype(np.float64)
        jacobians[0] = J[: , :3]  # Jacobian wrt landmark 1
        jacobians[1] = J[: , -3:]  # Jacobian wrt landmark 2

    return error

###############################################################################
###############################################################################
# TARGET CENTER OF MASS FACTOR SYMBOLIC EQUATIONS & FUNCTIONS
###############################################################################
###############################################################################

def descriptor2symbols(L,frame_idx,i):
    # functions takes a keyframe at given time step i containing
    # a list of landmark descriptors
    #
    # outputs a list of the variable ID node symbols recognized by gtsam
    list_symbols = []
    for descriptor in frame_idx:
        list_symbols.append(varL(L,descriptor,i))
    return list_symbols
    
       
def center_of_mass_factor_error(num_lan):
    # outputs symbolc equations that contains all the required variables to input
    # all the observed landmarks and the available map landmarks (not variables
    # not to be differentiated)
    
    # center of mass variable node
    comx, comy, comz = symbols('comx comy comz')
    
    # landmark variable nodes
    Lvar = []
    for i in range(num_lan):
        Lvar.append(symbols('var_x'+str(i)))
        Lvar.append(symbols('var_y'+str(i)))
        Lvar.append(symbols('var_z'+str(i)))
        
    Lvar.append(comx)
    Lvar.append(comy)
    Lvar.append(comz)
    
    
    # Average calculated from available Map
    mapx, mapy, mapz = symbols('map_x map_y map_z')
    Lpar = [mapx, mapy, mapz]
    
    #################################################
    
    expected = Matrix([[0],
                       [0],
                       [0]])
    
    for i in range(1 , num_lan+1):
        expected = expected + Matrix([[Lvar[3*i]],
                                      [Lvar[3*i+1]],
                                      [Lvar[3*i+2]]])
    
    expected = expected / num_lan
    
    #################################################
    
    map_avg = Matrix([[mapx],
                      [mapy],
                      [mapz]])
    expected = (expected + map_avg) / 2
    
    if mapx == 0 and mapy == 0 and mapz == 0: expected = 2*expected
    
    #################################################
    
    guess = Matrix([[comx],
                    [comy],
                    [comz]])
    
    error = expected - guess
        
    return error, Lvar, Lpar


def com_error(measurement: np.ndarray, map_avg, comf_error, comf_jacobian, Lvar, Lpar,
                this: gtsam.CustomFactor,
                values: gtsam.Values,
                jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    """Target Center of Mass Factor error function
    :param measurement: landmark position difference measurement, to be filled with `partial`
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the error
    """
    
    # Extract keys dynamically
    keys = this.keys()
    
    # DEBUG
    # print('Symbols')
    # for key in keys:
    #     print(gtsam.Symbol(key))

    # Extract values for all keys
    r_values = []
    for key in keys:
        r_values.extend(list(values.atPoint3(key)))
    
    # extract number of landmarks involved
    # num_lan = (len(r_values)-3)/3    
    
    
    # Combine all extracted variable values and map center of mass into one list
    if np.isnan(map_avg).any():
        val = r_values + [0, 0, 0]
    else:
        val = r_values + list(map_avg)
    
    # Update L variable list to reflect the variable number of inputs
    L = Lvar + Lpar
    values_dict = {key: val[s] for s, key in enumerate(L)}

    # Calculate the error
    error = np.array(comf_error.evalf(subs=values_dict)).astype(np.float64)



    # Calculate Jacobians
    if jacobians is not None:
        J = np.array(comf_jacobian.evalf(subs=values_dict)).astype(np.float64)
        
        # Dynamically allocate Jacobians for each variable node
        start_col = 0
        for i, key in enumerate(keys):
            num_columns = 3  # Assuming each key corresponds to a 3D Point3
            jacobians[i] = J[:, start_col:start_col + num_columns]
            start_col += num_columns

    return error

###############################################################################
###############################################################################
# TARGET VELOCITY FACTOR SYMBOLIC EQUATIONS & FUNCTIONS
###############################################################################
###############################################################################

def target_velocity_factor_error():
    # outputs symbolc equations that contains all the required variables to input
    # Event the time step is output as a variable but only as a constant
    # parameter (non-differentiable)
    
    # velocity variable node
    vx, vy, vz = symbols('vx vy vz')
    
    # center of mass variable nodes
    com1x, com1y, com1z = symbols('com1x com1y com1z')
    com2x, com2y, com2z = symbols('com2x com2y com2z')
    
    # variable nodes
    Lvar = [vx, vy, vz, com1x, com1y, com1z, com2x, com2y, com2z]
    
    # parameter time step (constant non-differentiable)
    dt = symbols('dt')
    Lpar = [dt]
    #################################################
    
    expected = Matrix([[(com2x - com1x)/dt],
                       [(com2y - com1y)/dt],
                       [(com2z - com1z)/dt]])
    
    
    guess = Matrix([[vx],
                    [vy],
                    [vz]])
    
    error = expected - guess
        
    return error, Lvar, Lpar


def vel_error(measurement: np.ndarray, dt, velf_error, velf_jacobian, Lvar, Lpar,
                          this: gtsam.CustomFactor,
                          values: gtsam.Values,
                          jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    """Target Velocity Factor error function
    :param measurement: landmark position difference measurement, to be filled with `partial`
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the error
    """
    
    # Extract keys dynamically
    keys = this.keys()
    
    # DEBUG
    # print('Symbols')
    # for key in keys:
    #     print(gtsam.Symbol(key))

    # Extract values for all keys
    r_values = []
    for key in keys:
        r_values.extend(list(values.atPoint3(key)))
    
    # extract number of landmarks involved
    # num_lan = (len(r_values)-3)/3    
    
    
    # Combine all extracted variable values and simulation time step into one list
    val = r_values + [dt]
    
    # Update L variable list to reflect the variable number of inputs
    L = Lvar + Lpar
    values_dict = {key: val[s] for s, key in enumerate(L)}

    # Calculate the error
    error = np.array(velf_error.evalf(subs=values_dict)).astype(np.float64)



    # Calculate Jacobians
    if jacobians is not None:
        J = np.array(velf_jacobian.evalf(subs=values_dict)).astype(np.float64)
        jacobians[0] = J[: , :3]  # Jacobian wrt velocity node
        jacobians[1] = J[: , 3:6]  # Jacobian wrt com 1
        jacobians[2] = J[: , -3:]  # Jacobian wrt com 2

    return error


###############################################################################
###############################################################################
# TARGET ANGULAR VELOCITY FACTOR SYMBOLIC EQUATIONS & FUNCTIONS
###############################################################################
###############################################################################

def numerical_jacobian(func, x, epsilon=1e-5):
    """
    Calculate the Jacobian of a function `func` at point `x` using central differences.
    
    :param func: Function that takes a 1D numpy array and returns a 1D numpy array (error function).
    :param x: 1D numpy array at which to evaluate the Jacobian.
    :param epsilon: Small perturbation for numerical differentiation.
    :return: Jacobian matrix as a 2D numpy array.
    """
    # Evaluate function at the original point
    f_x = func(x)
    jacobian = np.zeros((f_x.size, x.size))
    
    # Calculate each partial derivative using central differences
    for j in range(x.size):
        # Perturb x in the positive and negative direction
        x_pos = np.array(x)
        x_neg = np.array(x)
        
        x_pos[j] += epsilon
        x_neg[j] -= epsilon
        
        # Evaluate the function at perturbed points
        f_x_pos = func(x_pos)
        f_x_neg = func(x_neg)
        
        # Compute central difference for each component
        jacobian[:, j] = (f_x_pos - f_x_neg) / (2 * epsilon)
    
    return jacobian

def ang_vel_error(measurement: np.ndarray, dt, calculate_w, num_lan,
                  this: gtsam.CustomFactor,
                  values: gtsam.Values,
                  jacobians: Optional[List[np.ndarray]]) -> np.ndarray:
    """Target Angular Velocity Factor error function
    :param measurement: to be filled with `partial`
    :param dt: time step
    :param calculate_w: function to calculate angular velocity
    :param num_lan: number of landmarks
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the error
    """
    
    # print('\nThis is the custom angular velocity factor call\n')
    
    # Extract keys dynamically
    keys = this.keys()

    # Extract values for all keys
    r_values = []
    for key in keys:
        r_values.extend(list(values.atPoint3(key)))
    
    Points = np.array(r_values[:3*num_lan]).reshape(num_lan, 3)
    Points_prev = np.array(r_values[-3*num_lan-3:-3]).reshape(num_lan, 3)
    guess = np.array(r_values[-3:])
    
    # Calculate the expected angular velocity
    expected = calculate_w(Points, Points_prev, dt)
    
    # Calculate the error
    error = expected - guess
    
    ########################################################################
    # Calculate Jacobians (of the error)
    if jacobians is not None:
        # Define a lambda function that takes `r_values` and returns the error
        def error_func(r_values):
            Points = np.array(r_values[:3*num_lan]).reshape(num_lan, 3)
            Points_prev = np.array(r_values[-3*num_lan-3:-3]).reshape(num_lan, 3)
            guess = np.array(r_values[-3:])
            return calculate_w(Points, Points_prev, dt) - guess
        
        # Calculate the Jacobian using central differences
        J = numerical_jacobian(error_func, np.array(r_values))
        
        # Dynamically allocate Jacobians for each variable node
        start_col = 0
        for i, key in enumerate(keys):
            num_columns = 3  # Assuming each key corresponds to a 3D Point3
            jacobians[i] = J[:, start_col:start_col + num_columns]
            start_col += num_columns

    return error
