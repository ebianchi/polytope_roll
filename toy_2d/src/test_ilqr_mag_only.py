"""This file tests the iLQR implementation for the case where only the force magnitude is chosen by the
controller and the location and direction is pre-decided
"""
import numpy as np
import sys
import pdb
import matplotlib.pyplot as plt

from toy_2d.src import vis_utils
from toy_2d.src.two_dim_polytope import TwoDimensionalPolytopeParams, \
                                        TwoDimensionalPolytope
from toy_2d.src.two_dim_system import TwoDimensionalSystemParams, \
                                      TwoDSystemMagOnly
from toy_2d.src.ilqr import iLQR, SolverParams
from toy_2d.src.two_dim_lcs_approximation import TwoDSystemLCSApproximation


if __name__ == "__main__":
    test_number = 0
    if(len(sys.argv) !=2):
        print("Required arguments not provided")
        print("Usage: python3 path_to_script/test_ilqr_2d.py test_number")
        print("test_number should be an integer")
        sys.exit()
    else: 
        try:  
            test_number = int(sys.argv[1]) 
        except ValueError:
            print("Usage: python3 path_to_script/test_ilqr_2d.py test_number")
            print("test_number should be an integer")
            sys.exit()
    # Fixed parameters
    # A few polytope examples.
    SQUARE_CORNERS = np.array([[1, -1], [1, 1], [-1, 1], [-1, -1]])
    STICK_CORNERS = np.array([[1, 0], [-1, 0]])
    RAND_CORNERS = np.array([[0.5, 0], [0.7, 0.5], [0, 0.8], [-1.2, 0], [0, -0.5]])

    # Polytope properties
    MASS = 1
    MOM_INERTIA = 0.1
    MU_GROUND = 0.1

    # Control properties
    MU_CONTROL = 0.5    # Currently, this isn't being used.  The ambition is for
                        # this to help define a set of feasible control forces.

    # Simulation parameters.
    DT = 0.001          # If a generated trajectory looks messed up, it could be
                        # fixed by making this timestep smaller.


    # Create a polytope.
    poly_params = TwoDimensionalPolytopeParams(
        mass = MASS,
        moment_inertia = MOM_INERTIA,
        mu_ground = MU_GROUND,
        vertex_locations = SQUARE_CORNERS
    )
    polytope = TwoDimensionalPolytope(poly_params)

    # Create a system from the polytope, a simulation timestep, and a control
    # contact's friction parameter.
    system_params = TwoDimensionalSystemParams(
        dt = DT,
        polytope = polytope,
        mu_control = MU_CONTROL
    )

    solver_params = SolverParams(
        decay = 0.95,
        max_iterations = 1000,
        tolerance = 0.05
    )
    # Contact location and direction.
    CONTACT_LOC = np.array([-1, 1])
    CONTACT_ANGLE = 0.

    system = TwoDSystemMagOnly(system_params, CONTACT_LOC, CONTACT_ANGLE)


    # Create an LCS approximation from the system.
    lcs = TwoDSystemLCSApproximation(system)

    #Following the hw methodology of implementing ILQR
    #take the current state and action and give the current cost. Here the cost can be 
    #non-linear but defining a quadratic cost makes life easier

    # Initial conditions, in order of x, dx, y, dy, theta, dtheta
    x0 = np.array([0, 0, 1.0, 0, 0, 0])
    x_goal = np.array([4.0, 0, 1.0, 0, -np.pi/2, 0])
    Q = np.eye(6)
    Q[1,1]=Q[1,1]*0
    Q[3,3]=Q[3,3]*0
    Q[5,5]=Q[5,5]*0
    Qf = np.eye(6)*10

    #penalty on using more force. Experience suggests making this zero has bad effects
    R = np.eye(1)*0.01

    #number of timesteps in gthe rollout
    num_timesteps = 5000
    u_guess = np.ones((num_timesteps-1,1))*6

    num_states = 6
    num_controls = 1
    
    #load your previous run answer to get a warm start. Remember, if you change num_timesteps you will
    #get error loading the previous solution, the timesteps also decide the num of control steps. 
    #If you run the script with the same num_timesteps (as a previous run) and want to try different cost
    # make he flag true 
    load_old_run = False  
    if(load_old_run):
        file_name = "controls" + str(test_number)
        path = f'{file_utils.OUT_DIR}/{file_name}.txt'
        with open (path, 'rb') as fp:
            u_guess = pickle.load(fp)

    obj = iLQR(lcs, x_goal, num_timesteps, test_number,Q,R,Qf, solver_params)
    obj.calculate_optimal_trajectory(x0, u_guess)



