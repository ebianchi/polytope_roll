"""
This file tries to use 2 dimensional system dynamics using ADMM
"""

import cvxpy as cp
# import jax
# import jax.numpy as jnp
# from trajax import optimizers
# from trajax.integrators import rk4
# from functools import partial
import time
import numpy as np
from dataclasses import dataclass, field
import pdb
import matplotlib.pyplot as plt
import pickle
import sys
# sys.path.insert(1, '~/PycharmProjects/LCA-ADMM/contact-manipulation/polytope_roll/toy_2d/src')

import vis_utils, file_utils
from two_dim_polytope import TwoDimensionalPolytopeParams, TwoDimensionalPolytope
from two_dim_system import TwoDimensionalSystemParams, TwoDimensionalSystem, TwoDSystemMagOnly, TwoDSystemForceOnly
from two_dim_lcs_approximation import TwoDSystemLCSApproximation
from ilqr import iLQR, SolverParams
from lca import admm_lca, run_admm

# from toy_2d.src import vis_utils
# from toy_2d.src import file_utils
# from toy_2d.src.two_dim_polytope import TwoDimensionalPolytopeParams, \
#     TwoDimensionalPolytope
# from toy_2d.src.two_dim_system import TwoDimensionalSystemParams, \
#     TwoDimensionalSystem, TwoDSystemMagOnly, TwoDSystemForceOnly
# from toy_2d.src.two_dim_lcs_approximation import TwoDSystemLCSApproximation
# from toy2d.src.ilqr import iLQR, SolverParams


def main():
    # Define variables for the problem specification
    T = 20
    dt = 0.1
    eq_point = np.array([1, 1, 0])
    u_max = np.array([1, 4])
    u_min = np.array([-1, -4])

    # Sample the initial condition from a random normal distribution
    # np.random.seed(0)
    # rng = np.random.default_rng()
    # x0 = rng.standard_normal(3)
    x0 = np.array([0, 0, 0.2])
    u0 = np.zeros(2)
    rho = 50
    m = 2
    n = 3

    # Unicycle continuous dynamics
    # def car(x, u, t):
    #     return jnp.array([u[0] * jnp.cos(x[2]), u[0] * jnp.sin(x[2]), u[1]])

    # Discretize unicycle using rk4 - simple dynamics
    # dynamics = rk4(car, dt=dt)
    # admm_obj = admm_lca(dynamics, T, dt, m, n, x0, u0, eq_point, rho)

    # gain_K, gain_k, x, u, r = run_admm(admm_obj, u_max)
    # Plotting
    # ===============================
    # plt.figure()
    # # plt.axis("off")
    # plt.gca().set_aspect('equal', adjustable='box')
    #
    # x = admm_obj.rollout()
    # plt.plot(x[:, 0], x[:, 1], c='black')
    # plt.scatter(x[:, 0], x[:, 1], s=40)
    # plt.scatter(eq_point[0], eq_point[1], color='r', marker='.', s=200, label="Desired pose")
    # # plt.ylim(-0.1, 3)
    # # plt.xlim(-0.1, 3.5)
    # plt.legend()
    # plt.show()

    ###########################################################################################################

    ## Code for polytope roll taken from test_ilqr_2d

    # print ("Arguments Provided:", str(sys.argv))
    test_number = 0
    if (len(sys.argv) != 2):
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
    MU_GROUND = 0.4

    # Control properties
    MU_CONTROL = 0.5  # Currently, this isn't being used.  The ambition is for
    # this to help define a set of feasible control forces.

    # Simulation parameters.
    DT = 0.001  # If a generated trajectory looks messed up, it could be
    # fixed by making this timestep smaller.

    # Create a polytope.
    poly_params = TwoDimensionalPolytopeParams(
        mass=MASS,
        moment_inertia=MOM_INERTIA,
        mu_ground=MU_GROUND,
        vertex_locations=SQUARE_CORNERS
    )
    polytope = TwoDimensionalPolytope(poly_params)

    # Create a system from the polytope, a simulation timestep, and a control
    # contact's friction parameter.
    system_params = TwoDimensionalSystemParams(
        dt=DT,
        polytope=polytope,
        mu_control=MU_CONTROL
    )
    solver_params = SolverParams(
        decay=0.95,
        max_iterations=1000,
        tolerance=0.05
    )
    # Contact location and direction.
    CONTACT_LOC = np.array([-1, 1])
    CONTACT_ANGLE = 0.
    system = TwoDSystemForceOnly(system_params, CONTACT_LOC, CONTACT_ANGLE)
    # Create an LCS approximation from the system.
    lcs = TwoDSystemLCSApproximation(system)

    # Following the hw methodology of implementing ILQR
    # take the current state and action and give the current cost. Here the cost can be
    # non-linear but defining a quadratic cost makes life easier
    num_states = 6
    num_controls = 2
    # Initial conditions, in order of x, dx, y, dy, theta, dtheta
    x0 = np.array([0, 0, 1.0, 0, 0, 0])
    # goal to be achieved
    x_goal = np.array([5.0, 0, 1.0, 0, np.pi / 2, 0])
    # traj penalty which has to be tuned depending on where the goal is
    Q = np.eye(num_states)
    # Q[1,1]=Q[1,1]*0
    # Q[3,3]=Q[3,3]*0
    # Q[5,5]=Q[5,5]*0
    # final goal penalty. Needs tuning
    Qf = np.eye(num_states) * 1000
    # penalty on using more force. Experience suggests making this zero has bad effects
    R = np.eye(num_controls) * 0.05
    # number of timesteps in the rollout
    num_timesteps = 8000
    # initial guess required the most tuning. Experience suggests making this high initially
    # then let it optimize and cool down to the necessary value. Since we are putting a penalty on
    # the amount of force, starting with a low value might not be the best thing to do. Remember this is
    # a nonlinear program and if you start with a bad guess, you wont get a good solution.
    u_guess = np.ones((num_timesteps - 1, num_controls)) * 2
    # load your previous run answer to get a warm start. Remember, if you change num_timesteps you will
    # get error loading the previous solution, the timesteps also decide the num of control steps.
    # If you run the script with the same num_timesteps (as a previous run) and want to try different cost
    # make the flag true
    load_old_run = False
    if (load_old_run):
        file_name = "ilqr_controls_" + str(test_number)
        path = f'{file_utils.OUT_DIR}/{file_name}.txt'
        with open(path, 'rb') as fp:
            u_guess = pickle.load(fp)

    obj = iLQR(lcs, x_goal, num_timesteps, test_number, Q, R, Qf, solver_params)
    # obj.calculate_optimal_trajectory(x0, u_guess)
    u_new = obj.actions
    x_new = obj.states

    admm_obj = admm_lca(lcs, T, dt, num_controls, num_states, x0, u_guess, x_goal, rho)
    _, _, x_poly, u_poly, r_poly = run_admm(admm_obj, [lcs, num_timesteps, test_number, Q, R, Qf, solver_params])


if __name__ == '__main__':
    main()



