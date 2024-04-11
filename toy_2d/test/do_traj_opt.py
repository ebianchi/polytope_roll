"""This file sets up an offline trajectory optimization planning problem,
constructed via the TwoDTrajectoryOptimization class.  This example defines a
square polytope of half width = 1 meter."""

import numpy as np
import pdb

from toy_2d.src.two_dim_polytope import TwoDimensionalPolytope
from toy_2d.src.two_dim_polytope import TwoDimensionalPolytopeParams
from toy_2d.src.two_dim_system import TwoDSystemForceOnly
from toy_2d.src.two_dim_system import TwoDimensionalSystemParams
from toy_2d.src.two_dim_trajectory_optimization import \
    TwoDTrajectoryOptimizationParams, TwoDTrajectoryOptimization


# Set some parameters.
DT_SIM = 0.002
T_MULTIPLE = 50
DT_TRAJ_OPT = DT_SIM * T_MULTIPLE
HALF_FREQ = False
MU_GROUND = 0.4
MU_CONTROL = 1.
INPUT_LIMIT = 5.
OPTIMIZATION_TIME_LIMIT = 1000.
LOOPS = 60
LOOKAHEAD = 10
SAVE_OUTPUT = True


# Plotting.
file_title = "traj_opt"
title = "Trajectory Optimization"

# Begin to define the system starting with the contact location and direction.
CONTACT_LOC = np.array([1, 1])
CONTACT_ANGLE = np.pi

# Initial and goal conditions, in order of vx, vy, vth, x, y, th.
x0 = np.array([0., 0., 0., 0., 1., 0.])
x_goal = np.array([0., 0., 0., -4.5, 1., 0.])  #np.pi])

# Create a polytope.
poly_params = TwoDimensionalPolytopeParams(mass = 1, moment_inertia = 0.01,
    mu_ground = MU_GROUND, vertex_locations = np.array([[1, -1], [1, 1],
                                                        [-1, 1], [-1, -1]]))
polytope = TwoDimensionalPolytope(poly_params)

# Create a simulation system from the polytope, a simulation timestep, and a
# control contact's friction parameter.
sim_system_params = TwoDimensionalSystemParams(
    dt = DT_SIM, polytope = polytope, mu_control = MU_CONTROL)
sim_system = TwoDSystemForceOnly(sim_system_params, CONTACT_LOC, CONTACT_ANGLE)

# Build Q, R, and S matrices.  S_base is the amount to penalize slip in the x
# and y velocity directions.  It will get augmented as S = V.T @ S_base @ V for
# a state-dependent V to map directly from state to slip.  The order for Q uses
# the LCS ordering:  vx, vy, vth, x, y, th.
Q = np.diag([0.1, 0.1, 0.8, 1., 1., 5.])
R = np.diag([0.003, 0.003])
S_base = np.diag([0., 0.])

# Create trajectory optimization object.
traj_opt_params = TwoDTrajectoryOptimizationParams(
    sim_system=sim_system, traj_opt_dt=DT_TRAJ_OPT, Q=Q, R=R, S_base=S_base,
    lookahead=LOOKAHEAD, use_receding_horizon=False, input_limit=INPUT_LIMIT,
    optimization_time_limit=OPTIMIZATION_TIME_LIMIT
)
traj_opt = TwoDTrajectoryOptimization(traj_opt_params)

pdb.set_trace()

# Perform the trajectory optimization.
traj_opt.run_trajectory_optimization(x0, x_goal, LOOPS)

# Generate a gif from the iteration plots.
traj_opt.make_gif_from_temp_images()
pdb.set_trace()
