"""This file sets up a receding horizon model predictive control scenario,
constructed via the TwoDTrajectoryOptimization class.  This example defines a
square polytope of half width = 1 meter.  The SCENARIO parameter can be toggled
between 1, 2, and 3 to determine if the result will include angular error but no
penalty of slipping, position error only still without slipping penalty, or
position error only plus a cost on slipping.
"""

import numpy as np
import pdb

from toy_2d.src.two_dim_polytope import TwoDimensionalPolytope
from toy_2d.src.two_dim_polytope import TwoDimensionalPolytopeParams
from toy_2d.src.two_dim_system import TwoDSystemForceOnly
from toy_2d.src.two_dim_system import TwoDimensionalSystemParams
from toy_2d.src.two_dim_trajectory_optimization import \
    TwoDTrajectoryOptimizationParams, TwoDTrajectoryOptimization
from toy_2d.src.lca import admm_lca


# Set some parameters.
DT_SIM = 0.002
T_MULTIPLE = 50
DT_TRAJ_OPT = DT_SIM * T_MULTIPLE
MU_GROUND = 0.4
MU_CONTROL = 1.
LOOPS = 60
LOOKAHEAD = 10
INPUT_LIMIT = 5.
USE_BIG_M = False
USE_NON_CONVEX = not USE_BIG_M
OPTIMIZATION_TIME_LIMIT = 10.
SAVE_OUTPUT = False
ADMM_TOLERANCE = 1e0


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
sim_system_params = TwoDimensionalSystemParams(dt = DT_SIM, polytope = polytope,
                                               mu_control = MU_CONTROL)
sim_system = TwoDSystemForceOnly(sim_system_params, CONTACT_LOC, CONTACT_ANGLE)

# Build Q, R, and S matrices.  S_base is the amount to penalize slip in the x
# and y velocity directions.  It will get augmented as S = V.T @ S_base @ V for
# a state-dependent V to map directly from state to slip.
Q = np.diag([0.1, 0.1, 0.8, 1., 1., 1.])
# Q = np.diag([0.1, 0.1, 0.1, 10., 10., 10.])
# R = np.diag([0.003, 0.003])
R = np.diag([0.0, 0.0])
S_base = np.diag([0., 0.])

# Add a new matrix T which will penalize the size of the complementarity
# variables.  The complementarity vector is in the order of tangential, normal,
# then slack variables.
T_normal = [1.] * polytope.n_contacts
T_tangential = [1.] * polytope.n_contacts * polytope.n_friction
T_slack = [0.] * polytope.n_contacts
T = np.diag(T_tangential + T_normal + T_slack)

# Create trajectory optimization object.
traj_opt_params = TwoDTrajectoryOptimizationParams(sim_system=sim_system,
    traj_opt_dt=DT_TRAJ_OPT, Q=Q, R=R, S_base=S_base, lookahead=LOOKAHEAD,
    input_limit=INPUT_LIMIT, use_receding_horizon=False,
    optimization_time_limit=OPTIMIZATION_TIME_LIMIT
)
traj_opt = TwoDTrajectoryOptimization(traj_opt_params)

# pdb.set_trace()

# Test out using ADMM.
admm = admm_lca(traj_opt, x0, x_goal, rho=0.5, tol=ADMM_TOLERANCE, T=T)

# TEST_LOOPS = 8
# for _ in range(TEST_LOOPS):
#     admm.solve_reference_gurobi()
#     admm.solve_control_gurobi()
#     admm.construct_LCS_terms_from_inputs()
#     print('\n')

# pdb.set_trace()

try:
    admm.run_admm()
except KeyboardInterrupt:
    print('Interrupted.  Making gif from temp images.')
    admm.make_gif_from_temp_images()

pdb.set_trace()
