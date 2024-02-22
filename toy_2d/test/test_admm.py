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
from toy_2d.src.two_dim_traj_opt import TrajectoryOptimizationReport, \
                                        TwoDTrajectoryOptimizationParams, \
                                        TwoDTrajectoryOptimization
from toy_2d.src.lca import admm_lca


# Set some parameters.
DT_SIM = 0.002
T_MULTIPLE = 50
DT_TRAJ_OPT = DT_SIM * T_MULTIPLE
HALF_FREQ = False
MU_GROUND = 0.4
MU_CONTROL = 1.
LOOPS = 60
LOOKAHEAD = 4
USE_BIG_M = False
USE_NON_CONVEX = not USE_BIG_M
SCENARIOS = {1: 'Angular Error', 2: 'Position Error Only', 3: 'Minimum Slip'}
SCENARIOS_SHORT = {1: 'ang_err', 2: 'pos_err_only', 3: 'min_slip'}
SCENARIO = 3
SAVE_OUTPUT = False

# Based on the above settings, generate informative plot titles and file names.
blurb = SCENARIOS[SCENARIO]
short = SCENARIOS_SHORT[SCENARIO]
freq = ', 5Hz' if HALF_FREQ else ''
f_short = '_5hz' if HALF_FREQ else ''
title = f"Non-convex Trajectory Optimization, {blurb}, Double Pivot{freq}" \
        if USE_NON_CONVEX else \
        f"Big M Trajectory Optimization, {blurb}, Double Pivot{freq}"
file_title = f'traj_opt_{short}_double{f_short}' if USE_NON_CONVEX \
             else f'traj_opt_{short}_double{f_short}_M'


# Begin to define the system starting with the contact location and direction.
CONTACT_LOC = np.array([1, 1])
CONTACT_ANGLE = np.pi

# Initial and goal conditions, in order of vx, vy, vth, x, y, th.
x0 = np.array([0., 0., 0., 0., 1., 0.])
x_goal = np.array([0., 0., 0., -4.5, 1., np.pi])

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
R = np.diag([0.003, 0.003])
S_base = np.diag([0., 0.])

if SCENARIOS[SCENARIO] == 'Angular Error':
    pass
elif SCENARIOS[SCENARIO] == 'Position Error Only':
    Q[5,5] = 0.
elif SCENARIOS[SCENARIO] == 'Minimum Slip':
    Q[5,5] = 0.
    S_base = np.diag([0.3, 0.3])

# Create trajectory optimization object.
traj_opt_params = TwoDTrajectoryOptimizationParams(sim_system=sim_system,
    traj_opt_dt=DT_TRAJ_OPT, Q=Q, R=R, S_base=S_base, lookahead=LOOKAHEAD)
traj_opt = TwoDTrajectoryOptimization(traj_opt_params)

pdb.set_trace()

# Test out using ADMM.
u0 = np.zeros((traj_opt.n_controls,))
admm = admm_lca(traj_opt, x0, u0, x_goal, rho=0.5)
admm.solve_reference_gurobi()

pdb.set_trace()
