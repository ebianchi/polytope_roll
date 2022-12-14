"""This file is a different implementation of the TwoDTrajectoryOptimization
functionality that tests the trajectory optimization formulation where the
control force is constrained to be normal to the polytope face at a fixed
contact point.
"""

import numpy as np
import scipy.sparse as sp
import pdb
import timeit

import gurobipy as gp
from gurobipy import GRB

from toy_2d.src import file_utils
from toy_2d.src import vis_utils
from toy_2d.src.two_dim_polytope import TwoDimensionalPolytope
from toy_2d.src.two_dim_polytope import TwoDimensionalPolytopeParams
from toy_2d.src.two_dim_system import TwoDSystemMagOnly
from toy_2d.src.two_dim_system import TwoDimensionalSystemParams
from toy_2d.src.two_dim_lcs_approximation import TwoDSystemLCSApproximation



# Set some parameters.
DT_SIM = 0.002
T_MULTIPLE = 50
DT_TRAJ_OPT = DT_SIM * T_MULTIPLE
MU_GROUND = 0.5  # stick/slip transition is between mu = 0.5 and 0.49
M1 = 1e3
M2 = 1e3
LOOPS = 50
INPUT_LIMIT = 5.
LOOKAHEAD = 3
USE_BIG_M = False
USE_NON_CONVEX = not USE_BIG_M
SAVE_OUTPUT = False

# Contact location and direction.
CONTACT_LOC = np.array([-1, 1])
CONTACT_ANGLE = 0.

# Initial and goal conditions, in order of vx, vy, vth, x, y, th.  The optimal
# trajectory will involve pivoting the cube about its bottom right corner until
# its initial right side ends up on the ground.
x0 = np.array([0., 0., 0., 0., 1., 0.])
x_goal = np.array([0., 0., 0., 2., 1., -np.pi/2])

# Create a polytope.
poly_params = TwoDimensionalPolytopeParams(mass = 1, moment_inertia = 0.01,
    mu_ground = MU_GROUND, vertex_locations = np.array([[1, -1], [1, 1],
                                                        [-1, 1], [-1, -1]]))
polytope = TwoDimensionalPolytope(poly_params)

# Create a simulation system from the polytope, a simulation timestep, and a
# control contact's friction parameter.
sim_system_params = TwoDimensionalSystemParams(dt = DT_SIM, polytope = polytope,
                                               mu_control = 0.5)
sim_system = TwoDSystemMagOnly(sim_system_params, CONTACT_LOC, CONTACT_ANGLE)

# Create a system for trajectory optimization from the polytope, a longer
# timestep, and a control contact's friction parameter.
traj_opt_system_params = TwoDimensionalSystemParams(dt = DT_TRAJ_OPT,
                                        polytope = polytope, mu_control = 0.5)
traj_opt_system = TwoDSystemMagOnly(traj_opt_system_params, CONTACT_LOC,
                                    CONTACT_ANGLE)

# Create an LCS approximation from the trajectory optimization system.
lcs = TwoDSystemLCSApproximation(traj_opt_system)

# For convenience, get the configuration, contacts, and friction sizes.
n = polytope.n_config
p = polytope.n_contacts
k_friction = polytope.n_friction

# Build Q and R matrices.
Q = np.diag([0.1, 0.1, 0.1, 1., 1., 1.])
R = np.diag([0.5])

# Set the initial state of the system.
sim_system.set_initial_state(lcs._convert_lcs_state_to_system_state(x0))

# Set the linearization point of the LCS.
x_i = x0

# Keep track of the solved control inputs, objective costs, and loop times.
inputs, costs, times = np.array([]), np.array([]), np.array([])

for _ in range(LOOPS):
    # Start timer.
    start_time = timeit.default_timer()

    # Get the current state and set the LCS linearization point.
    state_sys = lcs._convert_lcs_state_to_system_state(x_i)
    lcs.set_initial_state(x_i)
    v, q = x_i[:3], x_i[3:]
    controls = np.array([0.])
    lcs.set_linearization_point(q, v, controls)

    # Get the relevant LCS matrices.
    A, B, C, d, G, H, J, l, P = lcs.get_lcs_terms()

    # First calculate the optimal control input for the LCS.
    try:
        # Create a new gurobi model.
        model = gp.Model("traj_opt")

        # Create variables.
        xs = model.addMVar(shape=(LOOKAHEAD+1, 2*n), lb=-np.inf, ub=np.inf,
                            vtype=GRB.CONTINUOUS, name="x_1")
        x_errs = model.addMVar(shape=(LOOKAHEAD, 2*n), lb=-np.inf, ub=np.inf,
                                vtype=GRB.CONTINUOUS, name="x_err_1")
        us = model.addMVar(shape=(LOOKAHEAD, 1), lb=0, ub=INPUT_LIMIT,
                            vtype=GRB.CONTINUOUS, name="u_0")
        lambdas = model.addMVar(shape=(LOOKAHEAD, p*(k_friction+2)),
                                lb=-np.inf, ub=np.inf,
                                vtype=GRB.CONTINUOUS, name="lambda_0")
        ys = model.addMVar(shape=(LOOKAHEAD, p*(k_friction+2)),
                           lb=-np.inf, ub=np.inf,
                           vtype=GRB.CONTINUOUS, name="y_0")

        # Set objective
        obj = 0
        for i in range(LOOKAHEAD):
            obj += x_errs[i, :] @ Q @ x_errs[i, :]
        model.setObjective(obj, GRB.MINIMIZE)

        # Build constraints.
        # -> Dynamics, initial condition, error coordinates, output, etc.
        model.addConstr(xs[0,:] == x_i, name="initial_condition")
        model.addConstrs(
            (xs[i+1,:] == A@xs[i,:] + B@P@us[i,:] + C@lambdas[i,:] + d \
             for i in range(LOOKAHEAD)), name="dynamics")
        model.addConstrs(
            (x_errs[i,:] == xs[i+1,:] - x_goal \
             for i in range(LOOKAHEAD)), name="error_coordinates")
        model.addConstrs(
            (ys[i,:] >= 0 for i in range(LOOKAHEAD)), name="comp_1")
        model.addConstrs(
            (lambdas[i,:] >= 0 for i in range(LOOKAHEAD)), name="comp_2")
        model.addConstrs(
            (ys[i,:] == G@xs[i,:] + H@P@us[i,:] + J@lambdas[i,:] + l \
             for i in range(LOOKAHEAD)), name="output")

        # -> Option 1:  Big M method (convex).
        if USE_BIG_M:
            ss = model.addMVar(shape=(LOOKAHEAD, p*(k_friction+2)),
                               vtype=GRB.BINARY, name="ss")
            model.addConstrs(
                (M1*ss[i,:] >= G@xs[i,:] + H@P@us[i,:] + J@lambdas[i,:] + l \
                 for i in range(LOOKAHEAD)), name="big_m_1")
            model.addConstrs(
                (M2*(1-ss[i,:]) >= lambdas[i,:] for i in range(LOOKAHEAD)),
                name="big_m_2")

        # -> Option 2:  Complementarity constraint (non-convex).
        elif USE_NON_CONVEX:
            model.params.NonConvex = 2
            model.addConstrs(
                (lambdas[i,:] @ ys[i,:] == 0 for i in range(LOOKAHEAD)),
                name="complementarity")

        # Optimize model
        model.optimize()

        # From here, can retrieve the values of the variables via, e.g., x_1.X
        # Print the objective for monitoring progress.
        print('Obj: %g' % model.ObjVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
        pdb.set_trace()

    except AttributeError:
        print('Encountered an attribute error')
        pdb.set_trace()

    # Then use the first control input to apply to the real system, doing a
    # zero-order hold with the trajectory optimization's output.  Use a max() to
    # ensure the control input isn't negative (can happen due to numerics).
    control_input = max(us.X[0], np.array([0.]))
    for _ in range(T_MULTIPLE):
        sim_system.step_dynamics(control_input)

    # Get the latest state to use for the next linearization.
    latest_sys_state = sim_system.state_history[-1, :]
    x_i = lcs._convert_system_state_to_lcs_state(latest_sys_state)

    # Save the control input and objective cost.
    inputs = np.hstack((inputs, control_input))
    costs = np.hstack((costs, model.ObjVal))

    # Record the loop time.
    loop_time = timeit.default_timer() - start_time
    times = np.hstack((times, loop_time))

# Collect the state and control histories.
states = sim_system.state_history
controls = sim_system.control_history

pdb.set_trace()

# Generate a plot of the simulated rollout.
title = "Non-convex Trajectory Optimization" if USE_NON_CONVEX else \
        "Big M Trajectory Optimization"
file_title = 'traj_opt' if USE_NON_CONVEX else 'traj_opt_M'
vis_utils.traj_plot(states, controls, file_title, save=SAVE_OUTPUT, costs=costs,
                    times=times, title=title)

# Generate a gif of the simulated rollout.
vis_utils.animation_gif_polytope(polytope, states, file_title, DT_SIM,
                                 controls=controls, save=SAVE_OUTPUT)

pdb.set_trace()















