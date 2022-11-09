"""This file formulates a mixed-integer optimization problem of the form:

           min         sum_{k=0}^{N-1} x_k^T Q x_k + u_k^T R u_k + x_N^T Q_N x_N
    x_k, lambda_k, u_k

        such that   x_{k+1} = A x_k + B u_k + C lambda_k + d
                    M_1 s_k >= G x_k + H u_k + J lambda_k + l >= 0
                    M_2 (1 - s_k) >= lambda_k >= 0
                    s_k in {0, 1}^{p(k+2)}
                    x_0 = x(0)

...where the mixed integer portion is contained in the p*(k+2) vector of binary
variables, s_k.  The scalars M_1 and M_2 should be large numbers (used for the
big M method).  Here we should note that the above formulation will want to send
the system to x_N = 0.  For a different goal position, switch the x_k's above to
(x_k - x^*) for some goal x^*.
"""

import numpy as np
import scipy.sparse as sp
import pdb

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
T_MULTIPLE = 10
DT_TRAJ_OPT = DT_SIM * T_MULTIPLE
MU_GROUND = 0.5
M1 = 1e10
M2 = 1e10
LOOPS = 350
INPUT_LIMIT = 5.

# Contact location and direction.
CONTACT_LOC = np.array([-1, 1])
CONTACT_ANGLE = 0.

# Initial and goal conditions, in order of vx, vy, vth, x, y, th.  The optimal
# trajectory will involve pivoting the cube about its bottom left corner until
# its initial left side ends up on the ground.
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
Q = np.diag([0.1, 0.1, 0.1, 1., 1., 1])
R = np.diag([1.])

# Set the initial state of the system.
sim_system.set_initial_state(lcs._convert_lcs_state_to_system_state(x0))

# Set the linearization point of the LCS.
x_i = x0

# Keep track of the solved control inputs.
inputs = np.array([])

for _ in range(LOOPS):
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
        x_1 = model.addMVar(shape=2*n, lb=-np.inf, ub=np.inf,
                            vtype=GRB.CONTINUOUS, name="x_1")
        x_err_1 = model.addMVar(shape=2*n, lb=-np.inf, ub=np.inf,
                                vtype=GRB.CONTINUOUS, name="x_err_1")
        u_0 = model.addMVar(shape=1, lb=0, ub=INPUT_LIMIT,
                            vtype=GRB.CONTINUOUS, name="u_0")
        lambda_0 = model.addMVar(shape=p*(k_friction+2), lb=-np.inf, ub=np.inf,
                                 vtype=GRB.CONTINUOUS, name="lambda_0")
        # s_0 = model.addMVar(shape=p*(k_friction+2), vtype=GRB.BINARY, name="s_0")
        y_0 = model.addMVar(shape=p*(k_friction+2), lb=-np.inf, ub=np.inf,
                            vtype=GRB.CONTINUOUS, name="y_0")

        # Set objective
        model.setObjective(x_err_1 @ Q @ x_err_1, GRB.MINIMIZE)

        # Build constraints.
        # -> The below are the 6 required to use the "big-M" method in Alp's
        #    paper to get this problem to be convex.  The below don't all work
        #    together, so see further down for the nonconvex constraints that
        #    actually run.
        model.addConstr(x_1 == A@x_i + B@P@u_0 + C@lambda_0 + d, name="dynamics")  #1
        # model.addConstr(M1*s_0 >= G@x0 + H@P@u_0 + J@lambda_0 + l, name="big_m_1") #2
        # model.addConstr(M2*(s_0-1) >= lambda_0, name="big_m_2") #3
        # model.addConstr(G@x0 + H@P@u_0 + J@lambda_0 + l >= 0, name="comp_1")  #4
        # model.addConstr(lambda_0 >= 0, name="comp_2")  #5
        model.addConstr(x_err_1 == x_1 - x_goal, name="error_coordinates")  #6

        # Constraint combinations that work:
        # - 1, 2, 3, 6 -- but not additionally with 4 or 5 (M1=M2=1e2)
        # - 1, 4, 5, 6 -- but not additionally with 2 or 3 (M1=M2=1e2)
        # - 1, 2, 3, 5, 6 -- but not additionally with 4 (M1=M2=1e6)
        # Try the below instead:
        model.addConstr(y_0 >= 0, name="comp_1")
        model.addConstr(lambda_0 >= 0, name="comp_2")
        model.addConstr(y_0 == G@x_i + H@P@u_0 + J@lambda_0 + l, name="output")
        model.params.NonConvex = 2
        model.addConstr(lambda_0 @ y_0 == 0, name="complementarity")
        # model.addConstr(u_0 <= INPUT_LIMIT, name="input_limits")

        # Optimize model
        model.optimize()

        # From here, can retrieve the values of the variables via, e.g., x_1.X
        # Print the objective for monitoring progress.
        print('Obj: %g' % model.ObjVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    # Then use the control input to apply to the real system, doing a zero-order
    # hold with the trajectory optimization's output.
    control_input = u_0.X
    for _ in range(T_MULTIPLE):
        sim_system.step_dynamics(control_input)

    # Save the control input.
    inputs = np.hstack((inputs, control_input))

    # Get the latest state to use for the next linearization.
    latest_sys_state = sim_system.state_history[-1, :]
    x_i = lcs._convert_system_state_to_lcs_state(latest_sys_state)


# Collect the state and control histories.
states = sim_system.state_history
controls = sim_system.control_history

pdb.set_trace()

# Generate a plot of the simulated rollout.
vis_utils.traj_plot(states, controls, 'traj_opt', save=True)

# Generate a gif of the simulated rollout.
vis_utils.animation_gif_polytope(polytope, states, 'traj_opt', DT_SIM,
                                 controls=controls, save=True)















