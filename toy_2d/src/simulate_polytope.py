"""This file describes a toy 2D system of an arbitrary polytope.

The state keeps track of the center of mass x and y positions plus
the angle theta from the ground's axes to the body's axes, in addition
to time derivatives of all 3 of these quantities:  thus the state
vector is 6-dimensional.
"""

import numpy as np
import pdb
import matplotlib.pyplot as plt

from toy_2d.src import vis_utils
from toy_2d.src import solver_utils
from toy_2d.src.two_dim_polytope import TwoDimensionalPolytopeParams, \
                                        TwoDimensionalPolytope


# Fixed parameters
# A few polytope examples.
SQUARE_CORNERS = np.array([[1, -1], [1, 1], [-1, 1], [-1, -1]])
STICK_CORNERS = np.array([[1, 0], [-1, 0]])
RAND_CORNERS = np.array([[0.5, 0], [0.7, 0.5], [0, 0.8], [-1.2, 0], [0, -0.5]])

# Polytope properties
MASS = 1
MOM_INERTIA = 0.01
MU_GROUND = 0.3

# Control properties
MU_CONTROL = 0.5    # Currently, this isn't being used.  The ambition is for
                    # this to help define a set of feasible control forces.

# Simulation parameters.
DT = 0.002          # If a generated trajectory looks messed up, it could be
                    # fixed by making this timestep smaller.

# Initial conditions, in order of x, dx, y, dy, theta, dtheta
x0 = np.array([0, 0, 1.5, 0, -1/6 * np.pi, 0])


"""Convert an input force and location to forces in the generalized coordinates
of the system, given its state."""
def convert_input_to_generalized_coords(state, control_force, control_loc):
    x, y = state[0], state[2]
    fx, fy = control_force[0], control_force[1]
    u_locx, u_locy = control_loc[0], control_loc[1]

    lever_x = u_locx - x
    lever_y = u_locy - y

    torque = lever_x*fy - lever_y*fx

    return np.array([fx, fy, torque])

"""Do a dynamics step assuming no contact forces."""
def step_dynamics_no_ground_contact(polytope, state, dt, control_force,
                                    control_loc):
    v0 = np.array([state[1], state[3], state[5]]).reshape(3, 1)
    q0 = np.array([state[0], state[2], state[4]]).reshape(3, 1)

    # Some of these quantities are more accurate if calculated at a point in the
    # future, so create a mid-state.
    q_mid_for_M = q0 + dt*v0
    q_mid_for_k = q0 + dt*v0/2

    state_for_M = np.array([q_mid_for_M[0], v0[0],
                            q_mid_for_M[1], v0[1],
                            q_mid_for_M[2], v0[2]])
    state_for_k = np.array([q_mid_for_k[0], v0[0],
                            q_mid_for_k[1], v0[1],
                            q_mid_for_k[2], v0[2]])

    # First, do a forward simulation as if forces were zero.
    M = polytope.get_M_matrix(state_for_M)
    k = polytope.get_k_vector(state_for_k)
    u = convert_input_to_generalized_coords(state, control_force, control_loc)
    u = u.reshape((3, 1))

    v_next = v0 + np.linalg.inv(M) @ (k + u) * dt
    q_next = q0 + dt * v_next

    v_next = v_next.reshape((3,))
    q_next = q_next.reshape((3,))

    next_state = np.array([q_next[0], v_next[0], 
                           q_next[1], v_next[1],
                           q_next[2], v_next[2]])
    return next_state

"""Make one helper function to calculate all of the simulation-related terms of
a system at the state."""
def get_simulation_terms(polytope, state, dt, control_force, control_loc):
    q0 = np.array([state[0], state[2], state[4]]).reshape(3, 1)
    v0 = np.array([state[1], state[3], state[5]]).reshape(3, 1)

    # Some of these quantities are more accurate if calculated at a point in the
    # future, so create a mid-state.  These midstates are defined for the mass
    # matrix, M, and the continuous force vector, k, in the top Equation 27 of
    # Stewart and Trinkle 1996.
    q_mid_for_M = q0 + dt*v0
    q_mid_for_k = q0 + dt*v0/2

    state_for_M = np.array([q_mid_for_M[0], v0[0],
                            q_mid_for_M[1], v0[1],
                            q_mid_for_M[2], v0[2]])
    state_for_k = np.array([q_mid_for_k[0], v0[0],
                            q_mid_for_k[1], v0[1],
                            q_mid_for_k[2], v0[2]])

    M = polytope.get_M_matrix(state_for_M)
    k = polytope.get_k_vector(state_for_k)

    u = convert_input_to_generalized_coords(state, control_force, control_loc)
    u = u.reshape((3, 1))

    D = polytope.get_D_matrix(state)
    N = polytope.get_N_matrix(state)
    E = polytope.get_E_matrix(state)
    Mu = polytope.get_mu_matrix(state)
    phi = polytope.get_phi(state)

    return M, D, N, E, Mu, k, u, q0, v0, phi

"""Given the current state, a timestep, control_force (given as (2,) array for
one control force), and control_loc (given as (2,) array for one location),
simulate the system one timestep into the future.  This function necessarily
determines the ground reaction forces.  This function does NOT check that the
provided control_force and control_loc are valid (i.e. within friction cone and 
acting on the surface of the object)."""
def step_dynamics(polytope, state, dt, control_force, control_loc):
    # First, pretend no contact with the ground occurred.
    next_state = step_dynamics_no_ground_contact(polytope, state, dt,
                                                 control_force, control_loc)
    if min(polytope.get_phi(next_state)) >= 0:
        return next_state

    # At this point, we've established that contact forces are necessary.
    # Need to solve LCP to get proper contact forces -- construct terms.
    p, k_friction = polytope.n_contacts, polytope.n_friction
    M, D, N, E, Mu, k, u, q0, v0, phi = get_simulation_terms(polytope, state,
                                            dt, control_force, control_loc)
    M_i = np.linalg.inv(M)

    # Formulating this inelastic frictional contact dynamics as an LCP is given
    # in Stewart and Trinkle 1996 (see Equation 29 for the structure of the
    # lcp_mat and lcp_vec).
    mat_top = np.hstack((D.T @ M_i @ D, D.T @ M_i @ N, E))
    mat_mid = np.hstack((N.T @ M_i @ D, N.T @ M_i @ N, np.zeros((p,p))))
    mat_bot = np.hstack((-E.T, Mu, np.zeros((p,p))))

    vec_top = D.T @ (v0 + dt * M_i @ k)
    vec_mid = (1./dt) * phi + N.T @ (v0 + dt * M_i @ k)
    vec_bot = np.zeros((p,1))

    lcp_mat = np.vstack((mat_top, mat_mid, mat_bot))
    lcp_vec = np.vstack((vec_top, vec_mid, vec_bot))

    # Get and use the LCP solution.
    lcp_sol = solver_utils.solve_lcp(lcp_mat, lcp_vec)

    Beta = lcp_sol[:p*k_friction].reshape(p*k_friction, 1)
    Cn = lcp_sol[p*k_friction:p*k_friction + p].reshape(p, 1)

    v_next = v0 + M_i @ (N@Cn + D@Beta + dt * (k + u))
    q_next = q0 + dt * v_next

    v_next = v_next.reshape((3,))
    q_next = q_next.reshape((3,))

    next_state = np.array([q_next[0], v_next[0], 
                           q_next[1], v_next[1],
                           q_next[2], v_next[2]])
    return next_state



params = TwoDimensionalPolytopeParams(
    mass = MASS,
    moment_inertia = MOM_INERTIA,
    mu_ground = MU_GROUND,
    vertex_locations = SQUARE_CORNERS
)
polytope = TwoDimensionalPolytope(None, params)


states = x0.reshape(1, 6)

# # rollout with no forces
# for _ in range(20):
#     next_state = step_dynamics(polytope, states[-1, :], DT, np.zeros(2,),
#                                np.zeros(2,))
#     print(next_state)
#     states = np.vstack((states, next_state.reshape(1,6)))

# rollout with a fixed force at one of the vertices
controls = np.zeros((0, 3))
control_forces = np.zeros((0, 2))
control_locs = np.zeros((0, 2))
for _ in range(1250):
    state = states[-1, :]

    # find the third vertex location
    control_loc = polytope.get_vertex_locations_world(state)[2, :]

    # apply the force at a fixed angle relative to the polytope
    theta = state[4]
    ang = np.pi + theta
    control_mag = 0.
    control_vec = control_mag * np.array([-np.cos(ang), -np.sin(ang)])
    generalized_controls = convert_input_to_generalized_coords(state,
                                                    control_vec, control_loc)

    next_state = step_dynamics(polytope, state, DT, #np.zeros(2,), np.zeros(2,))
        control_vec, control_loc)

    states = np.vstack((states, next_state.reshape(1,6)))
    controls = np.vstack((controls, generalized_controls.reshape(1,3)))
    control_forces = np.vstack((control_forces, control_vec.reshape(1,2)))
    control_locs = np.vstack((control_locs, control_loc.reshape(1,2)))

pdb.set_trace()

vis_utils.animation_gif_polytope(polytope, states, 'no_force', DT,
    controls=(control_forces, control_locs))






