"""This file describes a toy 2D system of an arbitrary polytope.

The state keeps track of the center of mass x and y positions plus the angle
theta from the ground's axes to the body's axes, in addition to time derivatives
of all 3 of these quantities:  thus the state vector is 6-dimensional.
"""

import numpy as np
import pdb
import matplotlib.pyplot as plt

from toy_2d.src import vis_utils
from toy_2d.src.two_dim_polytope import TwoDimensionalPolytopeParams, \
                                        TwoDimensionalPolytope
from toy_2d.src.two_dim_system import TwoDimensionalSystemParams, \
                                      TwoDimensionalSystem


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
states = x0.reshape(1, 6)


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
system = TwoDimensionalSystem(system_params)


# Rollout with a fixed (body-frame) force at one of the vertices.
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
    generalized_controls = system.convert_input_to_generalized_coords(state,
                                                    control_vec, control_loc)

    next_state = system.step_dynamics(state, control_vec, control_loc)

    states = np.vstack((states, next_state.reshape(1,6)))
    controls = np.vstack((controls, generalized_controls.reshape(1,3)))
    control_forces = np.vstack((control_forces, control_vec.reshape(1,2)))
    control_locs = np.vstack((control_locs, control_loc.reshape(1,2)))

pdb.set_trace()

# Generate a gif of the simulated rollout.
vis_utils.animation_gif_polytope(polytope, states, 'no_force', DT,
    controls=(control_forces, control_locs))






