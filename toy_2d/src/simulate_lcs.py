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
                                      TwoDSystemForceOnly
from toy_2d.src.two_dim_lcs_approximation import TwoDSystemLCSApproximation


# Fixed parameters
# A few polytope examples.
SQUARE_CORNERS = np.array([[1, -1], [1, 1], [-1, 1], [-1, -1]])
STICK_CORNERS = np.array([[1, 0], [-1, 0]])
RAND_CORNERS = np.array([[0.5, 0], [0.7, 0.5], [0, 0.8], [-1.2, 0], [0, -0.5]])

# Contact location and direction.
CONTACT_LOC = np.array([-1, 1])
CONTACT_ANGLE = 0.0

# Polytope properties
MASS = 1
MOM_INERTIA = 0.01
MU_GROUND = 0.3

# Control properties
MU_CONTROL = 0.5    # Currently, this isn't being used.  The ambition is for
                    # this to help define a set of feasible control forces.

# Simulation parameters.
DT = 0.1          # If a generated trajectory looks messed up, it could be
                    # fixed by making this timestep smaller.

# Initial conditions, in order of x, dx, y, dy, theta, dtheta
# x0_system = np.array([0, 0, 1.5, 0, -1/6 * np.pi, 0])
x0_system = np.array([0, 0, 1, 0, 0, 0])


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
system = TwoDSystemForceOnly(system_params, CONTACT_LOC, CONTACT_ANGLE)
lcs = TwoDSystemLCSApproximation(system)

# Convert the initial system state to an initial LCS state.
x0_lcs = lcs._convert_system_state_to_lcs_state(x0_system)


# Rollout with a fixed (body-frame) force at one of the vertices.
# Apply a force -- give a normal and tangential component.
control = np.tile(np.array([1., 6.]), (10, 1))
lcs.simulate_dynamics_over_horizon(control, x0_lcs, roll_out_lcs=True)

# Collect the state and control histories.
lcs_states = lcs.state_history
controls = lcs.full_control_history
control_forces, control_locs = controls[:, :2], controls[:, 2:]

system_states = []
for lcs_state in lcs_states:
    system_state = lcs._convert_lcs_state_to_system_state(lcs_state)
    system_states.append(system_state)
system_states = np.array(system_states)

pdb.set_trace()

# Generate a plot of the simulated rollout.
vis_utils.traj_plot(system_states, controls, 'simulated_lcs_traj', save=False)
pdb.set_trace()

# Generate a gif of the simulated rollout.
vis_utils.animation_gif_polytope(
    polytope, system_states, 'simulated_lcs_traj', DT,
    controls=(control_forces, control_locs), save=False, force_scale=0.1)

pdb.set_trace()






