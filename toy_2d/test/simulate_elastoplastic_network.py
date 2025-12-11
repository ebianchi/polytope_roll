"""This file describes a toy 2D system of an arbitrary spring particle network.

The state keeps track of each particle's x and y location in addition to time
derivatives of all of these quantities:  thus the state vector is
(4*n_particles)-dimensional.
"""

import numpy as np

from toy_2d.src import vis_utils
from toy_2d.src.two_dim_spring_network import (
    TwoDimensionalParticleParams,
    TwoDimensionalParticle,
    TwoDimensionalElastoPlasticNetworkParams,
    TwoDimensionalElastoPlasticNetwork,
)
from toy_2d.src.two_dim_system import (
    TwoDimensionalSystemParams,
    TwoDimensionalSystem,
)


# Fixed parameters
# Contact location and direction.
CONTACT_LOC = np.array([0, 0])
CONTACT_ANGLE = 0.0

# Particle properties
MASS = 1
MU_GROUND = 0.1

# Control properties
MU_CONTROL = 0.5  # Currently, this isn't being used.  The ambition is for
# this to help define a set of feasible control forces.

# Simulation parameters.
DT = 0.002  # If a generated trajectory looks messed up, it could be
# fixed by making this timestep smaller.

# Initial conditions, in order of x, dx, y, dy for 1 particle, then next, etc.
VX, VY = 3.0, 3.0
x0 = np.array(
    [-0.5, VX, 1, VY, -0.5, VX, 2, VY, 0.5, VX, 1.5, VY, 1, VX, 2, VY]
)
# A simpler system for debugging:  two vertically stacked particles.
# VX, VY = 0.0, 3.0
# x0 = np.array([0, VX, 1, VY, 0, VX, 2, VY])
states = x0.reshape(1, -1)


# Create multiple identical particles to go into a network.
particle_params = TwoDimensionalParticleParams(
    mass=MASS,
    mu_ground=MU_GROUND,
)
n_pts = x0.shape[0] // 4
connections = [(0, 1), (1, 2), (0, 2), (2, 3)]
n_connections = len(connections)
network_params = TwoDimensionalElastoPlasticNetworkParams(
    particles=[TwoDimensionalParticle(particle_params) for _ in range(n_pts)],
    connections=connections,
    spring_constants=[150] * n_connections,
    rest_lengths=[0.5] * n_connections,
    yield_forces=[50] * n_connections,
    damping=[1.0] * n_connections,
)
network = TwoDimensionalElastoPlasticNetwork(network_params)


# Create a system from the network, a simulation timestep, and a control
# contact's friction parameter.
system_params = TwoDimensionalSystemParams(
    dt=DT, polytope=network, mu_control=MU_CONTROL
)
system = TwoDimensionalSystem(system_params)


# Compute the hidden state so the springs begin with no tension.
hidden_state = np.zeros(len(network_params.connections))
for i, conn in enumerate(network_params.connections):
    p1_idx = conn[0]
    p2_idx = conn[1]
    p1_x = x0[4 * p1_idx]
    p1_y = x0[4 * p1_idx + 2]
    p2_x = x0[4 * p2_idx]
    p2_y = x0[4 * p2_idx + 2]
    vec_1_to_2 = np.array([p2_x - p1_x, p2_y - p1_y])
    length_1_to_2 = np.linalg.norm(vec_1_to_2)
    hidden_state[i] = length_1_to_2 - network_params.rest_lengths[i]

# Rollout with a fixed (body-frame) force at one of the vertices.
system.set_initial_state(x0, hidden_state)
control = np.zeros(4)  # Apply no force.
for _ in range(1200):
    system.step_dynamics(control)

# Collect the state and control histories.
states = system.state_history
hidden_states = system.hidden_state_history
controls = system.control_history
control_forces, control_locs = controls[:, :2], controls[:, 2:]

# Generate a gif of the simulated rollout.
vis_utils.animation_gif_polytope(
    network,
    states,
    "simulated_elastoplastic_particles",
    DT,
    hidden_states=hidden_states,
    controls=(control_forces, control_locs),
    save=True,
    force_scale=10.0,
)

# Generate a plot of the simulated rollout.
config_names = []
vis_utils.traj_plot(
    states,
    controls,
    "simulated_elastoplastic_particles",
    save=True,
    config_names=[
        f"{dir}{i}"
        for i in range(len(network_params.particles))
        for dir in ["x", "y"]
    ],
)

breakpoint()
