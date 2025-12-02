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
    TwoDimensionalSpringNetworkParams,
    TwoDimensionalSpringNetwork,
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
MU_GROUND = 0.0

# Control properties
MU_CONTROL = 0.5  # Currently, this isn't being used.  The ambition is for
# this to help define a set of feasible control forces.

# Simulation parameters.
DT = 0.002  # If a generated trajectory looks messed up, it could be
# fixed by making this timestep smaller.

# Initial conditions, in order of x, dx, y, dy for 1 particle, then next, etc.
x0 = np.array([-0.5, 0, 1, 0, -0.5, 0, 2, 0, 0.5, 0, 1.5, 0, 1, 0, 2, 0])
states = x0.reshape(1, -1)


# Create multiple identical particles to go into a network.
particle_params = TwoDimensionalParticleParams(
    mass=MASS,
    mu_ground=MU_GROUND,
)
n_pts = x0.shape[0] // 4
network_params = TwoDimensionalSpringNetworkParams(
    particles=[TwoDimensionalParticle(particle_params) for _ in range(n_pts)],
    connections=[(0, 1), (1, 2), (0, 2), (2, 3)],
    spring_constants=[150] * n_pts,
    rest_lengths=[1.0] * n_pts,
)
network = TwoDimensionalSpringNetwork(network_params)


# Create a system from the network, a simulation timestep, and a control
# contact's friction parameter.
system_params = TwoDimensionalSystemParams(
    dt=DT, polytope=network, mu_control=MU_CONTROL
)
system = TwoDimensionalSystem(system_params)


# Rollout with a fixed (body-frame) force at one of the vertices.
system.set_initial_state(x0)
control = np.zeros(4)  # Apply no force.
for _ in range(1500):
    system.step_dynamics(control)

# Collect the state and control histories.
states = system.state_history
controls = system.control_history
control_forces, control_locs = controls[:, :2], controls[:, 2:]

# Generate a gif of the simulated rollout.
vis_utils.animation_gif_polytope(
    network,
    states,
    "simulated_particles",
    DT,
    controls=(control_forces, control_locs),
    save=True,
    force_scale=10.0,
)

# Generate a plot of the simulated rollout.
config_names = []
vis_utils.traj_plot(
    states,
    controls,
    "simulated_particles",
    save=False,
    config_names=[
        f"{dir}{i}"
        for i in range(len(network_params.particles))
        for dir in ["x", "y"]
    ],
)

breakpoint()
