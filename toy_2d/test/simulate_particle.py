"""This file describes a toy 2D system of a single particle.

The state keeps track of the center of mass x and y positions in addition to
time derivatives of both of these quantities:  thus the state vector is 4-
dimensional.
"""

import numpy as np

from toy_2d.src import vis_utils
from toy_2d.src.two_dim_system import (
    TwoDimensionalSystemParams,
    TwoDimensionalSystem,
)
from toy_2d.src.two_dim_spring_network import (
    TwoDimensionalParticleParams,
    TwoDimensionalParticle,
)


# Fixed parameters
# Contact location and direction.
CONTACT_LOC = np.array([0, 0])
CONTACT_ANGLE = 0.0

# Particle properties
MASS = 1
MU_GROUND = 0.3

# Control properties
MU_CONTROL = 0.5  # Currently, this isn't being used.  The ambition is for
# this to help define a set of feasible control forces.

# Simulation parameters.
DT = 0.002  # If a generated trajectory looks messed up, it could be
# fixed by making this timestep smaller.

# Initial conditions, in order of x, dx, y, dy
x0 = np.array([-0.5, 5, 1.5, 0])
states = x0.reshape(1, 4)


# Create a particle.
particle_params = TwoDimensionalParticleParams(
    mass=MASS,
    mu_ground=MU_GROUND,
)
particle = TwoDimensionalParticle(particle_params)

# Create a system from the particle, a simulation timestep, and a control
# contact's friction parameter.
system_params = TwoDimensionalSystemParams(
    dt=DT, polytope=particle, mu_control=MU_CONTROL
)
system = TwoDimensionalSystem(system_params)


# Rollout with a fixed (body-frame) force at one of the vertices.
system.set_initial_state(x0)
control = np.zeros(4)  # Apply no force.
for _ in range(1250):
    system.step_dynamics(control)

# Collect the state and control histories.
states = system.state_history
controls = system.control_history
control_forces, control_locs = controls[:, :2], controls[:, 2:]

# Generate a gif of the simulated rollout.
vis_utils.animation_gif_polytope(
    particle,
    states,
    "simulated_particle",
    DT,
    controls=(control_forces, control_locs),
    save=False,
    force_scale=10.0,
)

# Generate a plot of the simulated rollout.
vis_utils.traj_plot(
    states, controls, "simulated_particle", save=False, config_names=["x", "y"]
)

breakpoint()
