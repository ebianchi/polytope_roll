"""This file describes a 2D spring network, defined by a set of particles (with
mass and friction) connected by springs.  The class includes methods that allow
for easy calculation of useful quantities including:
    - mass matrix
    - contact jacobians
    - continuous forces
    - interbody distances
... which include all terms required for inelastic rigid body simulation via a
linear complementarity problem (LCP).  Notation and formulation largely follow
Stewart and Trinkle, 1996.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from toy_2d.src.two_dim_polytope import (
    TwoDimensionalPolytopeParams,
    TwoDimensionalPolytope,
)


@dataclass
class TwoDimensionalParticleParams(TwoDimensionalPolytopeParams):
    mass: float = 1.0
    mu_ground: float = 0.5
    moment_inertia: float = None
    vertex_locations: np.array = None

    def __post_init__(self):
        assert (
            self.moment_inertia is None
        ), "Point particle should not have moment of inertia."
        assert (
            self.vertex_locations is None
        ), "Point particle should not have vertex locations."


class TwoDimensionalParticle(TwoDimensionalPolytope):
    """A point particle that can exist in 2D space and collide with the ground.
    It state is represented by its (x, y) position and (dx, dy) velocity, in
    order [x dx y dy].  This is defined as a subclass of TwoDimensionalPolytope
    for ease of integration into existing simulation code."""

    params: TwoDimensionalParticleParams

    def __init__(self, params: TwoDimensionalParticleParams):
        self.params = params

        # This class represents two dimensional problems.
        self.n_contacts = 1
        self.n_config = 2
        self.n_friction = 2
        self.n_dims = 2

        # The Jacobian of "a point on the particle", i.e. the particle itself,
        # with respect to the particle's world-frame velocity is simply the
        # identity.
        self.d_pdot_d_qdot_jac = np.eye(2)

    def get_vertex_locations_world(self, state, for_visualization=False):
        """Get the locations of the particle's single vertex in world
        coordinates, given the system's current state (this is trivial).
        Returns a numpy array of size (1, 2)."""

        # State is in form [x, dx, y, dy].
        x, y = state[0], state[2]

        return np.array([x, y]).reshape(1, 2)

    def _calculate_contact_jacobian_along_projections(self, _state, projs):
        """Calculate the contact jacobian of the particle along given projection
        direction(s).  Note that since the particle's state is the system's
        state itself, this is a constant function for a given projection."""

        # The resulting matrix will be of size (n_config, n_contacts * n_projs),
        # which is (n_config, n_projs) for a particle since n_contacts = 1.
        n = self.n_config
        k = projs.shape[0]
        contact_jac = np.zeros((n, k))

        # There is just one contact for a particle, and the jacobian is not
        # state-dependent.
        contact_jac = (projs @ self.d_pdot_d_qdot_jac).T

        return contact_jac

    def get_M_matrix(self, _):
        """Calculate the mass matrix.  The mass matrix is not state
        dependent."""

        m = self.params.mass
        return np.diag([m, m])

    def get_G_vector(self, _):
        """Calculate the (n_config, 1) vector of gravitational forces."""

        m = self.params.mass
        g = -9.81
        return np.array([0, -m * g]).reshape(self.n_config, 1)

    def get_k_vector(self, state):
        """Calculate the (n_config, 1) vector of continuous forces.  This vector
        aggregates all contributions due to gravity, Coriolis, and centrifugal
        forces, and it is defined as:

            k = -C*v - G
        """

        C = self.get_C_matrix(state)
        G = self.get_G_vector(state)

        vx, vy = state[1], state[3]
        v = np.array([vx, vy]).reshape(self.n_config, 1)

        return -C @ v - G

    def get_phi(self, state):
        """Calculate the (n_contacts, 1) vector of interbody distances between
        the particle and the ground."""

        # State is in form [x, dx, y, dy].
        return np.array([state[2]]).reshape(self.n_contacts, 1)


@dataclass
class TwoDimensionalSpringNetworkParams:
    particles: List[TwoDimensionalParticle] = field(default_factory=list)
    connections: List[tuple] = field(default_factory=list)
    spring_constants: List[float] = field(default_factory=list)
    rest_lengths: List[float] = field(default_factory=list)

    def __post_init__(self):
        assert len(self.particles) > 0, "There must be at least one mass."
        assert (
            len(self.connections)
            == len(self.spring_constants)
            == len(self.rest_lengths)
        ), "There must be a spring constant and length for each connection."
        for conn in self.connections:
            assert (
                conn[0] < len(self.particles)
                and conn[1] < len(self.particles)
                and conn[0] != conn[1]
            ), (
                "Connection indices must be unique pairs of valid particle "
                + "indices."
            )
        for k in self.spring_constants:
            assert k >= 0, "Spring constants cannot be negative."


class TwoDimensionalSpringNetwork(TwoDimensionalParticle):
    """A 2D spring network.

    Embedded in this class's methods is the assumption that there is a table at
    height y=0 in the scene and no other objects (the particles cannot exert
    contact forces on each other, only with the ground).

    Impose the state structure as follows:
        state = [x1, dx1, y1, dy1, x2, dx2, y2, dy2, ..., xn, dxn, yn, dyn]

    Properties:
        params:         2D spring network parameters.
    """

    params: TwoDimensionalSpringNetworkParams

    def __init__(self, params: TwoDimensionalSpringNetworkParams):
        self.params = params

        # This class represents two dimensional problems.
        self.n_contacts = len(self.params.particles)
        self.n_config = 2 * self.n_contacts
        self.n_friction = 2
        self.n_dims = 2

    def _get_particle_state_from_system_state(
        self, system_state, particle_index
    ):
        """Extract the state of a single particle from the full system state."""
        return system_state[
            2
            * self.n_dims
            * particle_index : 2
            * self.n_dims
            * (particle_index + 1)
        ]

    def get_vertex_locations_world(self, state, for_visualization=False):
        """Get the locations of the particle's single vertex in world
        coordinates, given the system's current state (this is trivial).
        Returns a numpy array of size (1, 2)."""

        vertex_locs = np.zeros((self.n_contacts, 2))

        for i in range(self.n_contacts):
            particle_state = self._get_particle_state_from_system_state(
                state, i
            )
            vertex_locs[i] = self.params.particles[
                i
            ].get_vertex_locations_world(
                particle_state, for_visualization=for_visualization
            )

        return vertex_locs

    def get_M_matrix(self, state):
        """Calculate the mass matrix.  This is a block diagonal matrix with each
        particle's mass matrix along the diagonal."""
        M = np.zeros((self.n_config, self.n_config))
        for i in range(self.n_contacts):
            particle_state = self._get_particle_state_from_system_state(
                state, i
            )
            M[2 * i : 2 * (i + 1), 2 * i : 2 * (i + 1)] = self.params.particles[
                i
            ].get_M_matrix(particle_state)
        return M

    def get_D_matrix(self, state):
        """Calculate the tangential contact jacobian of all vertices. Returns a
        numpy array of size (n_config, n_contacts * n_friction) with block
        structure for each particle's (2, 2) portion."""

        # Sizes for each particle.
        p = self.params.particles[0].n_contacts
        k = self.params.particles[0].n_friction

        D = np.zeros((self.n_config, self.n_contacts * self.n_friction))
        for i in range(self.n_contacts):
            particle_state = self._get_particle_state_from_system_state(
                state, i
            )
            D[2 * i : 2 * (i + 1), i * (p * k) : (i + 1) * (p * k)] = (
                self.params.particles[i].get_D_matrix(particle_state)
            )
        return D

    def get_N_matrix(self, state):
        """Calculate the normal contact jacobian of all vertices. Returns a
        numpy array of size (n_config, n_contacts) with block structure for each
        particle's (2, 1) portion."""

        # Sizes for each particle.
        p = self.params.particles[0].n_contacts

        N = np.zeros((self.n_config, self.n_contacts))
        for i in range(self.n_contacts):
            particle_state = self._get_particle_state_from_system_state(
                state, i
            )
            N[2 * i : 2 * (i + 1), i * p : (i + 1) * p] = self.params.particles[
                i
            ].get_N_matrix(particle_state)
        return N

    def get_mu_matrix(self, _):
        """Calculate the friction matrix.  This is a block diagonal matrix with
        each particle's mu matrix along the diagonal."""

        p = self.n_contacts
        mu = np.eye(self.n_contacts)
        for i in range(self.n_contacts):
            mu[i, i] *= self.params.particles[i].params.mu_ground
        return mu * np.eye(p)

    def get_G_vector(self, state):
        """Calculate the (n_config, 1) vector of gravitational forces.  This is
        defined as stacked blocks for each particle's G vector."""

        g = np.zeros((self.n_config))
        for i in range(self.n_contacts):
            particle_state = self._get_particle_state_from_system_state(
                state, i
            )
            g[2 * i : 2 * (i + 1)] = (
                self.params.particles[i].get_G_vector(particle_state).reshape(2)
            )

        return g.reshape(self.n_config, 1)

    def get_k_vector(self, state):
        """Calculate the (n_config, 1) vector of continuous forces.  This starts
        with stacked blocks for each particle's k vector:

            k_particle = -C*v - G

        Then, spring forces are added based on the connections of the spring
        network."""
        # Start with per-particle continuous forces from Coriolis/centrifugal
        # and gravity.
        k = np.zeros((self.n_config))
        for i in range(self.n_contacts):
            particle_state = self._get_particle_state_from_system_state(
                state, i
            )
            k[2 * i : 2 * (i + 1)] = (
                self.params.particles[i].get_k_vector(particle_state).reshape(2)
            )

        # Add spring forces.
        for i_spring in range(len(self.params.connections)):
            conn = self.params.connections[i_spring]
            p1_idx = conn[0]
            p2_idx = conn[1]
            k_spring = self.params.spring_constants[i_spring]
            rest_length = self.params.rest_lengths[i_spring]

            # Get particle positions.
            p1 = self._get_particle_state_from_system_state(state, p1_idx)[
                [0, 2]
            ]
            p2 = self._get_particle_state_from_system_state(state, p2_idx)[
                [0, 2]
            ]

            unit_1_to_2 = p2 - p1
            unit_1_to_2 /= np.linalg.norm(unit_1_to_2)

            # Compute the directional spring force acting on particle 1.  The
            # force on particle 2 is equal and opposite.
            f_on_1 = k_spring * (p2 - p1 - rest_length * unit_1_to_2)

            # Apply spring forces to each particle's k vector.
            k[2 * p1_idx : 2 * (p1_idx + 1)] += f_on_1
            k[2 * p2_idx : 2 * (p2_idx + 1)] -= f_on_1

        return k.reshape(self.n_config, 1)

    def get_phi(self, state):
        """Calculate the (n_contacts, 1) vector of interbody distances between
        each vertex and the ground."""

        return TwoDimensionalPolytope.get_phi(self, state)
