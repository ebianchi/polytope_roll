"""This file describes a polytope defined by a set of body-frame vertex
locations, and friction and inertial parameters.  The class includes methods
that allow for easy calculation of useful quantities including:
    - mass matrix
    - contact jacobians
    - continuous forces
    - interbody distances
... which include all terms required for inelastic rigid body simulation via a
linear complementarity problem (LCP).  Notation and formulation largely follow
Stewart and Trinkle, 1996.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pdb
import sympy


@dataclass
class TwoDimensionalPolytopeParams:
    mass: float = 1.0
    moment_inertia: float = 0.1
    mu_ground: float = 1.0
    vertex_locations: np.array = np.array([[1, -1], [1, 1], [-1, 1], [-1, -1]])


class TwoDimensionalPolytope:
    """A 2D Polytope.

    Embedded in this class's methods is the assumption that there is a table at
    height y=0 in the scene and no other objects.

    Properties:
        params:      2D polytope parameters, including geometry, friction, and
                     inertia.
        n_contacts:  The number of vertices of the polytope.
        n_config:    The number of quantities in the state configuration (this
                     is necessarily 3 for x, y, th).
        n_friction:  The number of friction polyhedron sides (this is
                     necessarily 2 for +/- x directions).
        n_dims:      The dimension of the problem (this is necessarily 2).
    """
    params: TwoDimensionalPolytopeParams
    n_contacts: int
    n_config: int
    n_friction: int
    n_dims: int

    def __init__(self, params: TwoDimensionalPolytopeParams):
        self.params = params

        vertex_locations = self.params.vertex_locations
        self.n_contacts = vertex_locations.shape[0]

        # This class represents two dimensional problems.
        self.n_config = 3
        self.n_friction = 2
        self.n_dims = 2

        # Set up a Jacobian function for later calculation of contact Jacobians.
        self.jac_func = self.__set_up_contact_jacobian_function()

    def get_vertex_locations_world(self, state, model=None):
        """Get the locations of the polytope's vertices in world coordinates,
        given the system's current state.  Returns a numpy array of size
        (n_contacts, 2) for the (x,y) position of each vertex."""

        # State is in form [x, dx, y, dy, th, dth].
        x, y, theta = state[0], state[2], state[4]

        p = self.n_contacts
        corners_world = np.zeros((p, 2))
    
        # Since non-linear expressions aren't supported with Gurobi, I'm adding additional constraints, building this now
        if(model):
            y   =   model.addVars(p,lb = -1,ub = 1,name = "u_"+str(i))
            for i in range(p):
                corner_body = self.params.vertex_locations[i, :]

                phi = np.arctan2(corner_body[1], corner_body[0])
                radius = np.sqrt(corner_body[1]**2 + corner_body[0]**2)

                corners_world[i, :] = np.array([x + radius * np.cos(phi + theta),
                                                y + radius * np.sin(phi + theta)])
        else:
            for i in range(p):
                corner_body = self.params.vertex_locations[i, :]

                phi = np.arctan2(corner_body[1], corner_body[0])
                radius = np.sqrt(corner_body[1]**2 + corner_body[0]**2)

                corners_world[i, :] = np.array([x + radius * np.cos(phi + theta),
                                                y + radius * np.sin(phi + theta)])
        return corners_world

    def __get_vertex_velocities_world(self, state):
        """Get the velocities of the polytope's vertices in world coordinates,
        given the system's current state.  Returns a numpy array of size
        (n_contacts, 2) for the (vx,vy) velocity of each vertex."""

        # State is in form [x, dx, y, dy, th, dth].
        x, y, theta = state[0], state[2], state[4]
        vx, vy, vth = state[1], state[3], state[5]

        p = self.n_contacts
        corner_velocities = np.zeros((p, 2))

        for i in range(p):
            corner_body = self.params.vertex_locations[i, :]

            phi = np.atan2(corner_body[1], corner_body[0])
            radius = np.sqrt(corner_body[1]**2 + corner_body[0]**2)

            rotx_contribution = -vth * radius * np.sin(phi + theta)
            roty_contribution = vth * radius * np.cos(phi + theta)

            corner_velocities[i, :] = np.array([vx + rotx_contribution,
                                                vy + roty_contribution])
        return corner_velocities

    def __set_up_contact_jacobian_function(self):
        """Create a callable function that returns the (2, 3) jacobian
        representing the partial derivative of a vertex's world-frame velocity
        with respect to the polytope's world-frame velocity.  This left-
        multiplied by a unit direction vector(s) yields the normal
        (corresponding to a vertical unit vector) and tangential (corresponding
        to horizontal unit vectors) contact jacobians later used for simulation.
        """

        # First, write a symbolic expression of a vertex's velocity given its
        # location relative to the object's CoM and the object's velocities.
        px_body, py_body, theta, vx, vy, vth = sympy.symbols(
                                            'px_body py_body theta vx vy vth')

        phi = sympy.atan2(py_body, px_body)
        radius = sympy.sqrt(px_body**2 + py_body**2)

        rotx_contrib = -vth * radius * sympy.sin(phi + theta)
        roty_contrib = vth * radius * sympy.cos(phi + theta)

        corner_vel = sympy.Matrix([vx + rotx_contrib, vy + roty_contrib])

        # Second, get a callable function for the contact Jacobian of the vertex
        # (this is the gradient of the vertex's velocity w.r.t. the system's
        # velocities [vx, vy, vth]).  This will be of shape (n_dims, n_config)
        # or (2, 3).
        contact_jac = sympy.Matrix([corner_vel.diff(vx).T,
                                    corner_vel.diff(vy).T,
                                    corner_vel.diff(vth).T]).T
        jac_func = sympy.lambdify([px_body, py_body, theta, vx, vy, vth],
                                  contact_jac, 'numpy')

        return jac_func

    def __calculate_contact_jacobian_along_projections(self, state, projs):
        """Calculate the contact jacobian of all vertices along given projection
        direction(s)."""

        # State is in form [x, dx, y, dy, th, dth].
        x, y, theta = state[0], state[2], state[4]
        vx, vy, vth = state[1], state[3], state[5]

        # The resulting matrix will be of size (n_config, n_contacts * n_projs).
        n = self.n_config
        p = self.n_contacts

        contact_jac = np.zeros((n, 0))

        # We can use the contact Jacobian function for each vertex.
        for vertex_i in range(p):
            px_body, py_body = self.params.vertex_locations[vertex_i, :]

            jac = self.jac_func(px_body, py_body, theta, vx, vy, vth)

            contact_jac = np.hstack((contact_jac, (projs @ jac).T))

        return contact_jac

    def get_M_matrix(self, _):
        """Calculate the mass matrix.  Note that since the polytope is not
        articulated and is represented by a mass at its origin, the mass matrix
        is not state dependent."""

        m = self.params.mass
        I = self.params.moment_inertia
        return np.diag([m, m, I])

    def get_D_matrix(self, state):
        """Calculate the tangential contact jacobian of all vertices. Returns a
        numpy array of size (n_config, n_contacts * n_friction) or (3,
        n_contacts * 2) for each vertex's positive and negative x-direction
        effect on the 3 generalized coordinates."""

        # Use projection in +/- x-directions.
        projs = np.array([[1, 0], [-1, 0]])
        return self.__calculate_contact_jacobian_along_projections(state, projs)

    def get_N_matrix(self, state):
        """Calculate the normal contact jacobian of all vertices. Returns a
        numpy array of size (n_config, n_contacts) or (3, n_contacts) for each
        vertex's positive y-direction effect on the 3 generalized coordinates.
        """

        # Use projection in y-direction.
        proj = np.array([[0, 1]])
        return self.__calculate_contact_jacobian_along_projections(state, proj)

    def get_mu_matrix(self, _):
        """Calculate the friction matrix.  We assume this is not state
        dependent."""

        mu = self.params.mu_ground
        p = self.n_contacts
        return mu * np.eye(p)

    def get_E_matrix(self, _):
        """Calculate the matrix E, defined by a block diagonal matrix composed
        of n_contacts repeats of ones vectors of size (n_friction, 1), or (2, 1)
        for this 2D example.  This is fixed and thus is not state dependent."""

        p = self.n_contacts
        k = self.n_friction

        return np.kron(np.eye(p, dtype=int), np.ones((k,1)))

    def get_C_matrix(self, _):
        """Calculate the (n_config, n_config) Coriolis/centrifugal matrix.  This
        is defined as:

            C = grad_q(M*v) * v - 0.5 * (grad_q(M*v))^T * v

        Note that since the polytope's mass matrix (and generalized velocities)
        are not configuration dependent, this is zero."""

        n = self.n_config
        return np.zeros((n, n))

    def get_G_vector(self, _):
        """Calculate the (n_config, 1) vector of gravitational forces."""

        m = self.params.mass
        g = -9.81
        return np.array([0, -m*g, 0]).reshape(self.n_config, 1)

    def get_k_vector(self, state):
        """Calculate the (n_config, 1) vector of continuous forces.  This vector
        aggregates all contributions due to gravity, Coriolis, and centrifugal
        forces, and it is defined as:

            k = -C*v - G
        """

        C = self.get_C_matrix(state)
        G = self.get_G_vector(state)

        vx, vy, vth = state[1], state[3], state[5]
        v = np.array([vx, vy, vth]).reshape(self.n_config, 1)

        return -C @ v - G

    def get_phi(self, state):
        """Calculate the (n_contacts, 1) vector of interbody distances between
        each vertex and the ground."""

        corners = self.get_vertex_locations_world(state)
        return corners[:, 1].reshape(self.n_contacts, 1)

