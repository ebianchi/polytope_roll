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

from scipy.spatial import ConvexHull


@dataclass
class TwoDimensionalPolytopeParams:
    mass: float = 1.0
    moment_inertia: float = 0.1
    mu_ground: float = 1.0
    vertex_locations: np.array = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])


class TwoDimensionalPolytope:
    """A 2D Polytope.

    Embedded in this class's methods is the assumption that there is a table at
    height y=0 in the scene and no other objects.

    Note:  For many of the geometry-related parameters, see the following
    drawing for definitions of the parameters.  Use the below as an exemplifying
    polytope, assuming theta=0 for this convex hull below:

                    4 -------- 5
                   /            \
                  /              \
                 /                \
                3 ----- O          0
                |     /           / \
                |   /       ---- 1   P
                | /    -----
                2 -----..........................G

    Given the above convex polytope with origin O and vertices {0, 1, 2, 3, 4,
    5}, we have the following definitions, shown with an example on the left and
    a general way of indexing on the right:

        radii[2] = length(O2)    ==>  radii[i] = length(Oi)
        sides[2] = length(23)    ==>  sides[i] = length(i{i+1})
        alphas[2] = angle(2O3)   ==>  alphas[i] = angle(iO{i+1})
        betas[2]  = angle(32O)   ==>  betas[i] = angle({i+1}iO)
        gammas[2] = angle(O32)   ==>  gammas[i] = angle(O{i+1}i)
        psis[0] = angle(P01)     ==>  psis[i] = pi - betas[i] - gammas[i-1]

    Note that so far these definitions are agnostic to how the state variable,
    theta, is defined.  We introduce a notion of theta when we define the angle
    delta, defined as the angle between the body frame's horizontal axis and the
    face just counter- clockwise from the vertex that would come into contact at
    theta = 0:

        delta = angle(G21)

    ...where G is some point at (x, y)_{body frame} = (>0, 0).

    Properties:
        params:         2D polytope parameters, including geometry, friction,
                        and inertia.
        hull_vertices:  A pared down array of vertices consisting of the convex
                        hull of the polytope, in clockwise order, stored as a
                        numpy array of size (n_contacts, 2) for [x, y] location.
        n_contacts:     The number of vertices of the polytope's convex hull.
        n_config:       The number of quantities in the state configuration
                        (this is necessarily 3 for x, y, th).
        n_friction:     The number of friction polyhedron sides (this is
                        necessarily 2 for +/- x directions).
        n_dims:         The dimension of the problem (this is necessarily 2).
        alphas:         The interior angles at the polytope's origin of each
                        triangle defined by two adjacent vertices of the convex
                        hull and the polytope's origin.  This is stored in order
                        [alpha_01, alpha_12, ... alphap_{p-1}0].
        betas:          The interior angles at the most counter-clockwise
                        exterior vertex of each triangle defined by two adjacent
                        vertices of the convex hull and the polytope's origin.
                        This is stored in the same order as the alphas.
        gammas:         The interior angles at the most clockwise exterior
                        vertex of each triangle defined by two adjacent vertices
                        of the convex hull and the polytope's origin. This is
                        stored in the same order as the alphas.
        sides:          The side lengths of the convex hull, stored in the same
                        order as the alphas.
        psis:           The angle range of motion that each corner of the convex
                        hull has as the ground pivot point.  This is stored in
                        the order [psi_0, psi_1, ... psi_{p-1}].
        lowest_vertex:  The index of the lowest vertex of the convex hull when
                        theta = 0.  If there are two lowest vertices (it cannot
                        be more since the convex hull is the minimum vertex
                        set), then this index is the most clockwise of the two. 
        delta:          The angle between the ground and the face clockwise of
                        the lowest_vertex when theta = 0.
    """
    params: TwoDimensionalPolytopeParams

    def __init__(self, params: TwoDimensionalPolytopeParams):
        self.params = params

        # Compute the convex hull of the polytope for more efficient simulation
        # later and for easier control.
        vertex_locations = self.params.vertex_locations
        hull_vertices = self.__get_convex_hull_vertices(vertex_locations)

        # Save the convex hull.
        self.hull_vertices = hull_vertices

        # This class represents two dimensional problems.
        self.n_contacts = hull_vertices.shape[0]
        self.n_config = 3
        self.n_friction = 2
        self.n_dims = 2

        # Analyze and store properties about the geometry.
        self.__analyze_and_store_geometry()

        # Set up a Jacobian function for later calculation of contact Jacobians.
        self.jac_func = self.__set_up_contact_jacobian_function()

    def __get_convex_hull_vertices(self, vertex_locations):
        """Compute the convex hull of the provided vertices and return the pared
        down numpy array of vertices in clockwise order."""

        # Get the convex hull of the vertex locations.
        convex_hull = ConvexHull(vertex_locations)

        # Return the list in clockwise order.
        return vertex_locations[np.flip(convex_hull.vertices)]

    def get_vertex_locations_world(self, state, for_visualization=False):
        """Get the locations of the polytope's vertices in world coordinates,
        given the system's current state.  Returns a numpy array of size
        (*, 2) for the (x,y) position of each vertex.  The * is either
        n_contacts if for_visualization is false, or it is equal to the total
        number of vertices in the potentially nonconvex polytope if
        for_visualization is true."""

        # State is in form [x, dx, y, dy, th, dth].
        x, y, theta = state[0], state[2], state[4]

        # If for visualization purposes, want to include all the vertices.
        # Otherwise, just include the convex hull vertices.
        vertices = self.params.vertex_locations if for_visualization else \
                   self.hull_vertices

        p = vertices.shape[0]

        corners_world = np.zeros((p, 2))

        for i in range(p):
            corner_body = vertices[i, :]

            phi = np.arctan2(corner_body[1], corner_body[0])
            radius = np.sqrt(corner_body[1]**2 + corner_body[0]**2)

            corners_world[i, :] = np.array([x + radius * np.cos(phi + theta),
                                            y + radius * np.sin(phi + theta)])

        return corners_world

    def get_vertex_radii_angles(self, for_visualization=False):
        """It may be more convenient in some cases to represent each vertex as a
        radius and angle instead of as a body-frame x and y offset.  Return two
        (*,) numpy arrays of radii and angles.  The * is either n_contacts if
        for_visualization is false, or it is equal to the total number of
        vertices in the potentially nonconvex polytope if for_visualization is
        true."""

        # If for visualization purposes, want to include all the vertices.
        # Otherwise, just include the convex hull vertices.
        vertices = self.params.vertex_locations if for_visualization else \
                   self.hull_vertices

        p = vertices.shape[0]

        radii, angles = np.zeros((p,)), np.zeros((p,))

        for i in range(p):
            px = vertices[i, 0]
            py = vertices[i, 1]

            radii[i] = np.sqrt(px**2 + py**2)
            angles[i] = np.arctan2(py, px)

        return radii, angles

    def __get_vertex_velocities_world(self, state):
        """Get the velocities of the polytope's vertices in world coordinates,
        given the system's current state.  Returns a numpy array of size
        (n_contacts, 2) for the (vx,vy) velocity of each vertex."""

        # State is in form [x, dx, y, dy, th, dth].
        x, y, theta = state[0], state[2], state[4]
        vx, vy, vth = state[1], state[3], state[5]

        p = self.n_contacts
        radii, angles = self.get_vertex_radii_angles()
        corner_velocities = np.zeros((p, 2))

        for i in range(p):
            corner_body = self.hull_vertices[i, :]

            radius, phi = radii[i], angles[i]

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
            px_body, py_body = self.hull_vertices[vertex_i, :]

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

    def __analyze_and_store_geometry(self):
        """This method analyzes the geometry of the polytope's convex hull.
        Specifically, this method calculates and stores the vectors alphas,
        betas, gammas, sides, and psis, as well as the quantities lowest_vertex
        and delta.  For an example to visualize these quantities, see the
        comment in the header of this class definition."""

        # Construct empty vectors for alpha, beta, gamma, side, and psi values.
        p = self.n_contacts

        alphas, betas, gammas = np.zeros((p,)), np.zeros((p,)), np.zeros((p,))
        sides, psis = np.zeros((p,)), np.zeros((p,))

        # Start with the set of convex hull vertices in radius angle form.
        radii, angles = self.get_vertex_radii_angles()

        # Iterate over each triangle formed by the origin and two adjacent
        # vertices.
        for i in range(p):
            # Grab the radii and angles of the current triangle.
            r1, r2 = radii[i], radii[(i+1)%p]
            a1, a2 = angles[i], angles[(i+1)%p]

            # The angle defined with the origin at the middle is the difference
            # between the angles for each vertex.
            a12 = (a1 - a2) % np.pi

            # Use the law of cosines to determine the exterior side length, then
            # the two unknown interior angles.
            r12 = np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(a12))
            b12 = np.arccos(r1/r12 - r2/r12 * np.cos(a12))
            c12 = np.arccos(r2/r12 - r1/r12 * np.cos(a12))

            # Set the correct indices in the storage arrays.
            sides[i], alphas[i], betas[i], gammas[i] = r12, a12, b12, c12

        # Calculate psis after the rest of the vectors are filled in.
        for i in range(p):
            # Grab the relevant beta and gamma value.
            b23 = betas[(i+1)%p]
            c12 = gammas[i]

            psis[(i+1)%p] = np.pi - b23 - c12

        # Find the vertex that would be in contact at theta=0.  If multiple (the
        # most it could be is 2), choose the one that is most clockwise.
        y_min = np.min(self.hull_vertices[:, 1])
        lowests = np.where(self.hull_vertices[:, 1] == y_min)[0]
        lowest_vertex = lowests[0] if lowests[-1] == p else lowests[-1]

        # Calculate the angle delta from the ground to the face counter
        # clockwise from the lowest vertex at theta = 0.
        dist = self.hull_vertices[(lowest_vertex - 1) % p] - \
               self.hull_vertices[lowest_vertex]
        delta = np.arctan2(dist[1], dist[0])

        # Store all of the calculated geometry.
        self.alphas = alphas
        self.betas = betas
        self.gammas = gammas
        self.sides = sides
        self.psis = psis
        self.lowest_vertex = lowest_vertex
        self.delta = delta

    def get_theta_v_and_pivot_index_from_theta(self, theta):
        """Given a current theta value, return theta_v as the angle between the
        ground and the face counter-clockwise to the lowest vertex, as well as
        the index of that lowest vertex."""

        # For convenience, get the number of contacts, psi vector, delta value,
        # and the lowest vertex corresponding to theta = 0.
        p = self.n_contacts
        psis = self.psis
        delta = self.delta
        lowest_vertex = self.lowest_vertex

        # Calculate theta_v by incrementally updating the lowest vertex.
        remainder = theta + delta
        pivot_index = lowest_vertex
        room_to_go = psis[pivot_index] - remainder

        # If room to go is negative, then the next clockwise vertex should have
        # already contacted the ground.
        while room_to_go <= 0:
            remainder -= psis[pivot_index]
            pivot_index = (pivot_index + 1) % p
            room_to_go = psis[pivot_index] - remainder

        theta_v = remainder

        # Return theta_v and the index of the pivoting vertex.
        return theta_v, pivot_index

