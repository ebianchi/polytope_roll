"""This file describes a system for simulating a single two dimensional polytope
interacting with gravity, control inputs, and a flat table with which the
polytope undergoes inelastic frictional contact.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pdb
import sympy

from toy_2d.src import solver_utils
from toy_2d.src.two_dim_polytope import TwoDimensionalPolytope


@dataclass
class TwoDimensionalSystemParams:
    dt: float = 0.001
    polytope: TwoDimensionalPolytope = None
    mu_control: float = 0.5

class TwoDimensionalSystem:
    """A system capable of simulation consisting of a flat ground and a 2D
    polytope, with the ability to set control inputs.

    Properties:
        params:             2D system parameters, including a time step, a 2D
                            polytope object, and the friction parameter between
                            a control contact and the polytope.
        state_history:      (T, 6) numpy array of the state trajectory of the
                            system.
        control_history:    (T-1, 4) numpy array of the control input history of
                            the system, stored as [fx, fy, locx, locy].  This
                            form is more convenient for plotting.  From this
                            information, it is simple to get the controls in
                            generalized coordinates using the system's
                            convert_input_to_generalized_coords method.
        lambda_history:     (T-1, p_contacts * (k_friction+2)) array of the full
                            lambda vector in the LCP M*lambda + q >= 0.  This
                            will help us compare these "real" dynamics with
                            linearization approximations later.  Note that the
                            array[:, :p_contacts*k_friction] gives the history
                            of tangential forces, Beta from Stewart and Trinkle,
                            and the array[:, p*k:p*(k+1)] gives the history of
                            normal forces, Cn from Stewart and Trinkle.
        output_history:     (T-1, p_contacts * (k_friction+2)) array of the full
                            z vector in the LCP M*lambda + q = z.  This will
                            help us compare these "real" dynamics with
                            linearization approximations later.
    """
    params: TwoDimensionalSystemParams

    def __init__(self, params: TwoDimensionalSystemParams):
        self.params = params

        # Get the number of contacts and friction cone directions from the
        # system's polytope parameters.
        p = self.params.polytope.n_contacts
        k = self.params.polytope.n_friction

        # Initialize histories to be empty.
        self.state_history = np.zeros((0, 6))
        self.control_history = np.zeros((0, 4))
        self.lambda_history = np.zeros((0, p*(k+2)))
        self.output_history = np.zeros((0, p*(k+2)))

    def __check_consistent_histories(self):
        """Check that the history lengths are compatible."""
        state_len = self.state_history.shape[0]
        control_len = self.control_history.shape[0]
        lam_len = self.lambda_history.shape[0]
        out_len = self.output_history.shape[0]

        if state_len == 0:
            assert control_len == lam_len == out_len == 0

        else:
            assert state_len - control_len == 1
            assert control_len == lam_len == out_len

    def __step_dynamics_no_ground_contact(self, state, controls):
        """Do a dynamics step assuming no contact forces."""

        polytope = self.params.polytope
        dt = self.params.dt

        v0 = np.array([state[1], state[3], state[5]]).reshape(3, 1)
        q0 = np.array([state[0], state[2], state[4]]).reshape(3, 1)

        # Some of these quantities are more accurate if calculated at a point in
        # the future, so create a mid-state.
        q_mid_for_M = q0 + dt*v0
        q_mid_for_k = q0 + dt*v0/2

        state_for_M = np.array([q_mid_for_M[0], v0[0],
                                q_mid_for_M[1], v0[1],
                                q_mid_for_M[2], v0[2]])
        state_for_k = np.array([q_mid_for_k[0], v0[0],
                                q_mid_for_k[1], v0[1],
                                q_mid_for_k[2], v0[2]])

        # Do a forward simulation as if forces were zero (no LCP required).
        M = polytope.get_M_matrix(state_for_M)
        k = polytope.get_k_vector(state_for_k)
        u = self.convert_input_to_generalized_coords(state, controls)
        u = u.reshape((3, 1))

        v_next = v0 + np.linalg.inv(M) @ (k + u) * dt
        q_next = q0 + dt * v_next

        v_next = v_next.reshape((3,))
        q_next = q_next.reshape((3,))

        next_state = np.array([q_next[0], v_next[0], 
                               q_next[1], v_next[1],
                               q_next[2], v_next[2]])
        return next_state

    def get_simulation_terms(self, state, controls):
        """Make one helper function to calculate all of the simulation-related
        terms of a system at the state."""

        polytope = self.params.polytope
        dt = self.params.dt

        q0 = np.array([state[0], state[2], state[4]]).reshape(3, 1)
        v0 = np.array([state[1], state[3], state[5]]).reshape(3, 1)

        # Some of these quantities are more accurate if calculated at a point in
        # the future, so create a mid-state.  These midstates are defined for
        # the mass matrix, M, and the continuous force vector, k, in the top
        # Equation 27 of Stewart and Trinkle 1996.
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

        u = self.convert_input_to_generalized_coords(state, controls)
        u = u.reshape((3, 1))

        D = polytope.get_D_matrix(state)
        N = polytope.get_N_matrix(state)
        E = polytope.get_E_matrix(state)
        Mu = polytope.get_mu_matrix(state)
        phi = polytope.get_phi(state)

        return M, D, N, E, Mu, k, u, q0, v0, phi

    def set_initial_state(self, state):
        """Set the initial state of the system.  This method will automatically
        clear out the state, control, and contact force histories of the system
        and set the initial state to that provided as an argument."""

        # Get the number of contacts and friction cone directions from the
        # system's polytope parameters.
        p = self.params.polytope.n_contacts
        k = self.params.polytope.n_friction
        
        # Clear out the histories, setting the first state_history entry to the
        # provided state and emptying the other histories.
        self.state_history = state.reshape(1, 6)
        self.control_history = np.zeros((0, 4))
        self.lambda_history = np.zeros((0, p*(k+2)))
        self.output_history = np.zeros((0, p*(k+2)))

    def step_dynamics(self, controls):
        """Given new control inputs, step the system forward in time, appending
        the next state and provided controls to the end of the state and control
        history arrays, respectively."""

        # Check that the histories are consistent.
        self.__check_consistent_histories()

        # Get the current state and step the dynamics.
        state = self.state_history[-1, :]
        next_state, lam, out = self.__step_dynamics(state, controls)

        # Get the controls in the full [fx, fy, loc_x, loc_y] format.
        full_controls = self.get_full_controls(state, controls)

        # Set the state and control histories.
        self.state_history = np.vstack((self.state_history, next_state))
        self.control_history = np.vstack((self.control_history, full_controls))
        self.lambda_history = np.vstack((self.lambda_history, lam))
        self.output_history = np.vstack((self.output_history, out))

    def __step_dynamics(self, state, controls):
        """Given the current state and controls (given as (4,) array for
        [force_x, force_y, world_loc_x, world_loc_y]), simulate the system one
        timestep into the future.  This function necessarily determines the
        ground reaction forces.  This function does NOT check that the provided
        control_force and control_loc are valid (i.e. within friction cone and
        acting on the surface of the object).  Returns the next state,
        complementarity lambda vector, and the LCP output vector."""

        polytope = self.params.polytope
        dt = self.params.dt
        p, k_friction = polytope.n_contacts, polytope.n_friction

        # Construct LCP terms.
        M, D, N, E, Mu, k, u, q0, v0, phi = self.get_simulation_terms(state,
                                                    controls)
        M_i = np.linalg.inv(M)

        # Formulating this inelastic frictional contact dynamics as an LCP is
        # given in Stewart and Trinkle 1996 (see Equation 29 for the structure
        # of the lcp_mat and lcp_vec).
        mat_top = np.hstack((D.T @ M_i @ D, D.T @ M_i @ N, E))
        mat_mid = np.hstack((N.T @ M_i @ D, N.T @ M_i @ N, np.zeros((p,p))))
        mat_bot = np.hstack((-E.T, Mu, np.zeros((p,p))))

        vec_top = D.T @ (v0 + dt * M_i @ (k+u))
        vec_mid = (1./dt) * phi + N.T @ (v0 + dt * M_i @ (k+u))
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

        # Return the next state, lambda, and outputs.
        lam = lcp_sol.squeeze()
        output = lcp_mat@lam + lcp_vec.squeeze()

        return next_state, lam, output

    def convert_input_to_generalized_coords(self, state, controls):
        """Convert an input force and location to forces in the generalized
        coordinates of the system, given its state.  This function itself makes
        no assumptions about the form of the controls, which is interpreted in
        self.get_map_from_controls_to_gen_coordinates, leaving flexibility in
        how the controls are provided."""

        # Use the map from controls to generalized coordinates.
        P = self.get_map_from_controls_to_gen_coordinates(state, controls)
        return P @ controls

    def get_map_from_controls_to_gen_coordinates(self, state, controls):
        """Return the map P that converts the control inputs into generalized
        coordinates, i.e. u_gen = P @ tilde{u}.  In this case, we assume the
        controls are in the form [fx, fy, loc_x, loc_y], and we linearize about
        the current state and control location."""

        controls = controls.squeeze()
        assert controls.shape == (4,)

        # Grab the current state and control location for linearization.
        x, y = state[0], state[2]
        u_locx, u_locy = controls[2], controls[3]

        # Build the expressions for force in the x direction, force in the y
        # direction, and torque about the CoM.
        lever_x = u_locx - x
        lever_y = u_locy - y

        return np.array([[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [-lever_y, lever_x, 0., 0.]])

    def get_full_controls(self, _state, controls):
        """Given the controls in whatever format, convert it into the format
        [fx, fy, loc_x, loc_y], which is most convenient for visualization
        later.  In this TwoDimensionalSystem, we assume controls is already in
        that format."""
        
        # Return controls itself.
        return controls.reshape(1, 4)


class TwoDSystemMagOnly(TwoDimensionalSystem):
    """An extension of the TwoDimensionalSystem that assumes the body-frame
    control contact location and direction is fixed.  This amendment only
    requires adapting self.get_map_from_controls_to_gen_coordinates() and
    self.get_full_controls(), which are assisted by keeping track of the body-
    frame location and direction in new properties outlined below.

    Properties:
        contact_point:  Location (given as (2,) x,y numpy array) in body-frame
                        of the control contact force.
        contact_angle:  Direction (given as an angle) in body-frame of the
                        control contact force.
    """
    params: TwoDimensionalSystemParams
    contact_point: np.array
    contact_angle: float

    def __init__(self, params: TwoDimensionalSystemParams,
                 contact_point: np.array, contact_angle: float):
        # Do some checks that the contact point and contact direction make
        # sense.
        contact_point = contact_point.squeeze()
        assert contact_point.shape == (2,)
        assert type(contact_angle) == float

        # Initialize the underlying TwoDimensionalSystem.
        super().__init__(params)

        # Save the contact point and contact angle.
        self.contact_point = contact_point
        self.contact_angle = contact_angle

    def get_map_from_controls_to_gen_coordinates(self, state, _controls):
        """Given controls as a (1,) vector corresponding to a magnitude of
        force, find the (3,1) linear map that will convert it into a generalized
        force vector corresponding to [fx, fy, torque]."""

        # First, the x and y components of the force just depend on the angle
        # of the vector.
        theta = state[4]
        body_angle = self.contact_angle
        world_angle = body_angle + theta

        # The torque additionally depends on the body-frame contact location.
        lever_x, lever_y = self.contact_point[0], self.contact_point[1]

        # Build the expression.
        return np.array([[np.cos(world_angle)],
                         [np.sin(world_angle)],
                         [lever_x * np.sin(body_angle) - \
                          lever_y * np.cos(body_angle)]])

    def get_full_controls(self, state, controls):
        """Given controls as a (1,) vector corresponding to a magnitude of
        force, find the (1,4) controls corresponding to [fx, fy, loc_x, loc_y]
        to be stored in the control history."""

        x, y, theta = state[0], state[2], state[4]

        # Get the control input as a single number.
        assert controls.shape == (1,)
        mag = controls[0]

        # First, the x and y components of the force just depend on the angle
        # of the vector.
        angle = self.contact_angle + theta
        fx = mag * np.cos(angle)
        fy = mag * np.sin(angle)

        # The location of the force depends on the location of the system as
        # well as the relative location of the contact location.
        dx_body_frame = self.contact_point[0]
        dy_body_frame = self.contact_point[1]

        dx_world = dx_body_frame*np.cos(theta) - dy_body_frame*np.sin(theta)
        dy_world = dx_body_frame*np.sin(theta) + dy_body_frame*np.cos(theta)

        loc_x = x + dx_world
        loc_y = y + dy_world

        # Return [fx, fy, loc_x, loc_y].
        return np.array([fx, fy, loc_x, loc_y]).reshape(1,4)
