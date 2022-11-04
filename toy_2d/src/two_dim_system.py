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
        force_history_n:    (T-1, p_contacts) array of the normal force history
                            of the system, equivalent to Stewart and Trinkle Cn.
        force_history_t:    (T-1, k_friction*p_contacts) array of the tangential
                            force history of the system, equivalent to Stewart
                            and Trinkle Beta.
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
        self.force_history_n = np.zeros((0, p))
        self.force_history_t = np.zeros((0, p*k))

    def __check_consistent_histories(self):
        """Check that the history lengths are compatible."""
        state_len = self.state_history.shape[0]
        control_len = self.control_history.shape[0]
        fn_len = self.force_history_n.shape[0]
        ft_len = self.force_history_t.shape[0]

        if state_len == 0:
            assert control_len == fn_len == ft_len == 0

        else:
            assert state_len - control_len == 1
            assert control_len == fn_len == ft_len

    def __step_dynamics_no_ground_contact(self, state, control_force,
                                          control_loc):
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
        u = self.convert_input_to_generalized_coords(state, control_force,
                                                     control_loc)
        u = u.reshape((3, 1))

        v_next = v0 + np.linalg.inv(M) @ (k + u) * dt
        q_next = q0 + dt * v_next

        v_next = v_next.reshape((3,))
        q_next = q_next.reshape((3,))

        next_state = np.array([q_next[0], v_next[0], 
                               q_next[1], v_next[1],
                               q_next[2], v_next[2]])
        return next_state

    def get_simulation_terms(self, state, control_force, control_loc):
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

        u = self.convert_input_to_generalized_coords(state, control_force,
                                                     control_loc)
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
        # provided state.
        self.state_history = state.reshape(1, 6)
        self.control_history = np.zeros((0, 4))
        self.force_history_n = np.zeros((0, p))
        self.force_history_t = np.zeros((0, p*k))

    def step_dynamics(self, control_force, control_loc):
        """Given new control inputs, step the system forward in time, appending
        the next state and provided controls to the end of the state and control
        history arrays, respectively."""

        # Check that the histories are consistent.
        self.__check_consistent_histories()

        # Get the current state and step the dynamics.
        state = self.state_history[-1, :]
        next_state, fn, ft = self.__step_dynamics(state, control_force,
                                                  control_loc)

        # Set the state and control histories.
        control_entry = np.hstack((control_force, control_loc))
        self.state_history = np.vstack((self.state_history, next_state))
        self.control_history = np.vstack((self.control_history, control_entry))
        self.force_history_n = np.vstack((self.force_history_n, fn))
        self.force_history_t = np.vstack((self.force_history_t, ft))

    def __step_dynamics(self, state, control_force, control_loc):
        """Given the current state, control_force (given as (2,) array for one
        control force), and control_loc (given as (2,) array for one location),
        simulate the system one timestep into the future.  This function
        necessarily determines the ground reaction forces.  This function does
        NOT check that the provided control_force and control_loc are valid
        (i.e. within friction cone and acting on the surface of the object).
        Returns the next state and normal/tangential force vectors."""

        polytope = self.params.polytope
        dt = self.params.dt
        p, k_friction = polytope.n_contacts, polytope.n_friction

        # First, pretend no contact with the ground occurred.
        next_state = self.__step_dynamics_no_ground_contact(state,
                                                            control_force,
                                                            control_loc)
        # No contact occurred if the simulation features no vertices under the
        # ground.
        if min(polytope.get_phi(next_state)) >= 0:
            fn = np.zeros((p,))
            ft = np.zeros((p*k_friction,))
            return next_state, fn, ft

        # At this point, we've established that contact forces are necessary.
        # Need to solve LCP to get proper contact forces -- construct terms.
        M, D, N, E, Mu, k, u, q0, v0, phi = self.get_simulation_terms(state,
                                                    control_force, control_loc)
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

        # Save the contact forces so we can return them with the next state.
        fn = Cn.squeeze()
        ft = Beta.squeeze()

        return next_state, fn, ft

    def get_linearized_discrete_dynamics(self, state, control_force, control_location):
        """Given current state, control, get linearised dynamics matrix A,B"""
        pass

    def convert_input_to_generalized_coords(self, state, control_force,
                                              control_loc):
        """Convert an input force and location to forces in the generalized
        coordinates of the system, given its state."""
        x, y = state[0], state[2]
        fx, fy = control_force[0], control_force[1]
        u_locx, u_locy = control_loc[0], control_loc[1]

        lever_x = u_locx - x
        lever_y = u_locy - y

        torque = lever_x*fy - lever_y*fx

        return np.array([fx, fy, torque])
