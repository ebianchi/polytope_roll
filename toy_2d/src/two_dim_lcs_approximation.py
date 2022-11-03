"""This file calculates a Linear Complementarity System (LCS) approximation of
a two dimensional system.  The linearization approach taken here is from
Aydinoglu et al., 2021.
"""

import numpy as np
import pdb

from toy_2d.src.two_dim_system import TwoDimensionalSystem


class TwoDSystemLCSApproximation:
    """A Linear Complementarity System (LCS) approximation for a two dimensional
    system consisting of a flat ground and a 2D polytope.  This object is
    capable of simulation approximation about preselected (x^*, u^*) points.

    From the nonlinear notation:

            x_{k+1} = f_1(x_k, u_k) + f_2(x_k) lambda_k
                y_k = f_3(x_k, u_k) + f_4(x_k) lambda_k
            0 <= lambda_k  PERP  y_k => 0

    ...we wish to find an LCS approximation of the form:

            x_{k+1} ~= A x_k + B u_k + C lambda_k + d
                y_k ~= G x_k + H u_k + J lambda_k + l
            0 <= lambda_k  PERP  y_k => 0

    NOTE:  The state history will contain the state in order of [vx, vy, vth, x,
    y, th], which is different from the order of the state vector in
    TwoDimensionalSystem objects.  This different order is more convenient for
    the linearization terms, though it unfortunately differs from how the system
    code currently is written.  This object has externally accessible methods to
    convert between these representations.  This isn't perfect, so be careful.

    Properties:
        system:                 TwoDimensionalSystem
        linearization_point:    A dictionary with keys 'q', 'v', 'u_force', and
                                'u_loc' for the state and control linearization
                                points.
        state_history:          (T, 6) numpy array of the linearized approximate
                                state trajectory of the system.
        control_history:        (T-1, 4) numpy array of the control input
                                history of the system, stored as [fx, fy, locx,
                                locy].  This form can be converted into
                                generalized coordinates using the method in
                                self.system.convert_input_to_generalized_coords.
        force_history_n:        (T-1, p_contacts) array of the normal force
                                history of the system, equivalent to Stewart and
                                Trinkle Cn.
        force_history_t:        (T-1, k_friction*p_contacts) array of the
                                tangential force history of the system,
                                equivalent to Stewart and Trinkle Beta.
        output_history:         (T-1, k_friction*p_contacts) array of the y_k
                                complementarity vectors corresponding to the
                                simulation history.
    """
    system: TwoDimensionalSystem

    def __init__(self, system: TwoDimensionalSystem):
        self.system = system

        # Get the number of contacts and friction cone directions from the
        # system's polytope parameters.
        p = self.system.params.polytope.n_contacts
        k = self.system.params.polytope.n_friction

        # Initialize histories to be empty.
        self.state_history = np.zeros((0, 6))
        self.control_history = np.zeros((0, 4))
        self.force_history_n = np.zeros((0, p))
        self.force_history_t = np.zeros((0, p*k))
        self.output_history = np.zeros((0, p*(k+1)))

        # Initialize the linearization point to be empty.
        self.linearization_point = {'q': None, 'v': None,
                                    'u_force': None, 'u_loc': None}

    def set_linearization_point(self, q, v, u_force, u_loc):
        """Set the linear approximation's point of linearization.  Note that
        this method will not will not do anything with the stored histories."""

        q, v = q.squeeze(), v.squeeze()
        u_force, u_loc = u_force.squeeze(), u_loc.squeeze()

        assert q.shape == (3,) and v.shape == (3,)
        assert u_force.shape == (2,) and u_loc.shape == (2,)
        
        # Store the linearization point.
        self.linearization_point['q'] = q
        self.linearization_point['v'] = v
        self.linearization_point['u_force'] = u_force
        self.linearization_point['u_loc'] = u_loc

    def _get_f_1(self, x, u):
        """From the nonlinear notation, evaluate the value of f_1 (of shape
        (6, 1)) at the provided x and u. Note that the x is expected to be in
        the LCS order [vx, vy, vth, x, y, th] and the control input is expected
        to be in generalized coordinates [fx, fy, tau]."""
        
        # Get some necessary values from inside the system.
        dt = self.system.params.dt
        polytope = self.system.params.polytope
        n = polytope.n_config

        # Will need the state vector in system form.
        state_sys = self._convert_lcs_state_to_system_state(x)

        # Need the mass matrix and continuous forces.
        M = polytope.get_M_matrix(state_sys)
        M_inv = np.linalg.inv(M)
        k = polytope.get_k_vector(state_sys).reshape(3,)

        # Build the expression.
        mat_1 = np.vstack((np.hstack((np.eye(n), np.zeros((n,n)))),
                           np.hstack((dt*np.eye(n), np.eye(n)))))
        mat_2 = np.vstack((np.eye(n), dt*np.eye(n)))

        return mat_1 @ x + dt * mat_2 @ M_inv @ (k+u)

    def _get_f_2(self, x):
        """From the nonlinear notation, evaluate the value of f_2 (of shape
        (6, p*(k+1))) at the provided. Note that the x is expected to be in the
        LCS order [vx, vy, vth, x, y, th]."""
        
        # Get some necessary values from inside the system.
        dt = self.system.params.dt
        polytope = self.system.params.polytope
        n = polytope.n_config
        p = polytope.n_contacts
        k_friction = polytope.n_friction

        # Will need the state vector in system form.
        state_sys = self._convert_lcs_state_to_system_state(x)

        # Need the mass matrix and contact jacobians.
        M = polytope.get_M_matrix(state_sys)
        M_inv = np.linalg.inv(M)
        D = polytope.get_D_matrix(state_sys)
        N = polytope.get_N_matrix(state_sys)

        # Build the expression.
        mat_1 = np.vstack((np.eye(n), dt*np.eye(n)))
        mat_2 = np.hstack((D, N))

        return mat_1 @ M_inv @ mat_2

    def _get_f_3(self, x, u):
        """From the nonlinear notation, evaluate the value of f_3 (of shape
        (p*(k+1), 1)) at the provided x and u.  Note that the x is expected to
        be in the LCS order [vx, vy, vth, x, y, th] and the control input is
        expected to be in generalized coordinates [fx, fy, tau]."""
        
        # Get some necessary values from inside the system.
        dt = self.system.params.dt
        polytope = self.system.params.polytope
        n = polytope.n_config
        p = polytope.n_contacts
        k_friction = polytope.n_friction

        # Need some information about the polytope vertices.
        radii, angles = polytope.get_vertex_radii_angles()

        # Will need the state vector in system form.
        state_sys = self._convert_lcs_state_to_system_state(x)
        y, theta = state_sys[2], state_sys[4]

        # Need the mass matrix, contact jacobians, and continuous forces.
        M = polytope.get_M_matrix(state_sys)
        M_inv = np.linalg.inv(M)
        D = polytope.get_D_matrix(state_sys)
        N = polytope.get_N_matrix(state_sys)
        k = polytope.get_k_vector(state_sys).reshape(3,)

        # Build the expression.
        P_2 = np.hstack((np.zeros((p,1)), np.ones((p,1)), np.zeros((p,n+1))))
        P_3 = np.hstack((np.zeros((p,2)), np.ones((p,1)), np.zeros((p,n))))

        mat_1 = np.vstack((D.T, N.T))
        mat_2 = np.hstack((np.zeros((n,n)), np.eye(n)))
        mat_3 = np.vstack((np.zeros((p*k_friction, 2*n)), P_2))
        mat_4 = np.vstack((np.zeros((p*k_friction, p)), np.eye(p)))

        diag_r = np.diag(radii)

        sin_vec = np.sin(P_3 @ x + angles)
        
        return mat_1 @ (mat_2@x + dt*M_inv@(k+u)) + (1/dt) * mat_3@x + \
               (1/dt) * mat_4@diag_r@sin_vec

    def _get_f_4(self, _x):
        """From the nonlinear notation, evaluate the value of f_4 (of shape
        (p*(k+1), p*(k+1))) at the provided x.  Note that the x is expected to
        be in the LCS order [vx, vy, vth, x, y, th]."""
        
        # It turns out that f_4(x_k) is all zeros.  Return all zeros in the
        # right shape.
        p = self.system.params.polytope.n_contacts
        k_friction = self.system.params.polytope.n_friction
        return np.zeros((p*(k_friction+1), p*(k_friction+1)))

    def _get_df1_dx(self, _x, _u):
        """From the nonlinear notation, evaluate the partial derivative of f1
        with respect to the state at the provided state and control input,
        yielding a jacobian of shape (6,6).  Again, the state is expected to be
        in the LCS order [vx, vy, vth, x, y, th]."""
        
        # Get some necessary values from inside the system.
        dt = self.system.params.dt
        polytope = self.system.params.polytope
        n = polytope.n_config

        # Build the expression.
        return np.vstack((np.hstack((np.eye(n), np.zeros((n,n)))),
                          np.hstack((dt*np.eye(n), np.eye(n)))))

    def _get_df1_du(self, x, _u):
        """From the nonlinear notation, evaluate the partial derivative of f1
        with respect to control input at the provided state and control input,
        yielding a jacobian of shape (6,3).  Again, the state is expected to be
        in the LCS order [vx, vy, vth, x, y, th]."""
        
        # Get some necessary values from inside the system.
        dt = self.system.params.dt
        polytope = self.system.params.polytope
        n = polytope.n_config

        # Will need the state vector in system form.
        state_sys = self._convert_lcs_state_to_system_state(x)

        # Need the mass matrix.
        M = polytope.get_M_matrix(state_sys)
        M_inv = np.linalg.inv(M)
        
        # Build the expression.
        mat_1 = np.vstack((np.eye(n), dt*np.eye(n)))

        return dt * mat_1 @ M_inv

    def _get_df3_dx(self, x, u):
        """From the nonlinear notation, evaluate the partial derivative of f3
        with respect to the state at the provided state and control input,
        yielding a jacobian of shape (p*(k+1),6).  Again, the state is expected
        to be in the LCS order [vx, vy, vth, x, y, th]."""
        
        # Get some necessary values from inside the system.
        dt = self.system.params.dt
        polytope = self.system.params.polytope
        n = polytope.n_config
        p = polytope.n_contacts
        k_friction = polytope.n_friction

        # Need some information about the polytope vertices.
        radii, angles = polytope.get_vertex_radii_angles()

        # Will need the state vector in system form.
        state_sys = self._convert_lcs_state_to_system_state(x)

        # Need the contact jacobians.
        D = polytope.get_D_matrix(state_sys)
        N = polytope.get_N_matrix(state_sys)

        # Build the expression.
        P_2 = np.hstack((np.zeros((p,1)), np.ones((p,1)), np.zeros((p,n+1))))
        P_3 = np.hstack((np.zeros((p,2)), np.ones((p,1)), np.zeros((p,n))))

        mat_1 = np.vstack((D.T, N.T))
        mat_2 = np.hstack((np.zeros((n,n)), np.eye(n)))
        mat_3 = np.vstack((np.zeros((p*k_friction, 2*n)), P_2))
        mat_4 = np.vstack((np.zeros((p*k_friction, p)), np.eye(p)))

        diag_r = np.diag(radii)

        cos_vec = np.cos(P_3 @ x + angles)
        diag_cos = np.diag(cos_vec)

        return mat_1 @ mat_2 + (1/dt) * mat_3 + \
               (1/dt) * mat_4 @ diag_r @ diag_cos @ P_3

    def _get_df3_du(self, x, _u):
        """From the nonlinear notation, evaluate the partial derivative of f3
        with respect to control input at the provided state and control input,
        yielding a jacobian of shape (p*(k+1),3).  Again, the state is expected
        to be in the LCS order [vx, vy, vth, x, y, th]."""
        
        # Get some necessary values from inside the system.
        dt = self.system.params.dt
        polytope = self.system.params.polytope

        # Will need the state vector in system form.
        state_sys = self._convert_lcs_state_to_system_state(x)

        # Need the mass matrix and contact jacobians.
        M = polytope.get_M_matrix(state_sys)
        M_inv = np.linalg.inv(M)
        D = polytope.get_D_matrix(state_sys)
        N = polytope.get_N_matrix(state_sys)

        # Build the expression.
        mat_1 = np.vstack((D.T, N.T))

        return dt * mat_1 @ M_inv

    def _convert_lcs_state_to_system_state(self, lcs_state):
        """Helper function to convert the LCS state [vx, vy, vth, x, y, th] to
        system state [x, vx, y, vy, th, vth]."""

        assert lcs_state.shape == (6,)

        vx, vy, vth = lcs_state[0], lcs_state[1], lcs_state[2]
        x, y, th = lcs_state[3], lcs_state[4], lcs_state[5]
        return np.array([x, vx, y, vy, th, vth])

    def _convert_system_state_to_lcs_state(self, system_state):
        """Helper function to convert the system state [x, vx, y, vy, th, vth]
        to LCS state [vx, vy, vth, x, y, th]."""

        assert system_state.shape == (6,)

        x, y, th = system_state[0], system_state[2], system_state[4]
        vx, vy, vth = system_state[1], system_state[3], system_state[5]
        return np.array([vx, vy, vth, x, y, th])

    def get_lcs_terms(self):
        """Make one helper function to calculate all of the LCS-related terms of
        the system about the linearization stored in self.linearization_point.
        It is expected that self.set_linearization_point() is called first with
        the appropriate linearization point specified."""

        # Check that the linearization point has been set (assume 'q' key is
        # filled only if all of the other relevant keys are also filled).
        assert self.linearization_point['q'] is not None

        # Get and return all of the matrices and vectors.
        A = self.get_A_matrix()
        B = self.get_B_matrix()
        C = self.get_C_matrix()
        d = self.get_d_vector()
        G = self.get_G_matrix()
        H = self.get_H_matrix()
        J = self.get_J_matrix()
        l = self.get_l_vector()

        return A, B, C, d, G, H, J, l

    def get_A_matrix(self):
        """Calculate the A matrix in the LCS approximation.  The linearization
        is done about the stored self.linearization_point, which must be set
        prior to calling this function."""

        # This requires using the state and control input stored in the
        # linearization.
        v, q = self.linearization_point['v'], self.linearization_point['q']
        x = np.hstack((v, q))

        control_force = self.linearization_point['u_force']
        control_loc = self.linearization_point['u_loc']
        u = self.system.convert_input_to_generalized_coords(x, control_force,
                                                            control_loc)

        # Build the expression via a call to df1_dx.
        return self._get_df1_dx(x, u)

    def get_B_matrix(self):
        """Calculate the B matrix in the LCS approximation.  The linearization
        is done about the stored self.linearization_point, which must be set
        prior to calling this function."""

        # This requires using the state and control input stored in the
        # linearization.
        v, q = self.linearization_point['v'], self.linearization_point['q']
        x = np.hstack((v, q))

        control_force = self.linearization_point['u_force']
        control_loc = self.linearization_point['u_loc']
        u = self.system.convert_input_to_generalized_coords(x, control_force,
                                                            control_loc)

        # Build the expression via a call to df1_du.
        return self._get_df1_du(x, u)

    def get_C_matrix(self):
        """Calculate the C matrix in the LCS approximation.  The linearization
        is done about the stored self.linearization_point, which must be set
        prior to calling this function."""

        # This requires calling the self._get_f_2(x) function at the state
        # stored in the linearization.
        v, q = self.linearization_point['v'], self.linearization_point['q']
        x = np.hstack((v, q))
        return self._get_f_2(x)

    def get_d_vector(self):
        """Calculate the d vector in the LCS approximation.  The linearization
        is done about the stored self.linearization_point, which must be set
        prior to calling this function."""

        # This requires using the state and control input stored in the
        # linearization.
        v, q = self.linearization_point['v'], self.linearization_point['q']
        x = np.hstack((v, q))

        control_force = self.linearization_point['u_force']
        control_loc = self.linearization_point['u_loc']
        u = self.system.convert_input_to_generalized_coords(x, control_force,
                                                            control_loc)

        # Build the expression.
        f1_val = self._get_f_1(x, u)
        df1_dx_val = self._get_df1_dx(x, u)
        df1_du_val = self._get_df1_du(x, u)

        return f1_val - df1_dx_val @ x - df1_du_val @ u

    def get_G_matrix(self):
        """Calculate the G matrix in the LCS approximation.  The linearization
        is done about the stored self.linearization_point, which must be set
        prior to calling this function."""

        # This requires using the state and control input stored in the
        # linearization.
        v, q = self.linearization_point['v'], self.linearization_point['q']
        x = np.hstack((v, q))

        control_force = self.linearization_point['u_force']
        control_loc = self.linearization_point['u_loc']
        u = self.system.convert_input_to_generalized_coords(x, control_force,
                                                            control_loc)

        # Build the expression via a call to df3_dx.
        return self._get_df3_dx(x, u)

    def get_H_matrix(self):
        """Calculate the H matrix in the LCS approximation.  The linearization
        is done about the stored self.linearization_point, which must be set
        prior to calling this function."""

        # This requires using the state and control input stored in the
        # linearization.
        v, q = self.linearization_point['v'], self.linearization_point['q']
        x = np.hstack((v, q))

        control_force = self.linearization_point['u_force']
        control_loc = self.linearization_point['u_loc']
        u = self.system.convert_input_to_generalized_coords(x, control_force,
                                                            control_loc)

        # Build the expression via a call to df3_du.
        return self._get_df3_du(x, u)

    def get_J_matrix(self):
        """Calculate the J matrix in the LCS approximation.  The linearization
        is done about the stored self.linearization_point, which must be set
        prior to calling this function."""

        # This requires calling the self._get_f_4(x) function at the state
        # stored in the linearization.
        v, q = self.linearization_point['v'], self.linearization_point['q']
        x = np.hstack((v, q))
        return self._get_f_4(x)

    def get_l_vector(self):
        """Calculate the l vector in the LCS approximation.  The linearization
        is done about the stored self.linearization_point, which must be set
        prior to calling this function."""

        # This requires using the state and control input stored in the
        # linearization.
        v, q = self.linearization_point['v'], self.linearization_point['q']
        x = np.hstack((v, q))

        control_force = self.linearization_point['u_force']
        control_loc = self.linearization_point['u_loc']
        u = self.system.convert_input_to_generalized_coords(x, control_force,
                                                            control_loc)

        # Build the expression.
        f3_val = self._get_f_3(x, u)
        df3_dx_val = self._get_df3_dx(x, u)
        df3_du_val = self._get_df3_du(x, u)

        return f3_val - df3_dx_val @ x - df3_du_val @ u

    def __check_consistent_histories(self):
        """Check that the history lengths are compatible."""
        state_len = self.state_history.shape[0]
        control_len = self.control_history.shape[0]
        fn_len = self.force_history_n.shape[0]
        ft_len = self.force_history_t.shape[0]
        output_len = self.output_history.shape[0]

        if state_len == 0:
            assert control_len == fn_len == ft_len == 0

        else:
            assert state_len - control_len == 1
            assert control_len == fn_len == ft_len == output_len

    def set_initial_state(self, state):
        """Set the initial state of the system.  This method will automatically
        clear out the state, control, contact force, and output histories of the
        system and set the initial state to that provided as an argument."""

        # Get the number of contacts and friction cone directions from the
        # system's polytope parameters.
        p = self.system.params.polytope.n_contacts
        k = self.system.params.polytope.n_friction
        
        # Clear out the histories, setting the first state_history entry to the
        # provided state.
        self.state_history = state.reshape(1, 6)
        self.control_history = np.zeros((0, 4))
        self.force_history_n = np.zeros((0, p))
        self.force_history_t = np.zeros((0, p*k))
        self.output_history = np.zeros((0, p*(k+1)))

    def step_lcs_dynamics(self, control_force, control_loc, cn, beta):
        """Given new control inputs and contact forces, step the system forward
        in time, appending the next state, provided controls, and provided
        contact forces to the end of the state and control history arrays,
        respectively."""

        # Check that the histories are consistent.
        self.__check_consistent_histories()

        # Get the current state and step the dynamics.
        state = self.state_history[-1, :]
        next_state, yk = self.__step_lcs_dynamics(state, control_force,
                                                  control_loc, cn, beta)

        # Set the state and control histories.
        control_entry = np.hstack((control_force, control_loc))
        self.state_history = np.vstack((self.state_history, next_state))
        self.control_history = np.vstack((self.control_history, control_entry))
        self.force_history_n = np.vstack((self.force_history_n, cn))
        self.force_history_t = np.vstack((self.force_history_t, beta))
        self.output_history = np.vstack((self.output_history, yk))

    def __step_lcs_dynamics(self, state, control_force, control_loc, cn, beta):
        """Given the current state, control_force (given as (2,) array for one
        control force), control_loc (given as (2,) array for one location),
        normal contact forces cn (given as (p,) array), and tangential contact
        forces beta (given as (p*k,) array), simulate the LCS approximation one
        timestep into the future.  This function uses the linearization point
        that is assumed to be already set via self.set_linearization_point().
        This function does NOT check that the provided control_force and
        control_loc are valid (i.e. within friction cone and acting on the
        surface of the object).  Returns the next state and the y_k vector."""

        # Get all the LCS terms.
        A, B, C, d, G, H, J, l = self.get_lcs_terms()
        x = state

        # Convert the control force and location to generalized coordinates.
        state_sys = self._convert_lcs_state_to_system_state(state)
        u = self.system.convert_input_to_generalized_coords(state_sys,
                                                    control_force, control_loc)
        u = u.squeeze()

        # Convert the normal and tangential forces into one lambda vector.
        lambda_k = np.hstack((beta.squeeze(), cn.squeeze()))

        # Evaluate the expressions for x_{k+1} and y_k.
        x_k1 = A@x + B@u + C@lambda_k + d
        y_k  = G@x + H@u + J@lambda_k + l

        return x_k1, y_k

