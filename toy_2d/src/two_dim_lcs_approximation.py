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
    capable of simulation approximation about preselected (x^*, u^*) state and
    generalized coordinate input points.

    From the nonlinear notation:

            x_{k+1} = f_1(x_k, u_k) + f_2(x_k) lambda_k
                y_k = f_3(x_k, u_k) + f_4(x_k) lambda_k
            0 <= lambda_k  PERP  y_k => 0

    ...we wish to find an LCS approximation of the form:

            x_{k+1} ~= A x_k + B u_k + C lambda_k + d
                y_k ~= G x_k + H u_k + J lambda_k + l
            0 <= lambda_k  PERP  y_k => 0

    Some systems may take control inputs as something other than the generalized
    coordinates.  Thus we define the generalized coordinates u_k via a linear
    map from the provided coordinates tilde{u}_k as:

            u_k = P tilde{u}_k

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
                                self.convert_input_to_generalized_coords.
        lambda_history:         (T-1, p_contacts * (k_friction+2)) array of the
                                full lambda vector in the LCP M*lambda + q >= 0.
                                This will help us compare these "real" dynamics
                                with linearization approximations later.  Note
                                that the array[:, :p_contacts*k_friction] gives
                                the history of tangential forces, Beta from
                                Stewart and Trinkle, and array[:, p*k:p*(k+1)]
                                gives the history of normal forces, Cn from
                                Stewart and Trinkle.
        output_history:         (T-1, p_contacts * (k_friction+2)) array of the
                                full z vector in the LCP M*lambda + q = z.  This
                                will help us compare these "real" dynamics with
                                linearization approximations later.
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
        self.full_control_history = np.zeros((0, 4))
        self.lambda_history = np.zeros((0, p*(k+2)))
        self.output_history = np.zeros((0, p*(k+2)))

        # Initialize the linearization point to be empty.
        self.linearization_point = {'q': None, 'v': None, 'controls': None}

    def set_linearization_point(self, q, v, controls):
        """Set the linear approximation's point of linearization.  Note that
        this method will not will not do anything with the stored histories."""

        q, v, controls = q.squeeze(), v.squeeze(), controls.squeeze()

        assert q.shape == v.shape == (3,)

        # No need to check the shape of the controls since it could be any shape
        # as long as it matches the underlying self.system's control
        # representation.  Just make sure we didn't compress a (1,) array to ().
        if controls.shape == ():
            controls = controls.reshape(1)
        
        # Store the linearization point.
        self.linearization_point['q'] = q
        self.linearization_point['v'] = v
        self.linearization_point['controls'] = controls

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
        (6, p*(k+2))) at the provided. Note that the x is expected to be in the
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
        mat_2 = np.hstack((D, N, np.zeros((n,p))))

        return mat_1 @ M_inv @ mat_2

    def _get_f_3(self, x, u):
        """From the nonlinear notation, evaluate the value of f_3 (of shape
        (p*(k+2), 1)) at the provided x and u.  Note that the x is expected to
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

        # Need the mass matrix, contact jacobians, and continuous forces.
        M = polytope.get_M_matrix(state_sys)
        M_inv = np.linalg.inv(M)
        D = polytope.get_D_matrix(state_sys)
        N = polytope.get_N_matrix(state_sys)
        k = polytope.get_k_vector(state_sys).reshape(3,)

        # Build the expression.
        P_5 = np.hstack((np.zeros((p,n+1)), np.ones((p,1)), np.zeros((p,1))))
        P_6 = np.hstack((np.zeros((p,n+2)), np.ones((p,1))))

        mat_1 = np.vstack((D.T, N.T, np.zeros((p,n))))
        mat_2 = np.hstack((np.eye(n), np.zeros((n,n))))
        mat_3 = np.vstack((np.zeros((p*k_friction,2*n)), P_5,
                           np.zeros((p,2*n))))
        mat_4 = np.vstack((np.zeros((p*k_friction, p)), np.eye(p),
                           np.zeros((p,p))))

        diag_r = np.diag(radii)

        sin_vec = np.sin(P_6 @ x + angles)
        
        return mat_1 @ (mat_2@x + dt*M_inv@(k+u)) + (1/dt) * mat_3@x + \
               (1/dt) * mat_4@diag_r@sin_vec

    def _get_f_4(self, x):
        """From the nonlinear notation, evaluate the value of f_4 (of shape
        (p*(k+2), p*(k+2))) at the provided x.  Note that the x is expected to
        be in the LCS order [vx, vy, vth, x, y, th]."""
        
        # Get some necessary values from inside the system.
        polytope = self.system.params.polytope
        p = polytope.n_contacts

        # Will need the state vector in system form.
        state_sys = self._convert_lcs_state_to_system_state(x)

        # Need the mass matrix, contact jacobians, E matrix, and friction.
        M = polytope.get_M_matrix(state_sys)
        M_inv = np.linalg.inv(M)
        D = polytope.get_D_matrix(state_sys)
        N = polytope.get_N_matrix(state_sys)
        E = polytope.get_E_matrix(state_sys)
        Mu = polytope.get_mu_matrix(state_sys)
        
        # Build the expression.
        mat_top = np.hstack((D.T @ M_inv @ D, D.T @ M_inv @ N, E))
        mat_mid = np.hstack((N.T @ M_inv @ D, N.T @ M_inv @ N, np.zeros((p,p))))
        mat_bot = np.hstack((-E.T, Mu, np.zeros((p,p))))

        lcp_mat = np.vstack((mat_top, mat_mid, mat_bot))

        return lcp_mat

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

    def _get_df3_dx(self, x, _u):
        """From the nonlinear notation, evaluate the partial derivative of f3
        with respect to the state at the provided state and control input,
        yielding a jacobian of shape (p*(k+2),6).  Again, the state is expected
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
        P_5 = np.hstack((np.zeros((p,n+1)), np.ones((p,1)), np.zeros((p,1))))
        P_6 = np.hstack((np.zeros((p,n+2)), np.ones((p,1))))

        mat_1 = np.vstack((D.T, N.T, np.zeros((p,n))))
        mat_2 = np.hstack((np.eye(n), np.zeros((n,n))))
        mat_3 = np.vstack((np.zeros((p*k_friction,2*n)), P_5,
                           np.zeros((p,2*n))))
        mat_4 = np.vstack((np.zeros((p*k_friction, p)), np.eye(p),
                           np.zeros((p,p))))

        diag_r = np.diag(radii)

        cos_vec = np.cos(P_6 @ x + angles)
        diag_cos = np.diag(cos_vec)

        return mat_1 @ mat_2 + (1/dt) * mat_3 + \
               (1/dt) * mat_4 @ diag_r @ diag_cos @ P_6

    def _get_df3_du(self, x, _u):
        """From the nonlinear notation, evaluate the partial derivative of f3
        with respect to control input at the provided state and control input,
        yielding a jacobian of shape (p*(k+2),3).  Again, the state is expected
        to be in the LCS order [vx, vy, vth, x, y, th]."""
        
        # Get some necessary values from inside the system.
        dt = self.system.params.dt
        polytope = self.system.params.polytope
        n = polytope.n_config
        p = polytope.n_contacts

        # Will need the state vector in system form.
        state_sys = self._convert_lcs_state_to_system_state(x)

        # Need the mass matrix and contact jacobians.
        M = polytope.get_M_matrix(state_sys)
        M_inv = np.linalg.inv(M)
        D = polytope.get_D_matrix(state_sys)
        N = polytope.get_N_matrix(state_sys)

        # Build the expression.
        mat_1 = np.vstack((D.T, N.T, np.zeros((p,n))))

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
        P = self.get_P_matrix()

        return A, B, C, d, G, H, J, l, P

    def convert_input_to_generalized_coords(self, x, controls):
        """Convert an input set of controls (in whatever format is used by the
        underlying self.system) to forces in the generalized coordinates of the
        system, given its state."""

        return self.system.convert_input_to_generalized_coords(x, controls)

    def get_A_matrix(self):
        """Calculate the A matrix in the LCS approximation.  The linearization
        is done about the stored self.linearization_point, which must be set
        prior to calling this function."""

        # This requires using the state and control input stored in the
        # linearization.
        v, q = self.linearization_point['v'], self.linearization_point['q']
        x = np.hstack((v, q))
        controls = self.linearization_point['controls']

        # Convert the controls to generalized coordinates using the P map.
        P = self.get_P_matrix()
        u = P @ controls

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
        controls = self.linearization_point['controls']

        # Convert the controls to generalized coordinates using the P map.
        P = self.get_P_matrix()
        u = P @ controls

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
        controls = self.linearization_point['controls']

        # Convert the controls to generalized coordinates using the P map.
        P = self.get_P_matrix()
        u = P @ controls

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
        controls = self.linearization_point['controls']

        # Convert the controls to generalized coordinates using the P map.
        P = self.get_P_matrix()
        u = P @ controls

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
        controls = self.linearization_point['controls']

        # Convert the controls to generalized coordinates using the P map.
        P = self.get_P_matrix()
        u = P @ controls

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
        controls = self.linearization_point['controls']

        # Convert the controls to generalized coordinates using the P map.
        P = self.get_P_matrix()
        u = P @ controls

        # Build the expression.
        f3_val = self._get_f_3(x, u)
        df3_dx_val = self._get_df3_dx(x, u)
        df3_du_val = self._get_df3_du(x, u)

        return f3_val - df3_dx_val @ x - df3_du_val @ u

    def get_P_matrix(self):
        """Wrapper around the self.system's method calculating the linearized
        map from the native controls representation to generalized coordinates,
        self.get_map_from_controls_to_gen_coordinates(), using the system state
        and controls set as the linearization point."""

        # This requires using the state and control input stored in the
        # linearization.
        v, q = self.linearization_point['v'], self.linearization_point['q']
        x = np.hstack((v, q))
        controls = self.linearization_point['controls']

        state_sys = self._convert_lcs_state_to_system_state(x)
        return self.system.get_map_from_controls_to_gen_coordinates(state_sys,
                                                                    controls)

    def __check_consistent_histories(self):
        """Check that the history lengths are compatible."""
        state_len = self.state_history.shape[0]
        control_len = self.full_control_history.shape[0]
        lam_len = self.lambda_history.shape[0]
        out_len = self.output_history.shape[0]

        if state_len == 0:
            assert control_len == lam_len == out_len == 0

        else:
            assert state_len - control_len == 1
            assert control_len == lam_len == out_len

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
        self.full_control_history = np.zeros((0, 4))
        self.lambda_history = np.zeros((0, p*(k+2)))
        self.output_history = np.zeros((0, p*(k+2)))

    def step_lcs_dynamics(self, controls, lamda_k):
        """Given new control inputs and contact forces, step the system forward
        in time, appending the next state, provided controls, and provided
        contact forces to the end of the state and control history arrays,
        respectively."""

        # Check that the histories are consistent.
        self.__check_consistent_histories()

        # Get the current state and step the dynamics.
        state = self.state_history[-1, :]
        next_state, yk = self.__step_lcs_dynamics(state, controls, lamda_k)

        # Convert the controls to full controls.
        full_control = self.system.get_full_controls(state, controls)

        # Set the state and control histories.
        self.state_history = np.vstack((self.state_history, next_state))
        self.full_control_history = np.vstack((self.full_control_history,
                                               full_control))
        self.lambda_history = np.vstack((self.lambda_history, lamda_k))
        self.output_history = np.vstack((self.output_history, yk))

    def __step_lcs_dynamics(self, state, controls, lambda_k):
        """Given the current state, controls (expressed how the base system
        expresses controls, such as but not necessarily a (4,) array for
        [force_x, force_y, world_loc_x, world_loc_y]), and complementarity
        vector lambda_k (given as (p*(k+2),) array), simulate the LCS
        approximation one time step into the future.  This function uses the
        linearization point that is assumed to be already set via
        self.set_linearization_point().  This function does NOT check that the
        provided controls are valid (i.e. within friction cone and acting on the
        surface of the object).  Returns the next state."""

        # Get all the LCS terms.
        A, B, C, d, G, H, J, l, P = self.get_lcs_terms()

        # # Convert the control force and location to generalized coordinates.
        # state_sys = self._convert_lcs_state_to_system_state(state)
        # u = self.convert_input_to_generalized_coords(state_sys, controls)
        # P = self.get_P_matrix(state, controls)
        x = state
        u = P @ controls

        # Evaluate the expressions for x_{k+1} and y_k.
        x_k1 = A@x + B@u + C@lambda_k + d
        y_k = G@x + H@u + J@lambda_k + l

        return x_k1, y_k

