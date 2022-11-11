"""This file describes a toy 2D system of an arbitrary polytope.

The state keeps track of the center of mass x and y positions plus the angle
theta from the ground's axes to the body's axes, in addition to time derivatives
of all 3 of these quantities:  thus the state vector is 6-dimensional.
"""

import numpy as np
import pdb
import matplotlib.pyplot as plt

from toy_2d.src import vis_utils
from toy_2d.src import file_utils
from toy_2d.src.two_dim_polytope import TwoDimensionalPolytopeParams, \
                                        TwoDimensionalPolytope
from toy_2d.src.two_dim_system import TwoDimensionalSystemParams, \
                                      TwoDimensionalSystem,TwoDSystemMagOnly

from toy_2d.src.two_dim_lcs_approximation import TwoDSystemLCSApproximation


class iLQR():

    def __init__(self, x_0: np.ndarray, x_goal: np.ndarray, N: int, dt: float, Q: np.ndarray, R: np.ndarray, Qf: np.ndarray):
        self.x0 = x0
        self.x_goal = x_goal
        self.u_goal = np.zeros((1))
        self.N = N
        self.dt = dt
        self.Q = Q
        self.R = R
        self.Qf = Qf
        # number of states
        self.nx = 6
        # number of actions
        self.nu = 1
        # Solver parameters
        self.alpha = 1         
        self.max_iter = 1000
        self.tol = 1e-4

    def total_cost(self, xx, uu):
        J = sum([self.running_cost(xx[k], uu[k]) for k in range(self.N - 1)])
        return J + self.terminal_cost(xx[-1])

    def running_cost(self, xk, uk):
        # print(uk.shape, self.R.shape)
        lqr_cost = 0.5 * ((xk - self.x_goal).T @ self.Q @ (xk - self.x_goal) +
                          (uk - self.u_goal).T @ self.R @ (uk - self.u_goal))
        return lqr_cost

    def grad_running_cost(self, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        """
        :param xk: state
        :param uk: input
        :return: [∂l/∂xᵀ, ∂l/∂uᵀ]ᵀ, evaluated at xk, uk
        """
        grad = np.zeros((self.nx+self.nu,))

        #TODO: Compute the gradient

        x_grad = (xk- self.x_goal).T @ self.Q
        u_grad = (uk- self.u_goal).T @ self.R
        grad[:6] = x_grad
        grad[6:] = u_grad

        return grad

    def hess_running_cost(self, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        """
        :param xk: state
        :param uk: input
        :return: The hessian of the running cost
        [[∂²l/∂x², ∂²l/∂x∂u],
         [∂²l/∂u∂x, ∂²l/∂u²]], evaluated at xk, uk
        """
        H = np.zeros((self.nx + self.nu, self.nx + self.nu))

        # TODO: Compute the hessian
        H[:6,:6] = self.Q
        H[6:, 6:] = self.R

        return H

    def terminal_cost(self, xf: np.ndarray) -> float:
        """
        :param xf: state
        :return: Lf(xf), the running cost incurred by xf
        """
        return 0.5*(xf - self.x_goal).T @ self.Qf @ (xf - self.x_goal)

    def grad_terminal_cost(self, xf: np.ndarray) -> np.ndarray:
        """
        :param xf: final state
        :return: ∂Lf/∂xf
        """

        grad = np.zeros((self.nx))

        # TODO: Compute the gradient
        x_grad = (xf- self.x_goal).T @ self.Qf
        grad = x_grad
        return grad
        
    def hess_terminal_cost(self, xf: np.ndarray) -> np.ndarray:
        """
        :param xf: final state
        :return: ∂²Lf/∂xf²
        """ 

        H = np.zeros((self.nx, self.nx))
        H = self.Qf

        # TODO: Compute H

        return H

    def forward_pass(self, xx, uu, dd, KK):
        """
        :param xx: list of states, should be length N
        :param uu: list of inputs, should be length N-1
        :param dd: list of "feed-forward" components of iLQR update, should be length N-1
        :param KK: list of "Feedback" LQR gain components of iLQR update, should be length N-1
        :return: A tuple (xtraj, utraj) containing the updated state and input
                 trajectories after applying the iLQR forward pass
        """

        xtraj = [np.zeros((self.nx,))] * self.N
        utraj = [np.zeros((self.nu,))] * (self.N - 1)
        xtraj[0] = xx[0]

        # TODO: compute forward pass
        for i in range(self.N-1):
            utraj[i] = uu[i] + KK[i]@(xtraj[i] - xx[i]) + self.alpha*dd[i]
            #TO DO
            # xtraj[i+1] = quad_sim.F(xtraj[i], utraj[i], self.dt)
            xtraj[i+1] = self.sim_forward(xtraj[i], utraj[i])
        return xtraj, utraj

    def get_linearized_discrete_dynamics(self, state, action):
        q = np.array([state[0], state[2], state[4]])
        v = np.array([state[1], state[3], state[5]])
        # control_loc = polytope.get_vertex_locations_world(state)[2, :]
        # theta = state[4]
        # ang = np.pi + theta
        # control_mag = action[0]
        # control_vec = control_mag * np.array([-np.cos(ang), -np.sin(ang)])
        lcs.set_linearization_point(q,v, action)
        A, B_gen = lcs.get_A_matrix(), lcs.get_B_matrix()
        pmat = lcs.get_P_matrix()
        B = B_gen@pmat
        # B = self.convert_gen_to_force(state, action, B_gen)

        return A,B
    
    
    # def convert_gen_to_force(self, state, action, B_gen):
    #     control_loc = polytope.get_vertex_locations_world(state)[2, :]
    #     theta = state[4]
    #     ang = np.pi + theta
    #     control_mag = action[0]
    #     control_vec = control_mag * np.array([-np.cos(ang), -np.sin(ang)])
    #     fx, fy = control_vec[0], control_vec[1]
    #     u_locx, u_locy = control_loc[0], control_loc[1]
    #     x, y = state[0], state[2]
    #     lever_x = u_locx - x
    #     lever_y = u_locy - y
    #     conv_mat = np.eye(3,2)
    #     conv_mat[-1,:] = np.array([-lever_y, lever_x])
    #     B_two_ctrls = B_gen@conv_mat #this is 6x2... convert to 6x1
    #     theta = state[4]
    #     ang = np.pi + theta
    #     control_vec = np.array([-np.cos(ang), -np.sin(ang)])
    #     B = (B_two_ctrls@control_vec).reshape((self.nx, self.nu))
    #     return B


    def backward_pass(self,  xx, uu):
        """
        :param xx: state trajectory guess, should be length N
        :param uu: input trajectory guess, should be length N-1
        :return: KK and dd, the feedback and feedforward components of the iLQR update
        """
        dd = [np.zeros((self.nu,))] * (self.N - 1)
        KK = [np.zeros((self.nu, self.nx))] * (self.N - 1)
        import sys
        # TODO: compute backward pass
        V_xp1 = np.zeros((self.nx)) #jacobian of value func at the next state
        V_xxp1 = np.zeros((self.nx, self.nx)) #hess of value func at the next state
        for i in range(self.N-1,-1,-1):
            # print(i)
            state = xx[i]
            if(i == self.N-1):
                V_xp1 = self.grad_terminal_cost(state)
                V_xxp1 = self.hess_terminal_cost(state)
                continue
            action = uu[i]
            #TODO
            A,B = self.get_linearized_discrete_dynamics(state, action)
            # print(V_xp1.shape)
            # A_mats[:,:,i] = A
            # B_mats[:,:,i] = B
            grad_mat = self.grad_running_cost(state,action)
            l_x = grad_mat[:6]
            l_u = grad_mat[6:] 
            hess_mat = self.hess_running_cost(state, action)
            l_uu = hess_mat[6:, 6:]
            l_xx = hess_mat[:6,:6]
            Q_u = l_u + B.T@V_xp1 #(2,)
            # print(B.shape)
            Q_uu = l_uu + B.T@V_xxp1@B #(2,2)
            Q_ux = B.T@V_xxp1@A #(2,6)
            Q_x = l_x + A.T@V_xp1
            Q_xx = l_xx + A.T@V_xxp1@A
            KK[i] = -np.linalg.inv(Q_uu)@Q_ux #(2,6)
            dd[i] = -np.linalg.inv(Q_uu)@Q_u #(2,)
            # KK[i] = -np.linalg.solve(Q_uu,Q_ux) #(2,6)
            # dd[i] = -np.linalg.solve(Q_uu,Q_u) #(2,)

            V_xp1 = Q_x - KK[i].T@Q_uu@dd[i]
            V_xxp1 = Q_xx - KK[i].T@Q_uu@KK[i]
        return dd, KK

    def sim_forward(self, state, action):
        system.set_initial_state(state)
        # control_loc = polytope.get_vertex_locations_world(state)[2, :]
        # theta = state[4]
        # ang = np.pi + theta
        # control_mag = action[0]
        # control_vec = control_mag * np.array([-np.cos(ang), -np.sin(ang)])
        # controls = np.hstack((control_vec, control_loc))
        # system.step_dynamics(control_vec, control_loc)
        controls = action
        system.step_dynamics(controls)
        next_state = system.state_history[-1,:]
        return next_state

    def animate(self, init_state, controls, show_vis=False):
        # Rollout with a fixed (body-frame) force at one of the vertices.
        x = init_state
        uu = controls
        system.set_initial_state(x)
        for o in range(len(uu)):
            state = system.state_history[-1, :]

            # Find the third vertex location.
            # control_loc = polytope.get_vertex_locations_world(state)[2, :]

            # Apply the force at a fixed angle relative to the polytope.
            # theta = state[4]
            if(o==len(uu)-1):
                print(state)
            # ang = np.pi + theta
            # control_mag = uu[o]
            # control_vec = control_mag * np.array([-np.cos(ang), -np.sin(ang)])
            # control_vec = uu[o]
            # control_vec = control_mag * np.array([-0.5, 0])

            system.step_dynamics(uu[o])

        # Collect the state and control histories.
        states = system.state_history
        controls = system.control_history
        control_forces, control_locs = controls[:, :2], controls[:, 2:]

        # pdb.set_trace()

        # Generate a gif of the simulated rollout.
        vis_utils.animation_gif_polytope(polytope, states, 'square', DT,
            controls=(control_forces, control_locs), save=False)     

    def save_contols(self, controls, iter):
        file_name = "controls"
        path = f'{file_utils.OUT_DIR}/{file_name}.txt'
        file1 = open(path, "w")
        # with open(path, "w") as txt_file:
        for line in controls:
            file1.write(str(line[0]) + "\n") 
        file1.close()

    def calculate_optimal_trajectory(self, x, uu_guess) :

        """
        Calculate the optimal trajectory using iLQR from a given initial condition x,
        with an initial input sequence guess uu
        :param x: initial state
        :param uu_guess: initial guess at input trajectory
        :return: xx, uu, KK, the input and state trajectory and associated sequence of LQR gains
        """
        print(self.N)
        assert (len(uu_guess) == self.N - 1)

        # Get an initial, dynamically consistent guess for xx by simulating the cube
        # system.set_initial_state(x)
        xx = [x]
        for k in range(self.N-1):
            nxt_state = self.sim_forward(xx[k], uu_guess[k])
            xx.append(nxt_state)


        Jprev = np.inf
        Jnext = self.total_cost(xx, uu_guess)
        uu = uu_guess
        KK = None

        i = 0
        print(f'first cost: {Jnext}')
        while np.abs(Jprev - Jnext) > self.tol and i < self.max_iter:
            dd, KK = self.backward_pass(xx, uu)
            # print(np.array(dd).shape, np.array(KK).shape)
            xx, uu = self.forward_pass(xx, uu, dd, KK)

            Jprev = Jnext
            Jnext = self.total_cost(xx, uu)
            i += 1
            if(i%20==0 and i >1):
                print(f'cost: {Jnext}')
                self.save_contols(uu, i)
                self.animate(x, uu, show_vis=True)

        print(f'Converged to cost {Jnext}')
        self.animate(x, uu, show_vis=True)
        return xx, uu, KK



# Fixed parameters
# A few polytope examples.
SQUARE_CORNERS = np.array([[1, -1], [1, 1], [-1, 1], [-1, -1]])
STICK_CORNERS = np.array([[1, 0], [-1, 0]])
RAND_CORNERS = np.array([[0.5, 0], [0.7, 0.5], [0, 0.8], [-1.2, 0], [0, -0.5]])

# Polytope properties
MASS = 1
MOM_INERTIA = 0.1
MU_GROUND = 1e6

# Control properties
MU_CONTROL = 0.5    # Currently, this isn't being used.  The ambition is for
                    # this to help define a set of feasible control forces.

# Simulation parameters.
DT = 0.001          # If a generated trajectory looks messed up, it could be
                    # fixed by making this timestep smaller.

# Initial conditions, in order of x, dx, y, dy, theta, dtheta
# x0 = np.array([0, 0, 1.5, 0, -1/6 * np.pi, 0])



# Create a polytope.
poly_params = TwoDimensionalPolytopeParams(
    mass = MASS,
    moment_inertia = MOM_INERTIA,
    mu_ground = MU_GROUND,
    vertex_locations = SQUARE_CORNERS
)
polytope = TwoDimensionalPolytope(poly_params)

# Create a system from the polytope, a simulation timestep, and a control
# contact's friction parameter.
system_params = TwoDimensionalSystemParams(
    dt = DT,
    polytope = polytope,
    mu_control = MU_CONTROL
)
# system = TwoDimensionalSystem(system_params)

# Contact location and direction.
CONTACT_LOC = np.array([-1, 1])
CONTACT_ANGLE = 0.

system = TwoDSystemMagOnly(system_params, CONTACT_LOC, CONTACT_ANGLE)


# Create an LCS approximation from the system.
lcs = TwoDSystemLCSApproximation(system)

#Following the hw methodology of implementing ILQR
#take the current state and action and give the current cost. Here the cost can be 
#non-linear but defining a quadratic cost makes life easier
# x0 = np.array([0, 0, 1.4, 0, -1/8 * np.pi, 0])
x0 = np.array([0, 0, 1.0, 0, 0, 0])
x_goal = np.array([2.0, 0, 1.0, 0, -np.pi/2, 0])
Q = np.eye(6)
Q[1,1]=Q[1,1]*10
Q[3,3]=Q[3,3]*10
Q[5,5]=Q[5,5]*10
# Q[0,0]=Q[0,0]*10
Qf = np.eye(6)*10
# Qf[1,1]=0
# Qf[3,3]=0
# Qf[5,5]=0
R = np.eye(1)*0
# R = np.zeros((2,2))

N = 1200
u_guess = np.ones((N-1,1))*5
load = True
if(load):
    file_name = "controls"
    path = f'{file_utils.OUT_DIR}/{file_name}.txt'
    with open(path, 'r') as fd:
    # reader = csv.reader(fd)
        j=0
        for row in fd:
            u_guess[j] = row
            j+=1

obj = iLQR(x0, x_goal, N, DT, Q, R, Qf)
obj.calculate_optimal_trajectory(x0, u_guess)

# Rollout with a fixed (body-frame) force at one of the vertices.
# system.set_initial_state(x0)
# for _ in range(1250):
#     state = system.state_history[-1, :]

#     # Find the third vertex location.
#     control_loc = polytope.get_vertex_locations_world(state)[2, :]

#     # Apply the force at a fixed angle relative to the polytope.
#     theta = state[4]
#     ang = np.pi + theta
#     control_mag = 0.0
#     control_vec = control_mag * np.array([-np.cos(ang), -np.sin(ang)])

#     system.step_dynamics(control_vec, control_loc)

# Collect the state and control histories.
# states = system.state_history
# controls = system.control_history
# control_forces, control_locs = controls[:, :2], controls[:, 2:]

# pdb.set_trace()

# # Generate a gif of the simulated rollout.
# vis_utils.animation_gif_polytope(polytope, states, 'small_force', DT,
#     controls=(control_forces, control_locs))

