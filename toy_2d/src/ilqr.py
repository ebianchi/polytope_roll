"""This file tries to solve the unconstrained optimization problem using the iLQR method
The cost we want to minimize is

           min         sum_{k=0}^{N-1} x_k^T Q x_k + u_k^T R u_k + x_N^T Q_N x_N
           u_k

        such that   x_{k+1} = f(x,u) + f(x)*lambda

Given an initial guess, the problem first performs a rollout and then uses the backward pass to calculate the new controls by linearizing
the dynamics and solving the riccati equation. 

During the backward pass, in order to capture the effect of lamda in the linearized dynamics we have,

        x_{k+1} = A x_k + B u_k + C lambda_k + d
        y_k = G x_k + H u_k + J lambda_k + l
        
        Also, 0=< lamda_k has a complementarity constraint with y_k=>0

Using this, we first zero out the columns in C and J where lamda_k is zero, then if lamda_k>0 then y_k =0 as
per the constriant. Hence, 
        0 = G x_k + H u_k + J lambda_k + l
        lamda_k = -J^{+}(G x_k + H u_k + l)
        lamda_k = A' x_k + B' u_k + l'

Substituting this back in original dyanmics eqaution gives us A'' and B'' which is used in the backward pass

        x_{k+1} = A'' x_k + B'' u_k + d''

"""

import numpy as np
import pdb
import matplotlib.pyplot as plt
import pickle
from toy_2d.src import vis_utils
from toy_2d.src import file_utils
from toy_2d.src.two_dim_polytope import TwoDimensionalPolytopeParams, \
                                        TwoDimensionalPolytope
from toy_2d.src.two_dim_system import TwoDimensionalSystemParams, \
                                      TwoDimensionalSystem,TwoDSystemMagOnly, TwoDSystemForceOnly

from toy_2d.src.two_dim_lcs_approximation import TwoDSystemLCSApproximation


class iLQR():

    def __init__(self, lcs, x_goal: np.ndarray, N: int, Q: np.ndarray, R: np.ndarray, Qf: np.ndarray):
        self.lcs = lcs
        self.x_goal = x_goal
        self.u_goal = np.zeros((R.shape[0]))
        self.N = N
        self.dt = lcs.system.params.dt
        self.Q = Q
        self.R = R
        self.Qf = Qf
        # number of states
        self.nx = Q.shape[0]
        # number of actions
        self.nu = R.shape[0]
        # Solver parameters
        self.decay = 0.8 #tune this to control decay of learning rate.
        self.max_iter = 1000
        self.tol = 0.05
        self.states = []
        self.actions= []
        self.curr_cost=0
        self.new_cost=0

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
        grad[:self.nx] = x_grad
        grad[self.nx:] = u_grad

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
        H[:self.nx,:self.nx] = self.Q
        H[self.nx:, self.nx:] = self.R

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

    def forward_pass(self, xx, uu, dd, KK, lr):
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
            utraj[i] = uu[i] + KK[i]@(xtraj[i] - xx[i]) + lr*dd[i]
            # if(utraj[i][0]<0 and utraj[i][1]>0): utraj[i] = np.array([0,0])
            #TO DO
            # xtraj[i+1] = quad_sim.F(xtraj[i], utraj[i], self.dt)
            xtraj[i+1] = self.sim_forward(xtraj[i], utraj[i])
        return xtraj, utraj

    def get_linearized_discrete_dynamics(self, state, action):
        q = np.array([state[0], state[2], state[4]])
        v = np.array([state[1], state[3], state[5]])
        self.lcs.set_linearization_point(q,v, action)
        A, B_gen = self.lcs.get_A_matrix(), self.lcs.get_B_matrix()
        pmat = self.lcs.get_P_matrix()
        B = B_gen@pmat
        C_mat = self.lcs.get_C_matrix()
        G_mat = self.lcs.get_G_matrix()
        J_mat = self.lcs.get_J_matrix()
        H_mat = self.lcs.get_H_matrix()
        #access original dynamics to get the original lambdas
        curr_lam = self.get_lamda(state, action)
        eps = 1e-3
        # check if there is a contact at all. if not use the normal dynamics itself as cube is in air
        if(np.argwhere(curr_lam>eps).shape[0]>0):
            curr_lam = curr_lam[curr_lam>eps]
            # print(curr_lam)
            C_mat = C_mat[:, np.where(curr_lam>eps)[0]]
            J_mat = J_mat[:, np.where(curr_lam>eps)[0]]
            J_pinv = np.linalg.pinv(J_mat)
            A_d, B_d = -C_mat@J_pinv@G_mat, -C_mat@J_pinv@H_mat@pmat
            A, B = A + A_d, B+B_d

        return A,B
    
    def get_lamda(self, state, action):
        self.lcs.system.set_initial_state(state)
        self.lcs.system.step_dynamics(action)
        return self.lcs.system.lambda_history[-1,:]

    def backward_pass(self,  xx, uu):
        """
        :param xx: state trajectory guess, should be length N
        :param uu: input trajectory guess, should be length N-1
        :return: KK and dd, the feedback and feedforward components of the iLQR update
        """
        dd = [np.zeros((self.nu,))] * (self.N - 1)
        KK = [np.zeros((self.nu, self.nx))] * (self.N - 1)
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
            grad_mat = self.grad_running_cost(state,action)
            l_x = grad_mat[:self.nx]
            l_u = grad_mat[self.nx:] 
            hess_mat = self.hess_running_cost(state, action)
            l_uu = hess_mat[self.nx:, self.nx:]
            l_xx = hess_mat[:self.nx,:self.nx]
            Q_u = l_u + B.T@V_xp1 
            Q_uu = l_uu + B.T@V_xxp1@B 
            Q_ux = B.T@V_xxp1@A 
            Q_x = l_x + A.T@V_xp1
            Q_xx = l_xx + A.T@V_xxp1@A
            KK_curr = -np.linalg.inv(Q_uu)@Q_ux
            dd_curr = -np.linalg.inv(Q_uu)@Q_u
            KK[i] =  KK_curr
            dd[i] =  dd_curr
            
            V_xp1 = Q_x - KK[i].T@Q_uu@dd[i]
            V_xxp1 = Q_xx - KK[i].T@Q_uu@KK[i]
        return dd, KK

    def sim_forward(self, state, action):
        self.lcs.system.set_initial_state(state)
        self.lcs.system.step_dynamics(action)
        next_state = self.lcs.system.state_history[-1,:]
        return next_state

    def animate(self, init_state, controls, show_vis=False):
        # Rollout with a fixed (body-frame) force at one of the vertices.
        x = init_state
        uu = controls
        self.lcs.system.set_initial_state(x)
        for o in range(len(uu)):
            state = self.lcs.system.state_history[-1, :]
            if(o==len(uu)-1):
                print(state)
            self.lcs.system.step_dynamics(uu[o])

        # Collect the state and control histories.
        states = self.lcs.system.state_history
        controls = self.lcs.system.control_history
        control_forces, control_locs = controls[:, :2], controls[:, 2:]
        # Generate a gif of the simulated rollout.
        vis_utils.animation_gif_polytope(self.lcs.system.params.polytope, states, 'square', self.lcs.system.params.dt,
            controls=(control_forces, control_locs), save=True)     

    def save_contols(self, controls, iter):
        file_name = "controls"
        path = f'{file_utils.OUT_DIR}/{file_name}.txt'
        with open(path, 'wb') as fp:
            pickle.dump(controls, fp)

    def calculate_optimal_trajectory(self, x, uu_guess) :

        """
        Calculate the optimal trajectory using iLQR from a given initial condition x,
        with an initial input sequence guess uu
        :param x: initial state
        :param uu_guess: initial guess at input trajectory
        :return: xx, uu, KK, the input and state trajectory and associated sequence of LQR gains
        """
        assert (len(uu_guess) == self.N - 1)
        init_state = x
        # Get an initial, dynamically consistent guess for xx by simulating the cube
        # system.set_initial_state(x)
        self.states = [init_state]
        self.actions = uu_guess
        for k in range(self.N-1):
            nxt_state = self.sim_forward(self.states[k], self.actions[k])
            self.states.append(nxt_state)

        self.curr_cost = self.total_cost(self.states, self.actions)
        i = 0
        cost_arr = []
        print(f'first cost: {self.curr_cost}')
        while i < self.max_iter:
            dd, KK = self.backward_pass(self.states, self.actions)
            learning_rate = 1
            while(learning_rate>=0.05):
                xx_new, uu_new = self.forward_pass(self.states, self.actions, dd, KK, learning_rate)
                self.new_cost = self.total_cost(xx_new, uu_new)
                cost_diff = self.new_cost - self.curr_cost
                if(cost_diff<0):
                    cost_arr.append(self.curr_cost)
                    self.curr_cost =self.new_cost 
                    self.states = xx_new
                    self.actions = uu_new
                    break
                else:
                    learning_rate *=self.decay
                    print("Learning rate decreased")
            if(learning_rate<0.05):
                print("Step size has become too small to move. Ending optimization") 
                break

            i += 1
            if(i%2==0 and i >1):
                print(f'cost: {self.curr_cost}')
                self.save_contols(self.actions, i)
                # self.animate(ini_state, self.actions, show_vis=True)
                plt.plot(cost_arr)
                plt.xlabel("Iterations")
                plt.ylabel("Total Cost")
                plt.grid()
                filename = f'{file_utils.OUT_DIR}/plot.png'
                plt.savefig(filename)
                # plt.show()

        print(f'Converged to cost {self.curr_cost}')
        self.animate(init_state, self.actions, show_vis=True)
        # return self.states, self.actions, KK




