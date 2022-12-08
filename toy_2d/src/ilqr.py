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
from dataclasses import dataclass, field
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

@dataclass
class SolverParams:
    decay: float = 0.8
    max_iterations: int = 1000
    tolerance: float = 0.05

class iLQR:
    """
    iLQR class for solving the 1d and 2d force problem 
    Properties:
        lcs:                LCS approximation of the system containing the system functionalities 
                            inherited within.
        x_goal:             final goal point to be reached.
        N:                  Number of timesteps to solve the problem for before back propagation.
        test_number:        The current test number to be used while saving the outputs.
        Q:                  Cost penalty matrix on the error with the goal state. 
        R:                  Cost penalty matrix on the error with the nominal control (zeros in this case).  
        Qf:                 Final cost penalty matrix on the error with the goal state.  
        solver_params:      params for the iLQR solver including learning rate decay,
                            maximum iterations before termination, and tolerance.
    """
    solver_params: SolverParams

    def __init__(self, lcs, x_goal: np.ndarray, N: int, test_number:int, Q: np.ndarray, R: np.ndarray, Qf: np.ndarray, solver_params: SolverParams):
        self.lcs = lcs
        self.x_goal = x_goal
        self.u_goal = np.zeros((R.shape[0]))
        self.N = N
        self.test_num = test_number
        self.dt = lcs.system.params.dt
        self.Q = Q
        self.R = R
        self.Qf = Qf
        # number of states
        self.nx = Q.shape[0]
        # number of actions
        self.nu = R.shape[0]
        # Solver parameters
        self.decay = solver_params.decay #tune this to control decay of learning rate.
        self.max_iter = solver_params.max_iterations
        self.tol = solver_params.tolerance
        self.states = []
        self.actions= []
        self.curr_cost=0
        self.new_cost=0

    def total_cost(self, xx, uu):
        J = sum([self.running_cost(xx[k], uu[k]) for k in range(self.N - 1)])
        return J + self.terminal_cost(xx[-1])

    def running_cost(self, xk, uk):
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

        for i in range(self.N-1):
            utraj[i] = uu[i] + KK[i]@(xtraj[i] - xx[i]) + lr*dd[i]
            xtraj[i+1] = self.sim_forward(xtraj[i], utraj[i])
        return xtraj, utraj

    def get_linearized_discrete_dynamics(self, state, action):
        q = np.array([state[0], state[2], state[4]])
        v = np.array([state[1], state[3], state[5]])
        self.lcs.set_linearization_point(q,v, action)
        A, B_gen, C_mat, _, G_mat, H_mat, J_mat, _, pmat = self.lcs.get_lcs_terms()
        B = B_gen@pmat
        #access original dynamics to get the original lambdas
        curr_lam = self.get_lamda(state, action)
        eps = 1e-3
        #Check if there is a contact at all. If not use the normal dynamics itself as polytope is in air
        #the curr_lam vector is a vector of shape [4*n,] of which the first 2*n are tangential, the next n 
        #are normal and the last n are slack variables. Hence to see if the polytope made contact, we only
        #check if the tangential or normal values are greater than zero. Slack variables might be non-zero
        #even with no contact.
        tot_num_forces = 3*self.lcs.system.params.polytope.n_contacts
        if(np.argwhere(curr_lam[:tot_num_forces]>eps).shape[0]>0):
            curr_lam = curr_lam[curr_lam>eps]
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
        final_state = xx[self.N-1]
        V_xp1 = self.grad_terminal_cost(final_state) #jacobian of value func at the next state
        V_xxp1 = self.hess_terminal_cost(final_state) #hess of value func at the next state
        for i in range(self.N-2,-1,-1):
            state = xx[i]
            action = uu[i]
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

    def animate(self, init_state, controls):
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
        gif_name = "ilqr_result_" + str(self.test_num)
        # Generate a gif of the simulated rollout.
        vis_utils.animation_gif_polytope(self.lcs.system.params.polytope, states, gif_name, self.lcs.system.params.dt,
            controls=(control_forces, control_locs), save=True)     

    def save_contols(self, controls):
        file_name = "ilqr_controls_" + str(self.test_num)
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
            if(i%10==0 and i >1):
                print(f'cost: {self.curr_cost}')
                self.save_contols(self.actions)
                # self.animate(ini_state, self.actions)
                plt.plot(cost_arr)
                plt.xlabel("Iterations")
                plt.ylabel("Total Cost")
                plt.grid()
                plot_name = "plot_" + str(self.test_num)
                filename = f'{file_utils.OUT_DIR}/{plot_name}.png'
                plt.savefig(filename)
                # plt.show()
            i += 1
            

        print(f'Converged to cost {self.curr_cost}')
        self.animate(init_state, self.actions)




