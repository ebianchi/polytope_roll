"""
SYNOPSIS
Implementation of ADMM based layered control architecture for a 2D polytope
interacting with a flat ground and a "robot end effector" that can contact the
polytope at a body-fixed point and exert a friction cone constrained force.  Due
to the contact-rich nature of the example, the formulation introduces additional
variables (lambda) representing contact forces that are distinct from states (x)
and inputs (u) but can be determined as a function of x and u by solving a
complementarity problem.  As such, the ADMM formulation introduces additional
dual variables (gamma) for the contact forces.

DESCRIPTION
Uses Gurobi to solve the r-subproblem and (TBD) iLQR from trajax to solve the
(x, u)-subproblem. Dual variables are updated according to the rule specified in
Boyd's paper.  (TBD) The code has been tested for various initial states,
maximum speed bounds, horizon and granularity of discretization.

AUTHORS
Anusha Srikanthan <sanusha@seas.upenn.edu>
Bibit Bianchini <bibit@seas.upenn.edu>
"""

# import cvxpy as cp
import copy
import numpy as np
import jax
import jax.numpy as jnp
from trajax import optimizers
from trajax.integrators import rk4

import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
from functools import partial
import time
import pdb

from toy_2d.src.optimization_utils import GurobiModelHelper


class admm_lca(object):
    """This code is an implementation of ADMM for deriving a layered control
    architecture (LCA) for any nonlinear dynamical system.

    Input arguments
        traj_opt_object: a TwoDTrajectoryOptimization object
        x_init: initial condition
        x_goal: final state
        rho: ADMM parameter for LCA problem
        tol: ADMM tolerance for terminating iterations

    Member functions
        construct_LCS_terms_from_inputs: function to construct the LCS terms
            by simulating the system with the current control inputs and using
            the resulting state trajectory as linearization points.
        solve_reference_gurobi: function to solve one instance of reference
            planning layer using Gurobi and relying on GurobiModelHelper.
        solve_control_gurobi: function to solve one instance of the feedback
            control layer using Gurobi.
        run_admm: function to run ADMM iterations until desired tolerance is
            achieved.
        rollout: function to compute state trajectory given initial conditions
            and input
    """
    # def __init__(self, dynamics, num_timesteps, dt, x_init, u_init, x_goal, rho,
    #              planning_state_idx=None, constr=None):
    def __init__(self, traj_opt_object, x_init, x_goal, rho, tol):
        assert x_init.ndim == x_goal.ndim == 1, 'Expected all to be of ' \
            f'dimension 1: {x_init.ndim=}, {x_goal.ndim=}'
        assert x_init.shape[0] == x_goal.shape[0] == traj_opt_object.n_state, \
            f'Expected all shapes to match: {x_init.shape=}, {x_goal.shape=},' \
            f' {traj_opt_object.n_state=}'
        # assert len(planning_state_idx) <= x_init.shape[0]

        # self.dynamics = dynamics
        self.dt = traj_opt_object.params.traj_opt_dt
        self.x_init = x_init
        self.x_goal = x_goal

        self.N = traj_opt_object.params.lookahead
        self.n = traj_opt_object.n_state
        self.m = traj_opt_object.n_controls
        self.k = traj_opt_object.n_friction
        self.p = traj_opt_object.n_contacts

        self.traj_opt_object = traj_opt_object

        self.rho = rho
        self.tol = tol

        # TODO: allow planning states to be subset of true states.  Or not,
        # since this contact rich example requires all states for
        # complementarity constraints.
        self.reduced_n = self.n
        self.Tr = np.eye(self.n)
        # if planning_state_idx is not None:
        #     self.reduced_n = len(planning_state_idx)
        #     self.Tr = np.zeros((self.n, len(self.reduced_n)))
        #     for i in range(len(planning_state_idx)):
        #         self.Tr[planning_state_idx[i], i] = 1

        # Instantiate sizes of vectors over time.
        self.x = np.zeros((self.N + 1, self.n))
        self.u = np.zeros((self.N, self.m))

        # Lambda vector contains 1 normal force, 1 tangential force for each
        # friction direction (self.k), and 1 slack variable for every contact
        # (self.p), yielding p*(k+2) size.
        self.lam = np.zeros((self.N, self.p*(self.k+2)))

        # Allow the dual variable r to be of reduced size from true states x.
        self.r = np.zeros((self.N + 1, self.reduced_n))

        self.a = np.zeros_like(self.u)
        self.gamma = np.zeros_like(self.lam)
        self.vr = np.zeros_like(self.r)
        self.vu = np.zeros_like(self.a)
        self.vgamma = np.zeros_like(self.gamma)

        # TODO: allow input constraints.
        # if constr is not None:
        #     self.constr_idx, self.constr = constr
        # else:
        self.constr_idx = None
        self.constr = None

        self.R = self.traj_opt_object.params.R

        # LCS terms to get filled in about a trajectory of states.  For now,
        # initialize them about the initial state for the full horizon.
        init_traj = [x_init] * self.N
        self.As, self.Bs, self.Cs, self.ds, self.Gs, self.Hs, self.Js, \
            self.ls, self.Ps, self.Qs, self.Rs, self.Ss = \
            self.traj_opt_object.get_lcs_plus_terms_over_horizon(init_traj)

        # To eventually fill in the LCS terms adaptively, will need to simulate
        # a coarsely-timestepped system that emulates the true (i.e. sim)
        # system.
        self.coarse_sim_system = copy.deepcopy(
            traj_opt_object.params.sim_system)
        self.coarse_sim_system.params.dt = self.dt

    def construct_LCS_terms_from_inputs(self):
        """Using the stored control inputs in self.u, simulate a coarse system
        with the inputs to get a trajectory of states.  Use those states as the
        linearization knot points for the LCS terms."""
        # Simulate the coarse system to get the trajectory of states.
        self.coarse_sim_system.simulate_dynamics_over_horizon(
            controls_over_horizon=self.u, init_state=self.x_init)

        x_traj = self.coarse_sim_system.state_history
        assert x_traj.shape == (self.N+1, self.n), f'Expected shape ' \
            f'{(self.N+1, self.n)}, got {x_traj.shape}'
        x_traj = x_traj[:-1, :]

        # Get the LCS terms from this state trajectory.
        self.As, self.Bs, self.Cs, self.ds, self.Gs, self.Hs, self.Js, \
            self.ls, self.Ps, _Qs, _Rs, _Ss = \
            self.traj_opt_object.get_lcs_plus_terms_over_horizon(x_traj)
        print('Updated LCS terms from self.u')

    def solve_reference_gurobi(self):
        """Build and solve a Gurobi optimization problem for the trajectory
        generation (or "reference") layer.  This is just like in traj_opt but
        without dynamics constraints and including dual costs in the objective.
        This implements Equation (6a) in the paper.

        This step requires that self.x, self.u, and self.lam are already set to
        numerical values.  It also requires the LCS terms self.As, self.Bs,
        self.Cs, self.ds, self.Gs, self.Hs, self.Js, self.ls, self.Ps, self.Qs,
        self.Rs, and self.Ss are set as well.  These LCS terms, self.u, and
        self.lam will be of length self.N while self.x will be of length
        self.N+1.

        [1] A. Srikanthan, V. Kumar, N. Matni, "Augmented Langrangian Methods as
        Layered Control Architectures," 2023.
        """
        # TODO: x_current as x_init works for first time, but may need to change
        # this if doing RHC.
        x_current = self.x_init

        # Grab a few variables for convenience.
        mu_control = self.traj_opt_object.params.sim_system.params.mu_control
        input_limit = self.traj_opt_object.params.input_limit
        use_big_M = self.traj_opt_object.params.use_big_M

        # Build a Gurobi optimization model.
        model = GurobiModelHelper.create_optimization_model(
            "admm_trajectory_generation_layer", verbose=False)
        
        # Create variables.
        r = GurobiModelHelper.create_xs(model, lookahead=self.N, n=self.n)
        r_err = GurobiModelHelper.create_x_errs(model, lookahead=self.N,
                                                n=self.n)
        a = GurobiModelHelper.create_us(model, lookahead=self.N, nu=self.m,
                                        input_limit=input_limit)
        gamma = GurobiModelHelper.create_lambdas(model, lookahead=self.N,
                                                 p=self.p, k=self.k)
        y = GurobiModelHelper.create_ys(model, lookahead=self.N, p=self.p,
                                        k=self.k)

        # Build constraints.  Explicitly exclude the dynamics constraint.
        # TODO: check if the initial condition constraint is needed / more of a
        # need to check what x_current should be.
        model = GurobiModelHelper.add_initial_condition_constr(
            model, xs=r, x_current=x_current)
        model = GurobiModelHelper.add_error_coordinates_constr(
            model, lookahead=self.N, xs=r, x_errs=r_err, x_goal=self.x_goal)
        model = GurobiModelHelper.add_complementarity_constr(
            model, lookahead=self.N, use_big_M=use_big_M, xs=r, us=a,
            lambdas=gamma, ys=y, p=self.p, k=self.k)
        model = GurobiModelHelper.add_output_constr(
            model, lookahead=self.N, xs=r, us=a, lambdas=gamma, ys=y, G=self.Gs,
            H=self.Hs, P=self.Ps, J=self.Js, l=self.ls)
        model = GurobiModelHelper.add_friction_cone_constr(
            model, lookahead=self.N, mu_control=mu_control, us=a)
        
        # Set the model's objective:  use stage cost to encourage smoothness,
        # dual error cost to encourage convergence, and final error to encourage
        # goal progress.
        obj = 0
        for i in range(self.N):
            stage_err = r[i+1, :] - r[i, :]
            obj += 0.1 * stage_err @ stage_err

            state_dual_err = self.x[i, :]@self.Tr - r[i, :] + self.vr[i, :]
            obj += (self.rho/2) * state_dual_err @ state_dual_err

            control_dual_err = self.u[i, :] - a[i, :] + self.vu[i, :]
            obj += (self.rho/2) * control_dual_err @ control_dual_err

            comp_dual_err = self.lam[i, :] - gamma[i, :] + self.vgamma[i, :]
            obj += (self.rho/2) * comp_dual_err @ comp_dual_err

        state_dual_err = self.x[self.N, :]@self.Tr - r[self.N, :] + \
            self.vr[self.N, :]
        obj += (self.rho/2) * state_dual_err @ state_dual_err

        final_err = r[-1, :] - self.x_goal
        obj += 1000 * final_err @ final_err
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Solve the optimization problem.
        try:
            model.optimize()
            print(f'Reference Obj: {model.ObjVal}')

            # Store the solved r, a, gamma variables.
            self.r = r.X
            self.a = a.X
            self.gamma = gamma.X

            # Avoid the pdb trace for error handling if this succeeded.
            return

        except gp.GurobiError as e:  print(f'Error code {e.errno}: {e}')
        except AttributeError:  print('Encountered an attribute error')
        pdb.set_trace()

    def solve_control_gurobi(self):
        """Build and solve a Gurobi optimization problem for the feedback
        conrol (or "control") layer.  This is just like in traj_opt but without
        any contact-related constraints and including dual costs in the
        objective.  This implements Equation (6b) in the paper.

        This step requires that self.r, self.a, and self.gamma are already set
        to numerical values.  It also requires the LCS terms self.As, self.Bs,
        self.Cs, self.ds, self.Gs, self.Hs, self.Js, self.ls, self.Ps, self.Qs,
        self.Rs, and self.Ss are set as well.  These LCS terms, self.a, and
        self.gamma will be of length self.N while self.r will be of length
        self.N+1.

        [1] A. Srikanthan, V. Kumar, N. Matni, "Augmented Langrangian Methods as
        Layered Control Architectures," 2023.
        """
        # TODO: figure out what x_current should be
        x_current = self.x_init

        # Grab a few variables for convenience.
        input_limit = self.traj_opt_object.params.input_limit

        # Build a Gurobi optimization model.
        model = GurobiModelHelper.create_optimization_model(
            "admm_feedback_control_layer", verbose=False)

        # Create variables.
        x = GurobiModelHelper.create_xs(model, lookahead=self.N, n=self.n)
        x_err = GurobiModelHelper.create_x_errs(model, lookahead=self.N,
                                                n=self.n)
        u = GurobiModelHelper.create_us(model, lookahead=self.N, nu=self.m,
                                        input_limit=input_limit)
        lam = GurobiModelHelper.create_lambdas(model, lookahead=self.N,
                                               p=self.p, k=self.k)

        # Build constraints.  Explicitly exclude any contact-related constraints
        # and include the dynamics constraint.
        # TODO: check if the initial condition constraint is needed / more of a
        # need to check what x_current should be.
        model = GurobiModelHelper.add_initial_condition_constr(
            model, xs=x, x_current=x_current)
        model = GurobiModelHelper.add_error_coordinates_constr(
            model, lookahead=self.N, xs=x, x_errs=x_err, x_goal=self.x_goal)
        model = GurobiModelHelper.add_dynamics_constr(
            model, lookahead=self.N, xs=x, us=u, lambdas=lam, A=self.As,
            B=self.Bs, P=self.Ps, C=self.Cs, d=self.ds)

        # Set the model's objective:  use dual error cost to encourage,
        # convergence, and control cost to encourage efficiency.
        obj = 0
        for i in range(self.N):
            input_cost = u[i, :] @ self.R @ u[i, :]
            obj += input_cost

            state_dual_err = x[i, :]@self.Tr - self.r[i, :] + self.vr[i, :]
            obj += (self.rho/2) * state_dual_err @ state_dual_err

            control_dual_err = u[i, :] - self.a[i, :] + self.vu[i, :]
            obj += (self.rho/2) * control_dual_err @ control_dual_err

            comp_dual_err = lam[i, :] - self.gamma[i, :] + self.vgamma[i, :]
            obj += (self.rho/2) * comp_dual_err @ comp_dual_err

        state_dual_err = x[self.N, :]@self.Tr - self.r[self.N, :] + \
            self.vr[self.N, :]
        obj += (self.rho/2) * state_dual_err @ state_dual_err

        model.setObjective(obj, GRB.MINIMIZE)

        # Solve the optimization problem.
        try:
            model.optimize()
            print(f'Control Obj: {model.ObjVal}')

            # Store the solved x, u, lambda variables.
            self.x = x.X
            self.u = u.X
            self.lam = lam.X

            # Avoid the pdb trace for error handling if this succeeded.
            return

        except gp.GurobiError as e:  print(f'Error code {e.errno}: {e}')
        except AttributeError:  print('Encountered an attribute error')
        pdb.set_trace()

    def run_admm(self):
        k = 0
        err = 100
        start = time.time()
        while err >= self.tol:
            k += 1
            # update r
            self.solve_reference_gurobi()
            # update x u
            prev_x = self.x
            prev_u = self.u

            self.solve_control_gurobi()

            # Update the LCS terms based on the new control inputs.
            self.construct_LCS_terms_from_inputs()

            # compute residuals
            sxk = self.rho * (prev_x - self.x).flatten()
            suk = self.rho * (prev_u - self.u).flatten()
            dual_res_norm = np.linalg.norm(np.hstack([sxk, suk]))
            pr_res_norm = np.linalg.norm(self.r - self.x @ self.Tr)

            # update rhok and rescale vk
            if pr_res_norm > 10 * dual_res_norm:
                self.rho = 2 * self.rho
                self.vr = self.vr / 2
                self.vu = self.vu / 2
                self.vgamma = self.vgamma / 2
            elif dual_res_norm > 10 * pr_res_norm:
                self.rho = self.rho / 2
                self.vr = self.vr * 2
                self.vu = self.vu * 2
                self.vgamma = self.vgamma * 2

            # admm_obj.u = np.where(admm_obj.u >= u_max, u_max, admm_obj.u)
            # admm_obj.u = np.where(admm_obj.u <= u_min, u_min, admm_obj.u)
            self.vr = self.vr + self.r - self.x @ self.Tr
            self.vu = self.vu + self.a - self.u
            self.vgamma = self.vgamma + self.lam - self.gamma + self.vgamma

            err = np.trace(
                (self.r - self.x @ self.Tr).T @ (self.r - self.x @ self.Tr)) + np.trace(
                (self.a - self.u).T @ (self.a - self.u)) + np.trace((self.lam - self.gamma + self.vgamma).T @ (self.lam - self.gamma + self.vgamma))

            print(f'ERROR: {err}\n\n=== ', end='')

        end = time.time()

        print("Time", end - start)

    def rollout(self):
        """
        Member function to compute state trajectory
        """
        self.x[0] = self.x_init
        for t in range(self.N - 1):
            self.x[t + 1] = self.dynamics(self.x[t], self.u[t], t)
        return self.x

    def plot_car(self, car_len, i=None, num_plots=None, col='black', col_alpha=1):
        w = car_len / 2
        x = np.zeros(self.n)
        if num_plots == -1:
            x[0:2] = self.x_goal[:2]
            x[2] = 0
        else:
            x[0:2] = self.x[int(i * (self.N+1)/num_plots), :2]
            if self.n == 5:
                x[2] = self.x[int(i * (self.N+1)/num_plots), 4]
            else:
                x[2] = self.x[int(i * (self.N+1)/num_plots), 2]
        x_rl = x[:2] + 0.5 * w * np.array([-np.sin(x[2]), np.cos(x[2])])
        x_rr = x[:2] - 0.5 * w * np.array([-np.sin(x[2]), np.cos(x[2])])
        x_fl = x_rl + car_len * np.array([np.cos(x[2]), np.sin(x[2])])
        x_fr = x_rr + car_len * np.array([np.cos(x[2]), np.sin(x[2])])
        x_plot = np.concatenate((x_rl, x_rr, x_fr, x_fl, x_rl))
        plt.plot(x_plot[0::2], x_plot[1::2], linewidth=2, c=col, alpha=col_alpha)
        plt.scatter(x[0], x[1], marker='.', s=200, c=col, alpha=col_alpha)



def test_car(dynamics_model, T, dt, m, n, goal, x0, u0, u_max, u_min, rho, idx, constr=None, filename="car.png"):
    """
    Function to test LCA on a dynamics simulation of a car
    :param dynamics_model: Specified dynamics model
    :param T: time horizon
    :param dt: discretization time
    :param m: input dimension
    :param n: state dimension
    :param goal: goal
    :param x0: initial state
    :param u0: initial input trajectory
    :param u_max: maximum allowable inputs
    :param rho: initial admm parameter
    :param constr: state constraints
    :return: None
    """
    # Discretize unicycle using rk4 - simple dynamics
    dynamics = rk4(dynamics_model, dt=dt)
    admm_obj = admm_lca(dynamics, T, dt, m, n, x0, u0, goal, rho, idx, constr)

    gain_K, gain_k, x, u, r = run_admm(admm_obj, ctl_prob, u_max, u_min)

    # Plotting figures
    # ==================================================================================================================
    plt.figure()
    # plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')

    car_len = 0.25
    nb_plots = 15
    for i in range(nb_plots):
        admm_obj.plot_car(car_len, i, nb_plots, 'black', 0.1 + 0.9 * i / nb_plots)
    admm_obj.plot_car(car_len, -1, T + 1, 'black')
    admm_obj.plot_car(car_len, -1, -1, 'red')
    x = admm_obj.rollout()
    plt.plot(x[:, 0], x[:, 1], c='black')
    plt.scatter(admm_obj.x_goal[0], admm_obj.x_goal[1], color='r', marker='.', s=200, label="Desired pose")
    if constr:
        plt.plot(np.array([1, 1]), np.array([0, 1.5]), color='b', linewidth=4)
        plt.plot(np.array([1, 4]), np.array([1.5, 1.5]), color='b', linewidth=4)
        plt.plot(np.array([0, 0]), np.array([0, 2.5]), color='b', linewidth=4)
        plt.plot(np.array([0, 4]), np.array([2.5, 2.5]), color='b', linewidth=4)
    plt.legend()
    plt.savefig("../examples/"+filename)
    plt.show()


def main():
    # Define variables for the problem specification
    T = 250
    dt = 0.02
    goal = np.array([3, 2])
    u_max = np.array([1, 4])
    u_min = np.array([0.1, -4])

    # Sample the initial condition from a random normal distribution
    # np.random.seed(0)
    # rng = np.random.default_rng()
    # x0 = rng.standard_normal(3)
    x0 = np.array([0, 0, 0.2])
    u0 = np.zeros(2)
    rho = 50
    m = 2
    n = 3

    A = np.diag(np.ones(n)) + np.diag(np.ones(n-1),1)
    B = np.zeros((n, m))
    B[-m:, :] = np.eye(m)

    # Linear system dynamics
    def linsys(x, u, t):
        return A @ x + B @ u

    test_car(linsys, T, dt, m, n, np.ones(n), x0, u0, u_max, u_min, rho, None, None, filename="linear.png")

    # Unicycle continuous dynamics
    def car(x, u, t):
        return jnp.array([u[0] * jnp.cos(x[2]), u[0] * jnp.sin(x[2]), u[1]])

    constr = ((0, 1), [[0, 1], [1.5, 2.5]])
    for i in range(5):
        goal = goal + i * np.ones(goal.shape)
        test_car(car, T, dt, m, n, goal, x0, u0, u_max, u_min, rho, (0, 1), filename="unicycle.png")
        # test_car(car, T, dt, m, n, goal, x0, u0, u_max, u_min, rho, (0, 1), constr, "unicycle.png")
        x0[0:2] = goal

    # def angle_wrap(theta):
    #     return theta % (2 * np.pi)

    # wrap to [-pi, pi]
    def angle_wrap(theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

    # Dubins car dynamics model
    def complex_car(x, u, t):
        """
        States : x, y, v, psi, omega
        Control: ul, ua - linear and angular acceleration
        Let m = 1, Iz = 1, a = 0.5
        :return:
        """
        px, py, v, psi, w = x
        psi = angle_wrap(psi)
        ul, ua = u
        return jnp.array([v * jnp.cos(psi) - 0.01 * w * jnp.sin(psi), v * jnp.sin(psi) + 0.01 * w * jnp.cos(psi), ul - 0.01 * w ** 2, w, ua])

    n = 5
    x0 = np.array([0, 0, 1, 0.2, 0])
    goal = np.array([4, 4, 0])
    u_max = np.array([0.74, 2])
    u_min = np.array([0, 0])
    constr = ((0, 1), [[0, 4], [1, 3]])
    test_car(complex_car, T, dt, m, n, goal, x0, u0, u_max, u_min, rho, (0, 1),None, "dubins_car.png")
    # test_car(complex_car, T, dt, m, n, goal, x0, u0, u_max, rho, constr, "dubins_car.png")


if __name__ == '__main__':
    main()


