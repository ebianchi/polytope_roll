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
import os
from trajax import optimizers
from trajax.integrators import rk4

import gurobipy as gp
from gurobipy import GRB

import matplotlib.pyplot as plt
from functools import partial
import time
import pdb

from toy_2d.src.optimization_utils import GurobiModelHelper
from toy_2d.src import file_utils, vis_utils


class admm_lca(object):
    """This code is an implementation of ADMM for deriving a layered control
    architecture (LCA) for any nonlinear dynamical system.

    Input arguments
        traj_opt_object: a TwoDTrajectoryOptimization object
        x_init: initial condition, in order [vx, vy, vth, x, y, th].
        x_goal: final state
        rho: ADMM parameter for LCA problem
        tol: ADMM tolerance for terminating iterations
        T: a matrix to penalize complementarity variables via lam^T@T@lam.

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
    def __init__(self, traj_opt_object, x_init, x_goal, rho, tol, T):
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
        self.Q = self.traj_opt_object.params.Q
        self.T = T

        # To eventually fill in the LCS terms adaptively, will need to simulate
        # an LCS.
        self.zero_order_hold_for = 1
        self.lcs = self.traj_opt_object._construct_lcs(
            self.traj_opt_object.params.sim_system,
            self.traj_opt_object.params.traj_opt_dt
        )

        # LCS terms to get filled in about a trajectory of states.  For now,
        # initialize them about a trajectory of all zero inputs starting from
        # the initial state.
        self.construct_LCS_terms_from_inputs()
        self.loop_times = []
        self.plot_trajectories(create=True)

    def construct_LCS_terms_from_inputs(self):
        """Using the stored control inputs in self.u, simulate the LCS with the
        inputs to get a trajectory of states.  Use those states as the
        linearization knot points for the LCS terms."""
        self.traj_opt_object._construct_LCS_terms_from_inputs(
            self.u, self.x_init)
        self.sim_traj = self.traj_opt_object.sim_traj

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
        x_current = self.x_init

        # Grab a few variables for convenience.
        mu_control = self.traj_opt_object.params.sim_system.params.mu_control
        input_limit = self.traj_opt_object.params.input_limit

        # Build a Gurobi optimization model.
        model = GurobiModelHelper.create_optimization_model(
            "admm_trajectory_generation_layer", verbose=False)
        
        # Create variables.
        r = GurobiModelHelper.create_xs(model, lookahead=self.N, n=self.n)
        a = GurobiModelHelper.create_us(model, lookahead=self.N, nu=self.m,
                                        input_limit=input_limit)
        gamma = GurobiModelHelper.create_lambdas(model, lookahead=self.N,
                                                 p=self.p, k=self.k)
        y = GurobiModelHelper.create_ys(model, lookahead=self.N, p=self.p,
                                        k=self.k)

        # Build constraints.  Explicitly exclude the dynamics constraint.
        model = GurobiModelHelper.add_initial_condition_constr(
            model, xs=r, x_current=x_current)
        # model = GurobiModelHelper.add_complementarity_constr(
        #     model, lookahead=self.N, use_big_M=use_big_M, xs=r, us=a,
        #     lambdas=gamma, ys=y, p=self.p, k=self.k)
        model = GurobiModelHelper.add_output_constr(
            model, lookahead=self.N, xs=r, us=a, lambdas=gamma, ys=y, G=self.Gs,
            H=self.Hs, P=self.Ps, J=self.Js, l=self.ls)
        model = GurobiModelHelper.add_friction_cone_constr(
            model, lookahead=self.N, mu_control=mu_control, us=a)
        model = GurobiModelHelper.add_dynamics_constr(
            model, lookahead=self.N, xs=r, us=a, lambdas=gamma, A=self.As,
            B=self.Bs, P=self.Ps, C=self.Cs, d=self.ds)
        
        # Set the model's objective:  use stage cost to encourage smoothness,
        # dual error cost to encourage convergence, and final error to encourage
        # goal progress.
        obj = 0
        for i in range(self.N):
            input_cost = a[i, :] @ self.R @ a[i, :]
            obj += input_cost

            # stage_err = r[i+1, :] - r[i, :]
            # obj += 0.1 * stage_err @ stage_err
            pdb.set_trace()
            goal_err = r[i, :] - self.x_goal
            obj += goal_err @ self.Q @ goal_err

            state_dual_err = self.x[i, :]@self.Tr - r[i, :] + self.vr[i, :]
            obj += (self.rho/2) * state_dual_err @ state_dual_err

            control_dual_err = self.u[i, :] - a[i, :] + self.vu[i, :]
            obj += (self.rho/2) * control_dual_err @ control_dual_err

            comp_dual_err = self.lam[i, :] - gamma[i, :] + self.vgamma[i, :]
            obj += (self.rho/2) * comp_dual_err @ comp_dual_err

            # Add penalty for complementarity variables.
            obj += gamma[i, :] @ self.T @ gamma[i, :]

        state_dual_err = self.x[self.N, :]@self.Tr - r[self.N, :] + \
            self.vr[self.N, :]
        obj += (self.rho/2) * state_dual_err @ state_dual_err

        final_err = r[self.N, :] - self.x_goal
        obj += 1.0 * final_err @ self.Q @ final_err
        model.setObjective(obj, GRB.MINIMIZE)
        
        # Set time limit if desired.
        if self.params.optimization_time_limit is not None:
            model.Params.TimeLimit = self.params.optimization_time_limit
        
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
        mu_control = self.traj_opt_object.params.sim_system.params.mu_control
        input_limit = self.traj_opt_object.params.input_limit
        use_big_M = self.traj_opt_object.params.use_big_M

        # Build a Gurobi optimization model.
        model = GurobiModelHelper.create_optimization_model(
            "admm_feedback_control_layer", verbose=False)

        # Create variables.
        x = GurobiModelHelper.create_xs(model, lookahead=self.N, n=self.n)
        u = GurobiModelHelper.create_us(model, lookahead=self.N, nu=self.m,
                                        input_limit=input_limit)
        lam = GurobiModelHelper.create_lambdas(model, lookahead=self.N,
                                               p=self.p, k=self.k)
        y = GurobiModelHelper.create_ys(model, lookahead=self.N, p=self.p,
                                        k=self.k)

        # Build constraints.  Explicitly exclude any contact-related constraints
        # and include the dynamics constraint.
        # TODO: check if the initial condition constraint is needed / more of a
        # need to check what x_current should be.
        model = GurobiModelHelper.add_initial_condition_constr(
            model, xs=x, x_current=x_current)
        # model = GurobiModelHelper.add_dynamics_constr(
        #     model, lookahead=self.N, xs=x, us=u, lambdas=lam, A=self.As,
        #     B=self.Bs, P=self.Ps, C=self.Cs, d=self.ds)
        model = GurobiModelHelper.add_complementarity_constr(
            model, lookahead=self.N, use_big_M=use_big_M, xs=x, us=u,
            lambdas=lam, ys=y, p=self.p, k=self.k)
        model = GurobiModelHelper.add_output_constr(
            model, lookahead=self.N, xs=x, us=u, lambdas=lam, ys=y, G=self.Gs,
            H=self.Hs, P=self.Ps, J=self.Js, l=self.ls)
        model = GurobiModelHelper.add_friction_cone_constr(
            model, lookahead=self.N, mu_control=mu_control, us=u)

        # Set the model's objective:  use dual error cost to encourage,
        # convergence, and control cost to encourage efficiency.
        obj = 0
        for i in range(self.N):
            stage_err = 0.1 * (x[i+1, :] - x[i, :])
            obj += stage_err @ stage_err

            input_cost = u[i, :] @ self.R @ u[i, :]
            obj += input_cost

            state_dual_err = x[i, :]@self.Tr - self.r[i, :] + self.vr[i, :]
            obj += (self.rho/2) * state_dual_err @ state_dual_err

            control_dual_err = u[i, :] - self.a[i, :] + self.vu[i, :]
            obj += (self.rho/2) * control_dual_err @ control_dual_err

            comp_dual_err = lam[i, :] - self.gamma[i, :] + self.vgamma[i, :]
            obj += (self.rho/2) * comp_dual_err @ comp_dual_err

            # Add penalty for complementarity variables.
            obj += lam[i, :] @ self.T @ lam[i, :]

        state_dual_err = x[self.N, :]@self.Tr - self.r[self.N, :] + \
            self.vr[self.N, :]
        obj += (self.rho/2) * state_dual_err @ state_dual_err

        model.setObjective(obj, GRB.MINIMIZE)
        
        # Set time limit if desired.
        if self.params.optimization_time_limit is not None:
            model.Params.TimeLimit = self.params.optimization_time_limit

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

    def plot_trajectories(self, loop: int = 0, create: bool = False,
                          save: bool = True):
        x_goal = self.x_goal[3]
        y_goal = self.x_goal[4]
        th_goal = self.x_goal[5]

        states = self.x
        xs, ys, ths = states[:, 3], states[:, 4], states[:, 5]

        controls = self.u
        uns, uts = controls[:, 0], controls[:, 1]

        d_states = self.r
        rxs, rys, rths = d_states[:, 3], d_states[:, 4], d_states[:, 5]

        d_controls = self.a
        ans, ats = d_controls[:, 0], d_controls[:, 1]

        s_states = self.sim_traj
        sxs, sys, sths = s_states[:, 3], s_states[:, 4], s_states[:, 5]

        if create:
            file_utils.clear_temp_dir()
            self.plot = {}

            plt.ion()
            self.fig = plt.figure(figsize=(9.16, 10.81))
            self.fig.suptitle(f'Iteration {loop}')

            self.ax1 = self.fig.add_subplot(321)
            self.plot['xs'] = self.ax1.plot(xs, label='Control')
            self.plot['rxs'] = self.ax1.plot(rxs, label='Reference')
            self.plot['sxs'] = self.ax1.plot(sxs, label='Simulated')
            self.ax1.plot([0, self.N], [x_goal, x_goal], 'r--', label='Target')
            self.ax1.set_ylabel('Meters')
            self.ax1.set_title('X direction')
            self.ax1.legend()

            self.ax2 = self.fig.add_subplot(323)
            self.plot['ys'] = self.ax2.plot(ys)
            self.plot['rys'] = self.ax2.plot(rys)
            self.plot['sys'] = self.ax2.plot(sys)
            self.ax2.plot([0, self.N], [y_goal, y_goal], 'r--')
            self.ax2.set_ylabel('Meters')
            self.ax2.set_title('Y direction')

            self.ax3 = self.fig.add_subplot(325)
            self.plot['ths'] = self.ax3.plot(ths)
            self.plot['rths'] = self.ax3.plot(rths)
            self.plot['sths'] = self.ax3.plot(sths)
            self.ax3.plot([0, self.N], [th_goal, th_goal], 'r--')
            self.ax3.set_ylabel('Radians')
            self.ax3.set_title('Z rotation')

            self.ax4 = self.fig.add_subplot(322)
            self.plot['uns'] = self.ax4.plot(uns, label='un')
            self.plot['ans'] = self.ax4.plot(ans, label='an')
            self.ax4.set_ylabel('Newtons')
            self.ax4.set_title('Normal direction')

            self.ax5 = self.fig.add_subplot(324)
            self.plot['uts'] = self.ax5.plot(uts, label='ut')
            self.plot['ats'] = self.ax5.plot(ats, label='at')
            self.ax5.set_ylabel('Newtons')
            self.ax5.set_title('Tangential direction')

            self.ax6 = self.fig.add_subplot(326)
            self.plot['times'] = self.ax6.plot(self.loop_times)
            self.ax6.set_ylabel('Seconds')
            self.ax6.set_title('ADMM Iteration times')
            self.ax6.set_yscale('log')
            if self.params.optimization_time_limit is not None:
                tlim = self.params.optimization_time_limit
                self.plot['time_limit'] = self.ax6.plot(
                    [0, 1], [tlim, tlim], 'r--', label='Limit')
                self.ax6.legend()


        else:
            self.fig.suptitle(f'Iteration {loop}')

            self.plot['xs'][0].set_ydata(xs)
            self.plot['rxs'][0].set_ydata(rxs)
            self.plot['sxs'][0].set_ydata(sxs)

            self.plot['ys'][0].set_ydata(ys)
            self.plot['rys'][0].set_ydata(rys)
            self.plot['sys'][0].set_ydata(sys)

            self.plot['ths'][0].set_ydata(ths)
            self.plot['rths'][0].set_ydata(rths)
            self.plot['sths'][0].set_ydata(sths)

            self.plot['uns'][0].set_ydata(uns)
            self.plot['ans'][0].set_ydata(ans)

            self.plot['uts'][0].set_ydata(uts)
            self.plot['ats'][0].set_ydata(ats)

            self.plot['times'][0].set_xdata(range(len(self.loop_times)))
            self.plot['times'][0].set_ydata(self.loop_times)
            if self.params.optimization_time_limit is not None:
                self.plot['time_limit'][0].set_xdata([0, len(self.loop_times)])

            self.ax1.relim()
            self.ax2.relim()
            self.ax3.relim()
            self.ax4.relim()
            self.ax5.relim()
            self.ax6.relim()
            self.ax1.autoscale(enable=True, axis="y")
            self.ax2.autoscale(enable=True, axis="y")
            self.ax3.autoscale(enable=True, axis="y")
            self.ax4.autoscale(enable=True, axis="y")
            self.ax5.autoscale(enable=True, axis="y")
            self.ax6.autoscale(enable=True, axis="both")

        if save:
            vis_utils.save_temp_fig(self.fig, f'admm_{loop:05d}')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # pdb.set_trace()

    def run_admm(self):
        k = 0
        err = 100
        start = time.time()

        while err >= self.tol:
            loop_start = time.time()
            k += 1
            # update r
            # if k % 1 == 1:
            self.solve_reference_gurobi()
            # update x u
            prev_x = self.x
            prev_u = self.u
            prev_lambda = self.lam

            self.solve_control_gurobi()

            # Update the LCS terms based on the new control inputs.
            self.construct_LCS_terms_from_inputs()

            # compute residuals
            sxk = self.rho * (prev_x - self.x).flatten()
            suk = self.rho * (prev_u - self.u).flatten()
            slambdak = self.rho * (prev_lambda - self.lam).flatten()
            dual_res_norm = np.linalg.norm(np.hstack([sxk, suk, slambdak]))
            rxk = np.linalg.norm(self.r - self.x @ self.Tr)
            auk = np.linalg.norm(self.a - self.u)
            gammalamk = np.linalg.norm(self.gamma - self.lam)
            pr_res_norm = np.linalg.norm(np.hstack([rxk, auk, gammalamk]))

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

            self.vr = self.vr + self.x @ self.Tr - self.r
            self.vu = self.vu + self.u - self.a
            self.vgamma = self.vgamma + self.lam - self.gamma

            err = \
                np.trace(
                    (self.r - self.x @ self.Tr).T @ (self.r - self.x @ self.Tr)
                ) + np.trace(
                    (self.a - self.u).T @ (self.a - self.u)
                ) + np.trace(
                    (self.gamma - self.lam).T @ (self.gamma - self.lam)
                )

            print(f'ERROR: {err}\n\n=== ', end='')
            self.loop_times.append(time.time() - loop_start)
            self.plot_trajectories(loop=k, save=True)

        end = time.time()

        print("Time", end - start)
        pdb.set_trace()
        self.make_gif_from_temp_images()

    def make_gif_from_temp_images(self):
        vis_utils.make_gif_from_temp_images(prefix='admm')


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


