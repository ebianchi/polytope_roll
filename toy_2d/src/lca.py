"""
    SYNOPSIS
    Implementation of ADMM based layered control architecture for unicycle model with input constraints,
    state constraints, and custom reference states x, y. The code has a class structure

    DESCRIPTION
    Uses cvxpy to solve the r-subproblem and iLQR from trajax to solve the (x, u)-subproblem. Dual
    variables are updated according to the rule specified in Boyd's paper. The dynamics is discretized
    using rk4. The code has been tested for various initial states, maximum speed bounds, horizon and
    granularity of discretization.

    AUTHOR
    Anusha Srikanthan <sanusha@seas.upenn.edu>
"""
import cvxpy as cp
import numpy as np
import jax
import jax.numpy as jnp
from trajax import optimizers
from trajax.integrators import rk4
import matplotlib.pyplot as plt
from functools import partial
import time


class admm_lca(object):
    """
    This code is an implementation of ADMM for deriving a layered control architecture (LCA)
    for any nonlinear dynamical system.

    Input arguments
    dynamics: dynamics in functional form with arguments (x, u, t)
    T: time horizon
    dt: discretization time step
    m: input dimension
    n: state dimension
    x0: initial condition
    u0: initial input
    goal: final state
    rho: admm parameter for LCA problem
    idx: indices of states to be used as reference variables
    constr box constraints on states

    Member functions
    solve_reference: function to solve one instance of reference planning layer
    rollout: function to compute state trajectory given initial conditions and input
    """
    def __init__(self, dynamics, T, dt, m, n, x0, u0, goal, rho, idx=None, constr=None):
        self.dynamics = dynamics
        self.dt = dt
        self.x0 = x0
        self.u0 = u0
        self.goal = goal
        self.T = T
        self.n = n
        self.m = m
        self.rho = rho
        if idx is None:
            self.idx = range(self.n)
            self.Tr = np.eye(self.n)
        else:
            self.idx = idx
            self.Tr = np.zeros((self.n, len(self.idx)))
            self.Tr[:len(self.idx), :] = np.eye((len(self.idx)))

        self.r = np.zeros((self.T, len(self.idx)))
        self.a = np.zeros((self.T-1, self.m))
        self.x = np.zeros((self.T, self.n))
        self.u = np.zeros((self.T - 1, self.m))
        self.vr = np.zeros((self.T, len(self.idx)))
        self.vu = np.zeros((self.T - 1, self.m))

        if constr is not None:
            self.constr_idx, self.constr = constr
        else:
            self.constr_idx = None
            self.constr = None

    def solve_reference(self):
        """
        Member function for solving one instance of planning layer sub-problem in the LCA
        """
        r = cp.Variable(self.r.shape)
        a = cp.Variable(self.a.shape)
        stage_err = cp.hstack([(r[t+1] - r[t]) for t in range(self.T - 1)])
        final_err = r[-1] - self.goal

        stage_cost = 0.1 * cp.sum_squares(stage_err)
        final_cost = 1000 * cp.sum_squares(final_err)
        utility_cost = stage_cost + final_cost
        admm_cost = (self.rho / 2) * cp.sum_squares(r - self.x @ self.Tr + self.vr) + (self.rho / 2) * cp.sum_squares(a - self.u + self.vu)

        constr = [r[0] == self.x0[0:len(self.idx)], r[-1] == self.goal]
        # Corridor constraints
        if self.constr:
            # for k in self.constr_idx:
            # Need to make state constraints more general
            constr.append(r[int(self.T/2):, 1] >= self.constr[1][0])
            constr.append(r[int(self.T / 2):, 1] <= self.constr[1][1])
            constr.append(r[:-int(self.T/2), 0] >= self.constr[0][0])
            constr.append(r[:-int(self.T / 2), 0] <= self.constr[0][1])
        ref_prob = cp.Problem(cp.Minimize(utility_cost + admm_cost), constr)
        ref_prob.solve()
        self.r = r.value
        self.a = a.value

    def rollout(self):
        """
        Member function to compute state trajectory
        """
        self.x[0] = self.x0
        for t in range(self.T - 1):
            self.x[t + 1] = self.dynamics(self.x[t], self.u[t], t)
        return self.x

    def plot_car(self, car_len, i=None, num_plots=None, col='black', col_alpha=1):
        w = car_len / 2
        x = np.zeros(self.n)
        if num_plots == -1:
            x[0:2] = self.goal[:2]
            x[2] = 0
        else:
            x[0:2] = self.x[int(i * (self.T+1)/num_plots), :2]
            if self.n == 5:
                x[2] = self.x[int(i * (self.T+1)/num_plots), 4]
            else:
                x[2] = self.x[int(i * (self.T+1)/num_plots), 2]
        x_rl = x[:2] + 0.5 * w * np.array([-np.sin(x[2]), np.cos(x[2])])
        x_rr = x[:2] - 0.5 * w * np.array([-np.sin(x[2]), np.cos(x[2])])
        x_fl = x_rl + car_len * np.array([np.cos(x[2]), np.sin(x[2])])
        x_fr = x_rr + car_len * np.array([np.cos(x[2]), np.sin(x[2])])
        x_plot = np.concatenate((x_rl, x_rr, x_fr, x_fl, x_rl))
        plt.plot(x_plot[0::2], x_plot[1::2], linewidth=2, c=col, alpha=col_alpha)
        plt.scatter(x[0], x[1], marker='.', s=200, c=col, alpha=col_alpha)


# @partial(jax.jit, static_argnums=0)
def ctl_prob(dynamics, x0, r, a, vr, vu, u0, rho, T, Tr):
    def cost(x, u, t):
        # state_err = state_wrap(r[t] - Tr @ x + vr[t])
        state_err = r[t] - x @ Tr + vr[t]
        input_err = a[t] - u + vu[t]
        stage_costs = ((rho / 2) * jnp.dot(state_err, state_err) +
                           (rho / 2) * jnp.dot(input_err, input_err) + 0.01 * jnp.dot(u, u))
        final_costs = rho / 2 * jnp.dot(state_err, state_err)
        return jnp.where(t == T, final_costs, stage_costs)

    X, U, _, _, alpha, lqr_val, _ = optimizers.ilqr(
            cost,
            dynamics,
            x0,
            u0,
            maxiter=10
    )

    return X, U, lqr_val

    # To use constrained ilqr, uncomment this part of the code and comment above
    # def eq_constr(x, u, t):
    #     del u
    #     def goal_constr(x):
    #         err = x[0:2] - r[-1]
    #         return err
    #     return jnp.where(t == T, goal_constr(x), np.zeros(u0.shape[1]))
    #
    # sol = optimizers.constrained_ilqr(cost, dynamics, x0, u0, equality_constraint=eq_constr, maxiter_ilqr=10, maxiter_al=10)

    # return sol[0], sol[1], None


def run_admm(admm_obj, solver=ctl_prob, u_max=None, u_min=None, tol=1e-2):
    """
    Function to run admm iterations until desired tolerance is achieved
    :param admm_obj: class object that contains details of control problem
    :param u_max: maximum input allowed
    :param tol: error tolerance for admm iterations
    :return:
    :param gain_K: gain for converged lqr iteration
    :param gain_k: solution from final lqr iteration
    :param admm_obj.x: final state trajectory
    :param admm_obj.u: final input trajectory
    :param admm_obj.r: final reference trajectory
    """
    T = admm_obj.T
    n = admm_obj.n
    dynamics = admm_obj.dynamics
    x0 = admm_obj.x0
    r = np.array(admm_obj.r)
    a = np.array(admm_obj.a)
    vr = np.array(admm_obj.vr)
    vu = np.array(admm_obj.vu)
    u = np.array(admm_obj.u)
    rho = admm_obj.rho
    if n > r.shape[1]:
        Tr = admm_obj.Tr
    else:
        Tr = np.eye(n)
    X, U, _ = solver(dynamics, x0, r, a, vr, vu, u, rho, T, Tr)
    admm_obj.x = np.array(X)
    admm_obj.u = np.array(U)

    k = 0
    err = 100
    start = time.time()
    while err >= tol:
        k += 1
        # update r
        admm_obj.solve_reference()
        # update x u
        prev_x = admm_obj.x
        prev_u = admm_obj.u

        r = np.array(admm_obj.r)
        a = np.array(admm_obj.a)
        vr = np.array(admm_obj.vr)
        vu = np.array(admm_obj.vu)
        u = np.array(admm_obj.u)
        rho = admm_obj.rho

        if solver is not None:
            X, U, lqr_val = solver(dynamics, x0, r, a, vr, vu, u, rho, T, Tr)
        else:
            X, U, lqr_val = ctl_prob(dynamics, x0, r, a, vr, vu, u, rho, T)
        admm_obj.x = np.array(X)
        admm_obj.u = np.array(U)

        # compute residuals
        sxk = admm_obj.rho * (prev_x - admm_obj.x).flatten()
        suk = admm_obj.rho * (prev_u - admm_obj.u).flatten()
        dual_res_norm = np.linalg.norm(np.hstack([sxk, suk]))
        pr_res_norm = np.linalg.norm(admm_obj.r - admm_obj.x @ admm_obj.Tr)

        # update rhok and rescale vk
        if pr_res_norm > 10 * dual_res_norm:
            admm_obj.rho = 2 * admm_obj.rho
            admm_obj.vr = admm_obj.vr / 2
            admm_obj.vu = admm_obj.vu / 2
        elif dual_res_norm > 10 * pr_res_norm:
            admm_obj.rho = admm_obj.rho / 2
            admm_obj.vr = admm_obj.vr * 2
            admm_obj.vu = admm_obj.vu * 2

        admm_obj.u = np.where(admm_obj.u >= u_max, u_max, admm_obj.u)
        admm_obj.u = np.where(admm_obj.u <= u_min, u_min, admm_obj.u)
        admm_obj.vr = admm_obj.vr + admm_obj.r - admm_obj.x @ admm_obj.Tr
        admm_obj.vu = admm_obj.vu + admm_obj.a - admm_obj.u

        err = np.trace((admm_obj.r - admm_obj.x @ admm_obj.Tr).T @ (admm_obj.r - admm_obj.x @ admm_obj.Tr)) + np.sum(
            (admm_obj.a - admm_obj.u).T @ (admm_obj.a - admm_obj.u))

    end = time.time()

    print("Time", end - start)

    Q, Qq, R, Rr, M, A, B = lqr_val
    gain_K, gain_k, _, _ = optimizers.tvlqr(Q, Qq, R, Rr, M, A, B, np.zeros((T, n)))
    return gain_K, gain_k, admm_obj.x, admm_obj.u, admm_obj.r
    # return None, None, admm_obj.x, admm_obj.u, admm_obj.r


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
    plt.scatter(admm_obj.goal[0], admm_obj.goal[1], color='r', marker='.', s=200, label="Desired pose")
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


