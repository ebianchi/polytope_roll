"""This file defines a class that represents a receding-horizon model predictive
control problem.  To represent the MPC, it formulates a mixed-integer
optimization problem of the form:

           min         sum_{k=0}^{N-1} (x_k-x_goal)^T Q (x_k-x_goal) \
    x_k, lambda_k, u_k                 + u_k^T R u_k + x_k^T S x_k

        such that   x_{k+1} = A x_k + B u_k + C lambda_k + d
                    M_1 s_k >= G x_k + H u_k + J lambda_k + l >= 0
                    M_2 (1 - s_k) >= lambda_k >= 0
                    sl_k = V x_K
                    s_k in {0, 1}^{p(k+2)}
                    x_0 = x(0)

...where the mixed integer portion is contained in the p*(k+2) vector of binary
variables, s_k.  The matrix S, when used as a norm on the current state vector,
penalizes any slip that is apparent in the state's velocity terms.  The scalars
M_1 and M_2 should be large numbers (used for the big M method).

Another option is the following (non-convex) formulation:

           min         sum_{k=0}^{N-1} (x_k-x_goal)^T Q (x_k-x_goal) \
    x_k, lambda_k, u_k                 + u_k^T R u_k + x_k^T S x_k

        such that   x_{k+1} = A x_k + B u_k + C lambda_k + d
                    y_k = G x_k + H u_k + J lambda_k + l >= 0
                    sl_k = V x_K
                    lambda_k >= 0
                    y_k @ lambda_k == 0        <-- complementarity (non-convex)
                    x_0 = x(0)
"""

from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import pdb
import timeit

import gurobipy as gp
from gurobipy import GRB

from toy_2d.src import file_utils
from toy_2d.src import vis_utils
from toy_2d.src.two_dim_polytope import TwoDimensionalPolytope
from toy_2d.src.two_dim_polytope import TwoDimensionalPolytopeParams
from toy_2d.src.two_dim_system import TwoDSystemForceOnly, TwoDimensionalSystem
from toy_2d.src.two_dim_system import TwoDimensionalSystemParams
from toy_2d.src.two_dim_lcs_approximation import TwoDSystemLCSApproximation



M1 = 1e3
M2 = 1e3



@dataclass
class TrajectoryOptimizationReport:
    costs: np.ndarray = None
    times: np.ndarray = None
    inputs: np.ndarray = None
    states: np.ndarray = None
    controls: np.ndarray = None


@dataclass
class TwoDTrajectoryOptimizationParams:
    sim_system: TwoDimensionalSystem = None
    traj_opt_dt: float = 0.1
    Q: np.ndarray = None
    R: np.ndarray = None
    S_base: np.ndarray = None
    half_freq: bool = False
    use_big_M: bool = False
    lookahead: int = 4
    input_limit: float = 5.


class TwoDTrajectoryOptimization:
    """This class offers the capability of defining an initial state and a goal
    state, then it can perform a receding-horizon model predictive control (MPC)
    to apply controls to the system and guide it to the goal state.

    Some cautionary notes:

        1. This expects the control inputs to be in the form [f_normal,
           f_tangent].  The location where this comes into play is in the
           optimization problem constraints in
           self._set_up_optimization_problem().

        2. This expects states to be in the same format as the LCS state
           vectors, namely [vx, vy, vth, x, y, th].

    Properties:
        params:         2D trajectory optimization parameters, including a
                        simulation system, a trajectory optimization time step
                        (coarser than the simulation system time step), the
                        matrices Q, R, and S_base, if the control frequency
                        should be halved or not, if the optimization problem
                        should use big M or not, and the number of lookahead
                        steps.
        lcs:            A TwoDSystemLCSApproximation object, constructed of a
                        system identical to self.params.sim_system except with a
                        coarser dt of self.params.traj_opt_dt.
        report:         A TrajectoryOptimizationReport that gets filled in with
                        statistics and results after the MPC is performed.
        n_config:       The number of the state configuration (3 for 2D objects,
                        representing x, y, th).
        n_contacts:     The number of contacts of the underlying polytope, i.e.
                        the number of polytope vertices.
        n_friction:     The number of friction polyhedron directions (2 for 2D
                        objects, representing +/-x).
        n_controls:     The number of control inputs of the underlying system.
    """
    params: TwoDTrajectoryOptimizationParams

    def __init__(self, params: TwoDTrajectoryOptimizationParams):
        # Construct an LCS of the simulation system using the coarser time step.
        self.lcs = self._construct_lcs(params.sim_system, params.traj_opt_dt)

        # Store some useful quantities, including the provided parameters.
        polytope = params.sim_system.params.polytope
        self.n_config = polytope.n_config
        self.n_contacts = polytope.n_contacts
        self.n_friction = polytope.n_friction
        self.n_controls = params.R.shape[0]
        self.params = params

        # Initialize an empty report.
        self.report = None

    def _construct_lcs(self, sim_system, dt):
        """Given a system for simulation and a coarser time step, return an LCS
        for performing trajectory optimization on the simulation system.  The
        underlying system is identical to the simulation system except for its
        coarser time step."""

        # Make a(n unlinked) copy of the simulation system and edit the time
        # step.
        traj_opt_system = deepcopy(sim_system)
        traj_opt_system.params.dt = dt

        # Create an LCS approximation from the trajectory optimization system.
        lcs = TwoDSystemLCSApproximation(traj_opt_system)

        return lcs

    def run_trajectory_optimization(self, x_init, x_goal, loops):
        """Given an initial state, a goal state, and a number of loops, perform
        trajectory optimization."""

        # Keep track of the solved control inputs, costs, and loop times.
        inputs = np.zeros((0, self.n_controls))
        costs, times = np.array([]), np.array([])

        # Set the goal and initial states of the system.
        self.x_goal = x_goal
        self.params.sim_system.set_initial_state(
            self.lcs._convert_lcs_state_to_system_state(x_init))

        # Set the running state to the current state.
        x_curr = x_init

        # Begin loop.
        for _ in range(loops):
            # Start timer.
            start_time = timeit.default_timer()

            # Build and solve the optimization problem model.
            us, cost = self._set_up_and_solve_optimization_problem(x_curr)

            # Apply the controls on the simulation system.
            control_input = self._simulate_with_controls(us)

            # Get the latest state to use for the next linearization.
            sys_state = self.params.sim_system.state_history[-1, :]
            x_curr = self.lcs._convert_system_state_to_lcs_state(sys_state)

            # Save the control input and objective cost.
            inputs = np.vstack((inputs, control_input))
            costs = np.hstack((costs, cost))

            # Record the loop time.
            loop_time = timeit.default_timer() - start_time
            times = np.hstack((times, loop_time))

        # Store the statistics and results of the trajectory optimization in the
        # form of a report.
        report = TrajectoryOptimizationReport(costs=costs, times=times,
            inputs=inputs, states = self.params.sim_system.state_history,
            controls = self.params.sim_system.control_history)

        self.report = report

    def _simulate_with_controls(self, us):
        """Given the control array from solving the optimization problem, use
        the first control input to apply to the real system, doing a zero-order
        hold with the trajectory optimization's output.  Return the control
        inputs used."""

        # For convenience...
        dt_traj_opt = self.params.traj_opt_dt
        dt_sim = self.params.sim_system.params.dt
        t_multiple = int(dt_traj_opt / dt_sim)
        mu_control = self.params.sim_system.params.mu_control

        # Clip the control input to enforce it is within feasible bounds (slight
        # infeasibility is possible due to numerics).
        control_input = us[0]
        control_input = np.clip(control_input,
                                [0, -mu_control*max(0, control_input[0])],
                                [np.inf, mu_control*max(0, control_input[0])])
        for _ in range(t_multiple):
            self.params.sim_system.step_dynamics(control_input)

        # If we want to apply control at half the frequency, then apply the
        # second solved control input too.
        if self.params.half_freq:
            inputs = np.vstack((inputs, control_input))

            control_input2 = us[1]
            control_input2 = np.clip(control_input2,
                                [0, -mu_control*max(0, control_input2[0])],
                                [np.inf, mu_control*max(0, control_input2[0])])
            for _ in range(t_multiple):
                self.params.sim_system.step_dynamics(control_input2)

            control_input = np.vstack((control_input, control_input2))

        return control_input

    def _get_lcs_plus_terms(self, x_current):
        """Given a current state, return all of the LCS and control vectors and
        matrices."""

        # For convenience, grab the number of contacts, control matrices, and
        # the polytope.
        nu = self.n_controls
        p = self.n_contacts
        Q = self.params.Q
        R = self.params.R
        S_base = self.params.S_base
        polytope = self.params.sim_system.params.polytope

        # Set the LCS initial state and linearization point, both to the current
        # state.
        self.lcs.set_initial_state(x_current)
        v, q = x_current[:3], x_current[3:]
        controls = np.zeros((nu))
        self.lcs.set_linearization_point(q, v, controls)

        # Get the relevant LCS matrices.
        A, B, C, d, G, H, J, l, P = self.lcs.get_lcs_terms()

        # To build the V matrix, get the angle between the ground and the
        # pivoting vertex as well as the index of that vertex.
        theta_v, pivot_index = polytope.get_theta_v_and_pivot_index_from_theta(
                                                                        q[2])

        # The lever angle of the center of mass is the ground angle plus the
        # angle between the convex hull face and the line between the pivot and
        # the center of mass.
        lever_angle = theta_v + polytope.gammas[(pivot_index-1) % p]

        # The lever arm is the distance between the pivot and center of mass.
        radii, _ = polytope.get_vertex_radii_angles()
        lever_arm = radii[pivot_index]

        # Build the matrix V such that V @ x yields the "slip vector".
        V = np.array([[1., 0., lever_arm*np.sin(lever_angle), 0., 0., 0.],
                      [0., 1., -lever_arm*np.cos(lever_angle), 0., 0., 0.]])

        # Get S to penalize the amount of slip, when S is used as the norm of the
        # current state vector.
        S = V.T @ S_base @ V

        # Return all the LCS and control vectors/matrices.
        return A, B, C, d, G, H, J, l, P, Q, R, S

    def _set_up_and_solve_optimization_problem(self, x_current):
        """Set up and solve a Gurobi optimization problem, returning the control
        inputs and objective cost."""

        # For convenience, get the configuration, contacts, friction, and
        # controls size, the number of lookahead steps, and control friction.
        n = self.n_config
        p = self.n_contacts
        k_friction = self.n_friction
        nu = self.n_controls
        lookahead = self.params.lookahead
        mu_control = self.params.sim_system.params.mu_control
        input_limit = self.params.input_limit

        # Get the relevant LCS and control matrices.
        A, B, C, d, G, H, J, l, P, Q, R, S = self._get_lcs_plus_terms(x_current)

        # Create a new gurobi model.
        model = gp.Model("traj_opt")

        # Mute the model (may want to comment this out for debugging).
        model.setParam('OutputFlag', 0)

        # Create variables.
        xs = model.addMVar(shape=(lookahead+1, 2*n), lb=-np.inf, ub=np.inf,
                            vtype=GRB.CONTINUOUS, name="xs")
        x_errs = model.addMVar(shape=(lookahead, 2*n), lb=-np.inf, ub=np.inf,
                                vtype=GRB.CONTINUOUS, name="x_errs")
        us = model.addMVar(shape=(lookahead, nu), lb=-input_limit,
                           ub=input_limit, vtype=GRB.CONTINUOUS, name="us")
        lambdas = model.addMVar(shape=(lookahead, p*(k_friction+2)),
                                lb=-np.inf, ub=np.inf,
                                vtype=GRB.CONTINUOUS, name="lambdas")
        ys = model.addMVar(shape=(lookahead, p*(k_friction+2)),
                           lb=-np.inf, ub=np.inf,
                           vtype=GRB.CONTINUOUS, name="ys")

        # Set objective:  penalize distance to goal, control input, and slip
        # measurement.
        obj = 0
        for i in range(lookahead):
            obj += x_errs[i, :] @ Q @ x_errs[i, :]
            obj += us[i, :] @ R @ us[i, :]
            obj += xs[i, :] @ S @ xs[i, :]
        model.setObjective(obj, GRB.MINIMIZE)

        # Build constraints.
        # -> Dynamics, initial condition, error coordinates, output, etc.
        model.addConstr(xs[0,:] == x_current, name="initial_condition")
        model.addConstrs(
            (xs[i+1,:] == A@xs[i,:] + B@P@us[i,:] + C@lambdas[i,:] + d \
             for i in range(lookahead)), name="dynamics")
        model.addConstrs(
            (x_errs[i,:] == xs[i+1,:] - self.x_goal \
             for i in range(lookahead)), name="error_coordinates")
        model.addConstrs(
            (ys[i,:] >= 0 for i in range(lookahead)), name="comp_1")
        model.addConstrs(
            (lambdas[i,:] >= 0 for i in range(lookahead)), name="comp_2")
        model.addConstrs(
            (ys[i,:] == G@xs[i,:] + H@P@us[i,:] + J@lambdas[i,:] + l \
             for i in range(lookahead)), name="output")
        # -> Note:  the below 3 friction cone constraints expect the input
        # forces to be in the form [f_normal, f_tangent].
        model.addConstrs(
            (us[i,0] >= 0 for i in range(lookahead)), name="friction_cone_1")
        model.addConstrs(
            (-mu_control*us[i,0] <= us[i,1] \
             for i in range(lookahead)), name="friction_cone_2a")
        model.addConstrs(
            (us[i,1] <= mu_control*us[i,0] \
             for i in range(lookahead)), name="friction_cone_2b")

        # -> Option 1:  Big M method (convex).
        if self.params.use_big_M:
            ss = model.addMVar(shape=(lookahead, p*(k_friction+2)),
                               vtype=GRB.BINARY, name="ss")
            model.addConstrs(
                (M1*ss[i,:] >= G@xs[i,:] + H@P@us[i,:] + J@lambdas[i,:] + l \
                 for i in range(lookahead)), name="big_m_1")
            model.addConstrs(
                (M2*(1-ss[i,:]) >= lambdas[i,:] for i in range(lookahead)),
                name="big_m_2")

        # -> Option 2:  Complementarity constraint (non-convex).
        else:
            model.params.NonConvex = 2
            model.addConstrs(
                (lambdas[i,:] @ ys[i,:] == 0 for i in range(lookahead)),
                name="complementarity")

        # Solve the optimization problem, returning the control input and cost.
        try:
            model.optimize()
            print('Obj: %g' % model.ObjVal)
            return us.X, model.ObjVal

        except gp.GurobiError as e:  print(f'Error code {e.errno}: {e}')
        except AttributeError:  print('Encountered an attribute error')
        pdb.set_trace()

    def generate_visuals_from_latest_report(self, file_title=None, title=None,
                                            save=False):
        """Given the results of a previously run receding horizon model
        predictive control experiment, generate helpful visuals of the results,
        saving them if desired."""

        # Ensure there is a report to visualize.
        assert self.report is not None

        # Grab all the relevant parameters in the latest report.
        costs = self.report.costs
        times = self.report.times
        states = self.report.states
        controls = self.report.controls

        # For convenience, grab some necessary values out of the params.
        polytope = self.params.sim_system.params.polytope
        dt_sim = self.params.sim_system.params.dt

        # Plot the results in the trajectory optimization report.
        vis_utils.traj_plot(states, controls, file_title, save=save,
                            costs=costs, times=times, title=title)

        # Generate a gif of the simulated rollout.
        vis_utils.animation_gif_polytope(polytope, states, file_title, dt_sim,
                                         controls=controls, save=save,
                                         title=title)

