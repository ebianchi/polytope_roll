"""This file contains helper methods for building a Gurobi optimization problem.
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB


# Big M method values.
M1 = 1e3
M2 = 1e3


class GurobiModelHelper:
    """Helper methods for building a Gurobi optimization problem."""

    @staticmethod
    def create_optimization_model(name, verbose=True):
        model = gp.Model(name)
        if not verbose:
            model.setParam('OutputFlag', 0)

        return model

    @staticmethod
    def create_xs(model, lookahead, n):
        return model.addMVar(shape=(lookahead+1, n), lb=-np.inf, ub=np.inf,
                             vtype=GRB.CONTINUOUS, name="xs")

    @staticmethod
    def create_x_errs(model, lookahead, n):
        return model.addMVar(shape=(lookahead, n), lb=-np.inf, ub=np.inf,
                             vtype=GRB.CONTINUOUS, name="x_errs")

    @staticmethod
    def create_us(model, lookahead, nu, input_limit):
        return model.addMVar(shape=(lookahead, nu), lb=-input_limit,
                             ub=input_limit, vtype=GRB.CONTINUOUS, name="us")

    @staticmethod
    def create_lambdas(model, lookahead, p, k):
        return model.addMVar(shape=(lookahead, p*(k+2)),
                             lb=-np.inf, ub=np.inf,
                             vtype=GRB.CONTINUOUS, name="lambdas")

    @staticmethod
    def create_ys(model, lookahead, p, k):
        return model.addMVar(shape=(lookahead, p*(k+2)),
                             lb=-np.inf, ub=np.inf,
                             vtype=GRB.CONTINUOUS, name="ys")
    
    @staticmethod
    def set_objective(model, lookahead, Q, R, S, x_errs, us, xs):
        obj = 0
        for i in range(lookahead):
            obj += x_errs[i, :] @ Q @ x_errs[i, :]
            obj += us[i, :] @ R @ us[i, :]
            obj += xs[i, :] @ S @ xs[i, :]
        model.setObjective(obj, GRB.MINIMIZE)
        return model
    
    @staticmethod
    def add_initial_condition_constr(model, xs, x_current):
        model.addConstr(xs[0,:] == x_current, name="initial_condition")
        return model
    
    @staticmethod
    def add_error_coordinates_constr(model, lookahead, xs, x_errs, x_goal):
        model.addConstrs(
            (x_errs[i,:] == xs[i+1,:] - x_goal \
             for i in range(lookahead)), name="error_coordinates")
        return model
    
    @staticmethod
    def add_dynamics_constr(model, lookahead, xs, us, lambdas, A, B, P, C, d):
        model.addConstrs(
            (xs[i+1,:] == A@xs[i,:] + B@P@us[i,:] + C@lambdas[i,:] + d \
             for i in range(lookahead)), name="dynamics")
        return model

    @staticmethod
    def add_complementarity_constr(model, lookahead, use_big_M, xs, us, lambdas,
                                   ys, p, k, G, H, P, J, l):
        model.addConstrs(
            (ys[i,:] >= 0 for i in range(lookahead)), name="comp_1")
        model.addConstrs(
            (lambdas[i,:] >= 0 for i in range(lookahead)), name="comp_2")
        
        if use_big_M:
            ss = model.addMVar(shape=(lookahead, p*(k+2)),
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
            
        return model

    @staticmethod
    def add_output_constr(model, lookahead, xs, us, lambdas, ys, G, H, P, J, l):
        model.addConstrs(
            (ys[i,:] == G@xs[i,:] + H@P@us[i,:] + J@lambdas[i,:] + l \
             for i in range(lookahead)), name="output")
        return model

    @staticmethod
    def add_friction_cone_constr(model, lookahead, mu_control, us):
        # Note:  the below 3 friction cone constraints expect the input forces
        # to be in the form [f_normal, f_tangent].
        model.addConstrs(
            (us[i,0] >= 0 for i in range(lookahead)), name="friction_cone_1")
        model.addConstrs(
            (-mu_control*us[i,0] <= us[i,1] \
             for i in range(lookahead)), name="friction_cone_2a")
        model.addConstrs(
            (us[i,1] <= mu_control*us[i,0] \
             for i in range(lookahead)), name="friction_cone_2b")
        return model
    
