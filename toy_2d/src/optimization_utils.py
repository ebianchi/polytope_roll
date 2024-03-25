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
    def create_xs(model, lookahead=None, n=None):
        return model.addMVar(shape=(lookahead+1, n), lb=-np.inf, ub=np.inf,
                             vtype=GRB.CONTINUOUS, name="xs")

    @staticmethod
    def create_x_errs(model, lookahead=None, n=None):
        return model.addMVar(shape=(lookahead, n), lb=-np.inf, ub=np.inf,
                             vtype=GRB.CONTINUOUS, name="x_errs")

    @staticmethod
    def create_us(model, lookahead=None, nu=None, input_limit=None):
        return model.addMVar(shape=(lookahead, nu), lb=-input_limit,
                             ub=input_limit, vtype=GRB.CONTINUOUS, name="us")

    @staticmethod
    def create_lambdas(model, lookahead=None, p=None, k=None):
        return model.addMVar(shape=(lookahead, p*(k+2)),
                             lb=-np.inf, ub=np.inf,
                             vtype=GRB.CONTINUOUS, name="lambdas")

    @staticmethod
    def create_ys(model, lookahead=None, p=None, k=None):
        return model.addMVar(shape=(lookahead, p*(k+2)),
                             lb=-np.inf, ub=np.inf,
                             vtype=GRB.CONTINUOUS, name="ys")
    
    @staticmethod
    def set_objective(
        model, lookahead=None, Q=None, R=None, S=None, x_errs=None, us=None,
        xs=None
    ):
        obj = 0
        for i in range(lookahead):
            obj += x_errs[i, :] @ Q @ x_errs[i, :]
            obj += us[i, :] @ R @ us[i, :]
            obj += xs[i, :] @ S @ xs[i, :]
        model.setObjective(obj, GRB.MINIMIZE)
        return model
    
    @staticmethod
    def add_initial_condition_constr(model, xs=None, x_current=None):
        model.addConstr(xs[0,:] == x_current, name="initial_condition")
        return model
    
    @staticmethod
    def add_error_coordinates_constr(
        model, lookahead=None, xs=None, x_errs=None, x_goal=None
    ):
        model.addConstrs(
            (x_errs[i,:] == xs[i+1,:] - x_goal \
             for i in range(lookahead)), name="error_coordinates")
        return model
    
    @staticmethod
    def add_dynamics_constr(
        model, lookahead=None, xs=None, us=None, lambdas=None, A=None, B=None,
        P=None, C=None, d=None
    ):
        # First handle the provided LCS terms.
        if type(A) == list:
            assert type(B)==type(P)==type(C)==type(d)==list, 'If given one ' + \
                'listed LCS term, all must be provided as lists.'
            assert lookahead==len(A)==len(B)==len(P)==len(C)==len(d), 'LCS ' + \
                'terms must all be the same length (and equal to lookahead).'
        else:
            assert type(A)==type(B)==type(P)==type(C)==type(d)==np.ndarray, \
                'If LCS terms not lists, they must all be numpy arrays.'
            A = [A] * lookahead
            B = [B] * lookahead
            P = [P] * lookahead
            C = [C] * lookahead
            d = [d] * lookahead

        # Add the constraints.
        model.addConstrs(
            (xs[i+1,:] == \
                A[i]@xs[i,:] + B[i]@P[i]@us[i,:] + C[i]@lambdas[i,:] + d[i] \
             for i in range(lookahead)), name="dynamics")
        return model

    @staticmethod
    def add_complementarity_constr(
        model, lookahead=None, use_big_M=None, xs=None, us=None, lambdas=None,
        ys=None, p=None, k=None
    ):
        model.addConstrs(
            (ys[i,:] >= 0 for i in range(lookahead)), name="comp_1")
        model.addConstrs(
            (lambdas[i,:] >= 0 for i in range(lookahead)), name="comp_2")
        
        # -> Option 1:  Big M method (convex).
        if use_big_M:
            ss = model.addMVar(shape=(lookahead, p*(k+2)),
                               vtype=GRB.BINARY, name="ss")
            model.addConstrs(
                (M1*ss[i,:] >= ys[i,:] for i in range(lookahead)),
                name="big_m_1")
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
    def add_output_constr(
        model, lookahead=None, xs=None, us=None, lambdas=None, ys=None, G=None,
        H=None, P=None, J=None, l=None
    ):
        # First handle the provided LCS terms.
        if type(G) == list:
            assert type(H)==type(P)==type(J)==type(l)==list, 'If given one ' + \
                'listed LCS term, all must be provided as lists.'
            assert lookahead==len(G)==len(H)==len(P)==len(J)==len(l), 'LCS ' + \
                'terms must all be the same length (and equal to lookahead).'
        else:
            assert type(G)==type(H)==type(P)==type(J)==type(l)==np.ndarray, \
                'If LCS terms not lists, they must all be numpy arrays.'
            G = [G] * lookahead
            H = [H] * lookahead
            P = [P] * lookahead
            J = [J] * lookahead
            l = [l] * lookahead

        # Add the constraints.
        model.addConstrs(
            (ys[i,:] == \
                G[i]@xs[i,:] + H[i]@P[i]@us[i,:] + J[i]@lambdas[i,:] + l[i] \
             for i in range(lookahead)), name="output")
        return model

    @staticmethod
    def add_friction_cone_constr(
        model, lookahead=None, mu_control=None, us=None
    ):
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
    
