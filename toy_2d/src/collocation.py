import gurobipy as gp
from gurobipy import GRB
import numpy as np
import copy
import timeit

def add_collocation_constr(model,x,u,N,poly_instance,timesteps,polytope):
    # TODO - Add collocation constraints, calls add_constraint_one_spline internally
    for i in range(N-1):
        x_i     =   x[i]
        x_ip1   =   x[i+1]
        u_i     =   u[i]
        u_ip1   =   u[i+1]

        # Adds collocation constraint for each spline
        add_constraint_one_spline(model, x_i, u_i, x_ip1, u_ip1, poly_instance,timesteps[i+1] - timesteps[i],polytope)


def add_constraint_one_spline(model, x_i, u_i, x_ip1, u_ip1, poly_instance,dt,polytope):
    '''
    Helper function to add collocation constraints for one particular cubic spline
    x_i      -   x_k
    x_ip1   -   x_{k+1}
    Similar for u    
    '''
    # Assuming u_mid = (uk + u{k+1})/2 -- need to check if this assumption is okay
    u_mid =   {0: (u_ip1[0] + u_i[0])/2, 1: (u_ip1[1] + u_i[1])/2}

    s0    =   x_i 
    s1    =   rollout_dynamics(poly_instance,x_i,u_i,polytope)
    s2    =   (1/dt**2) * (3  *   (x_ip1 - x_i)  -  dt  * (2*rollout_dynamics(poly_instance,x_i,u_i,polytope) + rollout_dynamics(poly_instance,x_ip1,u_ip1,polytope) )   )
    s3    =   (1/dt**3) * (2*(x_i - x_ip1) + dt * ( rollout_dynamics(poly_instance,x_i,u_i,polytope) + rollout_dynamics(poly_instance,x_ip1,u_ip1,polytope) ) )

    # returns the cubic spline
    def get_spline(s0,s1,s2,s3,dt):
        return s0 + s1*dt + s2*dt**2 + s3* dt**3

    # returns the first derivative
    def get_der(s0,s1,s2,s3,dt):
        return s1 + 2*s2*dt + 3*s3*dt**2

    # Constraining dynamics at mid pt of spline
    h_i   =   - rollout_dynamics(poly_instance,get_spline(s0,s1,s2,s3,dt/2),u_mid,polytope) + get_der(s0,s1,s2,s3,dt/2)
    model.addConstr(h_i ==  0.0)


def rollout_dynamics(poly_instance,xi,ui,polytope):
    '''
    Returns x_{k+1} (where x{k+1} =   f(xk,uk)) 
    '''
    # Assuming that control can only be acted on one corner of cube (top left corner in this case)
    # TODO non-linearities currently not supported in Gurobi -- need to debug this 
    control_loc                     =   polytope.get_vertex_locations_world(xi)[2, :]
    
    theta                           =   state[4]
    ang                             =   ui[1] + theta
    control_mag                     =   ui[0]

    control_vec                     =   control_mag * np.array([-np.cos(ang), -np.sin(ang)])
    poly_instance.set_initial_state(xi)
    poly_instance.step_dynamics(control_vec, control_loc)

    return poly_instance.state_history[-1]

def find_rolling_trajectory(N, init_state, final_state, tf, system,polytope):
    """
    Assumption: Force only acts on the third vertex
    Args:
    N               -   number of knot points
    init_state      -   initial state of the cube
    final_state     -   required final state of the cube (assuming that its rotated by 
                                                            90 degrees clockwise)
    tf              -   required final rotated time
    system          -   object of class two_dim_system. Used to get dynamics
    
    Outputs:
    u[n]            -   (N dims) set of control inputs for achieving the required roll
                        each 2 dimensional (force,theta)
    """
    model           =   gp.Model()
    
    # Creating a copy of the defined system parameters, because python functions are call by ref
    poly_instance   =   copy.deepcopy(system)

    # Creating decision variables, in this case, state and control for N knot points
    # x               =   np.zeros((N,init_state.shape[0]),dtype='object')
    # u               =   np.zeros((N,2),dtype='object')

    x               =   np.zeros((N,),dtype='object')
    u               =   np.zeros((N,),dtype='object')
    for i in range(N):
        x[i]        =  model.addVars(init_state.shape[0],lb = [0.0]*init_state.shape[0],ub =[float('inf'),float('inf'),\
                                float('inf'),float('inf'),2*np.pi,float('inf')],name = "x_"+str(i))

        u[i]        =  model.addVars(2,lb = 0.0,ub =[float('inf'),np.pi],name = "u_"+str(i))
    
    # Creating equally spaced timesteps for collocation
    t0              =   0.0
    timesteps       =   np.linspace(t0,tf,N)

    x0              =   x[0]
    xf              =   x[-1]

    # Adding initial position constraint and final position constraint
    for i in range(init_state.shape[0]):
        model.addConstr(x0[i]          ==   init_state[i])
        model.addConstr(xf[i]          ==   final_state[i])

    # Adding collocation constraints
    add_collocation_constr(model,x,u,N,poly_instance,timesteps,polytope)

    # Cost function -> Quadratic cost 
    quad_expr                     = 0
    for i in range(N-1):
        dt                        = timesteps[i+1]  -   timesteps[i]
        error_i                   = x[i]            -   final_state
        error_ip1                 = x[i+1]          -   final_state

        quad_expr                 = 0.5*dt*(error_i.T@ error_i + error_ip1.T@ error_ip1)

    # Setting objective
    model.setObjective(quad_expr,GRB.MINIMIZE)

    # Solving the set up above problem - taken from Alp's ADMM code
    starttime                     = timeit.default_timer()
    model.optimize()
    t_diff                        = timeit.default_timer() - starttime

    # Getting the solution
    v                             = model.getVars()