"""This file numerically tests out the linearization of the nonlinear LCP-driven
contact dynamics.  The full LCP dynamics are implemented in two_dim_system.py,
while the linearization is implemented in two_dim_lcs_approximation.py.

This script tests two approximations, both embedded within the LCS object.  The
first uses the possibly nonlinear functions f1(x,u), f2(x), f3(x,u), and f4(x)
of which the LCS approximation takes gradients to form its linearization.  This
approach should, to computer accuracy, match the LCP results.  This approach is
not intended to be used, as it is the same as the full LCP dynamics, but the
numerical comparison is included in this script as a baseline and sanity check
that the implementation is correct.

            x_{k+1} = f_1(x_k, u_k) + f_2(x_k) lambda_k
                y_k = f_3(x_k, u_k) + f_4(x_k) lambda_k
            0 <= lambda_k  PERP  y_k => 0

The second uses the linearization derived from taking gradients of the four
above functions, getting the matrices/vectors as in the following linearization:

            x_{k+1} ~= A x_k + B u_k + C lambda_k + d
                y_k ~= G x_k + H u_k + J lambda_k + l
            0 <= lambda_k  PERP  y_k => 0

This linearization approach is from Aydinoglu et al., 2021.  The output of this
script is a plot of errors over simulated timesteps of the two approaches
compared to the true LCP dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb

from toy_2d.src import file_utils
from toy_2d.src.two_dim_polytope import TwoDimensionalPolytope
from toy_2d.src.two_dim_polytope import TwoDimensionalPolytopeParams
from toy_2d.src.two_dim_system import TwoDimensionalSystem
from toy_2d.src.two_dim_system import TwoDimensionalSystemParams
from toy_2d.src.two_dim_lcs_approximation import TwoDSystemLCSApproximation


# Initial conditions, in order of x, dx, y, dy, theta, dtheta
x0 = np.array([0, 0, 1.5, 0, -1/6 * np.pi, 0])

# Create a polytope.
poly_params = TwoDimensionalPolytopeParams(mass = 1, moment_inertia = 0.01,
    mu_ground = 0.3, vertex_locations = np.array([[1, -1], [1, 1], [-1, 1],
                                                 [-1, -1]]))
polytope = TwoDimensionalPolytope(poly_params)

# Create a system from the polytope, a simulation timestep, and a control
# contact's friction parameter.
system_params = TwoDimensionalSystemParams(dt = 0.002, polytope = polytope,
                                           mu_control = 0.5)
system = TwoDimensionalSystem(system_params)

# Create an LCS approximation from the system.
lcs = TwoDSystemLCSApproximation(system)
pdb.set_trace()

# Rollout with a fixed (body-frame) force at one of the vertices.
states_from_nllcs = x0.reshape(1,6)
outputs_from_nllcs = np.zeros((0, 16))
states_from_lcs = x0.reshape(1,6)
outputs_from_lcs = np.zeros((0, 16))
system.set_initial_state(x0)
for _ in range(1250):
    # First, get the control force.
    state_sys = system.state_history[-1, :]

    # -> Find the third vertex location.
    control_loc = polytope.get_vertex_locations_world(state_sys)[2, :]

    # -> Apply the force at a fixed angle relative to the polytope.
    theta = state_sys[4]
    ang = np.pi + theta
    control_mag = 0.3
    control_vec = control_mag * np.array([-np.cos(ang), -np.sin(ang)])

    # Second, step the true system, using the LCP simulation.
    system.step_dynamics(control_vec, control_loc)
    next_state_sys = system.state_history[-1, :]

    # Third, get the same state from the nonlinear form embedded in the LCS
    # representation.
    state_lcs = lcs._convert_system_state_to_lcs_state(state_sys)
    u_lcs = system.convert_input_to_generalized_coords(state_sys, control_vec,
                                                       control_loc)
    f1 = lcs._get_f_1(state_lcs, u_lcs)
    f2 = lcs._get_f_2(state_lcs)
    f3 = lcs._get_f_3(state_lcs, u_lcs)
    f4 = lcs._get_f_4(state_lcs)

    # -> Get the solved contact forces from the true system.
    lambda_k = system.lambda_history[-1, :]

    # -> Do the nonlinear LCS simulation step and store the results.
    next_state_nllcs = f1 + f2 @ lambda_k
    next_state_from_nllcs = lcs._convert_lcs_state_to_system_state(
                                                            next_state_nllcs)
    states_from_nllcs = np.vstack((states_from_nllcs,
                                   next_state_from_nllcs.reshape(1,6)))
    output_from_nllcs = f3 + f4 @ lambda_k
    outputs_from_nllcs = np.vstack((outputs_from_nllcs,
                                    output_from_nllcs.reshape(1,16)))

    # Fourth, do the linearized LCS approximation, linearizing about the current
    # state and applying the controls and true contact forces.
    # -> Set the initial state and linearize about it.
    lcs.set_initial_state(state_lcs)
    v, q = state_lcs[:3], state_lcs[3:]
    lcs.set_linearization_point(q, v, control_vec, control_loc)

    # -> Step the LCS dynamics.
    lcs.step_lcs_dynamics(control_vec, control_loc, lambda_k)

    # -> Save the LCS simulation results.
    next_state_lcs = lcs.state_history[-1, :]
    next_state_from_lcs = lcs._convert_lcs_state_to_system_state(next_state_lcs)
    states_from_lcs = np.vstack((states_from_lcs,
                                 next_state_from_lcs.reshape(1,6)))
    output_from_lcs = lcs.output_history[-1, :]
    outputs_from_lcs = np.vstack((outputs_from_lcs,
                                  output_from_lcs.reshape(1,16)))

# Compare the dynamics from the 3 methods, against the "true" system dynamics.
states_from_sys = system.state_history
errors_nllcs = np.array([np.linalg.norm(states_from_sys[i] - \
                                        states_from_nllcs[i]) \
                   for i in range(states_from_sys.shape[0])])
print(f'Worst state mismatch between true system and nonlinear LCS ' \
      + f'representation is {max(errors_nllcs)}.')

errors_lcs = np.array([np.linalg.norm(states_from_sys[i]-states_from_lcs[i]) \
                       for i in range(states_from_sys.shape[0])])
print(f'Worst state mismatch between true system and LCS approximation ' \
      + f'is {max(errors_lcs)}.')

plt.ion()
plt.figure()
plt.plot(errors_nllcs, label='Nonlinear LCS Representation vs True')
plt.plot(errors_lcs, label='LCS Approximation vs True')
plt.yscale("log")
plt.ylim(1e-18, 1e-10)
plt.xlabel('Timesteps')
plt.ylabel('State error')
plt.legend()

filename = f'{file_utils.OUT_DIR}/lcs_dynamics_test.png'
plt.savefig(filename)

pdb.set_trace()


# Compare the outputs from the 3 methods, against the "true" system outputs.
outputs_from_sys = system.output_history
errors_nllcs = np.array([np.linalg.norm(outputs_from_sys[i] - \
                                        outputs_from_nllcs[i]) \
                   for i in range(outputs_from_sys.shape[0])])
print(f'Worst output mismatch between true system and nonlinear LCS ' \
      + f'representation is {max(errors_nllcs)}.')

errors_lcs = np.array([np.linalg.norm(outputs_from_sys[i]-outputs_from_lcs[i]) \
                       for i in range(outputs_from_sys.shape[0])])
print(f'Worst output mismatch between true system and LCS approximation ' \
      + f'is {max(errors_lcs)}.')

plt.ion()
plt.figure()
plt.plot(errors_nllcs, label='Nonlinear LCS Representation vs True')
plt.plot(errors_lcs, label='LCS Approximation vs True')
plt.yscale("log")
plt.ylim(1e-8, 1e-2)
plt.xlabel('Timesteps')
plt.ylabel('Output error')
plt.legend()

filename = f'{file_utils.OUT_DIR}/lcs_output_test.png'
plt.savefig(filename)

pdb.set_trace()







