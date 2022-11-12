# Visualization utilities

import os
import pdb
import time
import imageio
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from toy_2d.src import file_utils
from toy_2d.src.two_dim_polytope import TwoDimensionalPolytopeParams, \
                                 TwoDimensionalPolytope


FORCE_SCALING = 1.  # Scaling factor for viewing forces.

"""Make and save a gif of the polytope's state trajectory."""
def animation_gif_polytope(polytope, states, gif_name, dt, controls=None,
                           save=False, force_scale=1.):
    # Subsample the states and controls to get 10 samples per second of
    # simulated data.
    step = int(0.1/dt)
    if step > 1:
        states = states[0::step]

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    xs, ys = states[:, 0], states[:, 2]
    plt.xlim(min(xs)-4, max(xs)+4)   
    plt.ylim(-1, max(ys)+4)
    ax.set_aspect('equal', 'box')

    # Plot the ground, polytope, and corners.
    ground = plt.fill_between(x=np.arange(min(xs)-5,max(xs)+5,1),
                              y1=0, y2=-1, color='b', alpha=0.2)
    init_state = states[0, :]
    init_corners = polytope.get_vertex_locations_world(init_state)
    poly = Polygon(init_corners, closed=True)
    ax.add_patch(poly)
    corner_dots, = ax.plot(init_corners[:, 0], init_corners[:, 1], 'ro',
                           markersize=8, linewidth=0)

    # If controls are given, plot them too.
    if controls is not None:
        if type(controls) == tuple:
            forces, locs = controls
        else:
            forces, locs = controls[:, :2], controls[:, 2:]
        if step > 1:
            forces = forces[0::step]
            locs = locs[0::step]

        # scale up the forces so they're more visible
        forces = forces.copy() * force_scale
        ctrl = ax.arrow(locs[0,0]-forces[0,0], locs[0,1]-forces[0,1],
                        forces[0,0], forces[0,1],
                        width=0.1, length_includes_head=True)

    filename = f'{file_utils.TEMP_DIR}/0.png'
    plt.savefig(filename)
    filenames = [filename]

    for i in range(states.shape[0]):
        new_state = states[i, :]
        new_corners = polytope.get_vertex_locations_world(new_state)
        corner_dots.set_data((new_corners[:, 0], new_corners[:, 1]))

        poly.set(xy=new_corners)

        if (controls is not None) and (i+1 < states.shape[0]):
            # update arrow
            ctrl.remove()
            ctrl = ax.arrow(locs[i,0]-forces[i,0], locs[i,1]-forces[i,1],
                            forces[i,0], forces[i,1],
                            width=0.1, length_includes_head=True)

        fig.canvas.draw()
        fig.canvas.flush_events()

        filename = f'{file_utils.TEMP_DIR}/{i+1}.png'
        plt.savefig(filename)
        filenames.append(filename)

    if save:
        gif_file = f'{file_utils.OUT_DIR}/{gif_name}.gif'
        with imageio.get_writer(gif_file, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        fps = 1./dt
        gif = imageio.mimread(gif_file)
        imageio.mimsave(gif_file, gif, fps=fps)

        print(f'Saved gif at {gif_file}')

    for filename in set(filenames):
        os.remove(filename)

"""Make and save plots of a system's state and control history."""
def traj_plot(states, controls, plot_name, save=False, costs=None, times=None,
              title=""):
    xs, ys, ths = states[:, 0], states[:, 2], states[:, 4]
    vxs, vys, vths = states[:, 1], states[:, 3], states[:, 5]

    fx, fy = controls[:, 0], controls[:, 1]
    force_mag = np.linalg.norm(controls[:, :2], axis=1)

    plt.ion()
    fig = plt.figure(figsize=(8,8))

    ax1 = fig.add_subplot(321)
    ax1.plot(xs, label='x')
    ax1.plot(ys, label='y')
    ax1.plot(ths, label='angle')
    ax1.set_ylabel('Meters or Radians')
    ax1.legend()

    ax2 = fig.add_subplot(323)
    ax2.plot(vxs, label='v_x')
    ax2.plot(vys, label='v_y')
    ax2.plot(vths, label='v_angle')
    ax2.set_ylabel('Velocity')
    ax2.legend()

    ax3 = fig.add_subplot(325)
    ax3.plot(fx, label='f_x')
    ax3.plot(fy, label='f_y')
    ax3.plot(force_mag, label='force_mag')
    ax3.set_ylabel('Force')
    ax3.legend()
    ax3.set_xlabel('Timesteps')

    if costs is not None:
        ax4 = fig.add_subplot(322)
        ax4.plot(costs)
        ax4.set_ylabel('Optimization Cost')
        ax4.yaxis.set_label_position("right")
        ax4.yaxis.tick_right()
        ax4.set_yscale("log")

    if times is not None:
        total_time = sum(times)

        ax5 = fig.add_subplot(324)
        ax5.plot(times, label=f'Total Time: {total_time:.2f}s')
        ax5.set_ylabel('Loop Time (seconds)')
        ax5.yaxis.set_label_position("right")
        ax5.yaxis.tick_right()
        ax5.set_yscale("log")
        ax5.set_ylim(3e-2, 2e0)
        ax5.grid(which='both', axis='y')
        ax5.legend()
        ax5.set_xlabel('Loops')

    fig.suptitle(title)

    if save:
        filename = f'{file_utils.OUT_DIR}/{plot_name}.png'
        plt.savefig(filename)


