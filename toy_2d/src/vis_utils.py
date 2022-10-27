# Visualization utilities

import os
import pdb
import time
import imageio
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from src import file_utils
from src.two_dim_polytope import TwoDimensionalPolytopeParams, \
                                 TwoDimensionalPolytope


"""Make and save a gif of the polytope's state trajectory."""
def animation_gif_polytope(polytope, states, gif_name, dt, controls=None):
    # subsample the states and controls to get 10 samples per second
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
        forces, locs = controls
        if step > 1:
            forces = forces[0::step]
            locs = locs[0::step]

        # scale up the forces so they're more visible
        forces *= 10
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

    gif_file = f'{file_utils.OUT_DIR}/{gif_name}.gif'
    with imageio.get_writer(gif_file, mode='I') as writer:
        for filename in filenames:
            filepath = f'/Users/bibit/Desktop/{filename}'
            image = imageio.imread(filename)
            writer.append_data(image)
    fps = 1./dt
    gif = imageio.mimread(gif_file)
    imageio.mimsave(gif_file, gif, fps=fps)

    for filename in set(filenames):
        os.remove(filename)

    print(f'Saved gif at {gif_file}')















