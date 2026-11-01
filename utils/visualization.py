"""Headless DarkRoom trajectory animations used by the environment wrapper."""
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


def draw_grid(grid_size, initial_cost_to_go=None):
    fig, ax = plt.subplots(figsize=(grid_size, grid_size))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(np.arange(0, grid_size, 1))
    ax.set_yticks(np.arange(0, grid_size, 1))
    ax.set_xticks(np.arange(0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(0.5, grid_size, 1), minor=True)
    ax.set_xticklabels(np.arange(0, grid_size, 1), minor=True)
    ax.set_yticklabels(np.arange(0, grid_size, 1), minor=True)
    ax.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.grid(which="major", axis="both", color="k")
    colormesh = ax.pcolormesh(np.zeros((grid_size, grid_size)), vmin=0, vmax=4)
    if initial_cost_to_go is not None:
        ax.set_title(f"Initial Cost-to-Go: {initial_cost_to_go}", fontsize=14, pad=20)
    return fig, ax, colormesh


def animate_traj(size, states, actions, goal, key_pos=None, name="", save_dir=".",
                 high_cost_positions=None, initial_cost_to_go=None):
    fig, ax, colormesh = draw_grid(size, initial_cost_to_go)
    states = np.asarray(states).T
    actions_dict = {0: "O", 1: "↓", 2: "→", 3: "↑", 4: "←"}
    text_obj = None

    def animate(index, goal):
        nonlocal text_obj
        if text_obj is not None:
            text_obj.remove()
        mesh = np.zeros((size, size))
        if high_cost_positions is not None:
            for cost_pos in high_cost_positions:
                mesh[cost_pos[0], cost_pos[1]] = 4
        mesh[int(states[0][index]), int(states[1][index])] = 1
        mesh[goal[0], goal[1]] = 2
        if key_pos is not None:
            mesh[key_pos[0], key_pos[1]] = 3
        colormesh.set_array(mesh.ravel())
        text_obj = ax.text(states[1][index] + 0.4, states[0][index] + 0.4,
                           actions_dict[actions[index]], color="orange", fontsize=20)
        return [colormesh, text_obj]

    frames = animation.FuncAnimation(fig, partial(animate, goal=goal),
                                     frames=states.shape[1], blit=True)
    output = Path(save_dir) / f"basic_animation_{name}.gif"
    try:
        frames.save(str(output), writer=animation.PillowWriter(fps=15))
    finally:
        plt.close(fig)
