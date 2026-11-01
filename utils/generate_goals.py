import json
import multiprocessing as mp
import os
import uuid
from collections import defaultdict
from functools import partial

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from .misc import set_seed
import envs  # Import envs to register custom environments


def train_test_goals_darkroom(grid_size, num_train_goals, num_test_goals, seed, start_pos=None, empty_radius=0, train_radial_alpha=None):
    set_seed(seed)
    assert num_train_goals <= grid_size ** 2

    if start_pos is not None:
        start_pos = np.array(start_pos)
    else:
        start_pos = np.array([grid_size // 2, grid_size // 2])

    goals = np.mgrid[0:grid_size, 0:grid_size].reshape(2, -1).T
    manhattan_distances = np.sum(np.abs(goals - start_pos), axis=1)
    goals = goals[manhattan_distances > empty_radius]
    
    if train_radial_alpha is None:
        goals = np.random.permutation(goals)
        train_goals = goals[:num_train_goals]
        test_goals = goals[num_train_goals:num_train_goals+num_test_goals]
    else:
        center_pos = (grid_size // 2, grid_size // 2)
        probs = np.linalg.norm(np.array(goals) - np.array(center_pos), axis=1)
        probs = np.exp(train_radial_alpha * probs)
        probs = probs / probs.sum()
        train_goals = goals[np.random.choice(len(goals), size=num_train_goals, replace=False, p=probs)]
        mask = np.ones(len(goals), dtype=bool)
        for i, goal in enumerate(goals):
            if any(np.array_equal(goal, train_goal) for train_goal in train_goals):
                mask[i] = False
        others = goals[mask]
        others = np.random.permutation(others)
        test_goals = others[:num_test_goals]

    return train_goals, test_goals

