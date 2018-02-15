"""Test 2D Constant Velocity Model Synthetic Data Generation
"""

import numpy as np
from pymrt.tracking.models import GmPhdCvModel, CubicUniformPoissonClutter
from pymrt.tracking.models import sd_generate_state, sd_generate_observation
from pymrt.visualizer.gmphd import plot2d_observation, plot2d_truth

# Dimensionality
n = 2

# 12 targets, 2D Constant Velocity Model
n_targets = 12
x_init = np.zeros((n_targets, 4))
x_init[ 0, :] = [   0,    0,    0, -10]
x_init[ 1, :] = [ 400, -600,  -10,   5]
x_init[ 2, :] = [-800, -200,   20,  -5]
x_init[ 3, :] = [ 400, -600,   -7,  -4]
x_init[ 4, :] = [ 400, -600, -2.5,  10]
x_init[ 5, :] = [   0,    0,  7.5,  -5]
x_init[ 6, :] = [-800, -200,   12,   7]
x_init[ 7, :] = [-200,  800,   15, -10]
x_init[ 8, :] = [-800, -200,    3,  15]
x_init[ 9, :] = [-200,  800,   -3, -15]
x_init[10, :] = [   0,    0,  -20, -15]
x_init[11, :] = [-200,  800,   15,  -5]

# Total 100 steps
n_steps = 100
birth = np.zeros((n_targets, 1), np.int)
# #target      0  1  2   3   4   5   6   7   8   9  10  11
birth[:, 0] = [1, 1, 1, 20, 20, 20, 40, 40, 60, 60, 80, 80]

death = np.zeros((n_targets, 1), np.int)
death[:, :] = n_steps
death[0, 0] = 70
death[2, 0] = 70

# STD of CV disturbance on state updates
w_cov_std = 1
# STD of CV disturbance on measurements
r_cov_std = 5
# Lambda parameter of clutter process - on average, 10 false-alarms per
# observation
poisson_lambda = 10


def generate_2dcv_model():
    # Create 2D CV Model
    model = GmPhdCvModel(n=n)
    # Initialize Generator for state updates
    model.sigma_w = w_cov_std
    # Initialize Generator for measurement updates
    model.sigma_r = r_cov_std
    return model


def generate_2dcv_states(model):
    return sd_generate_state(
        model=model, n_steps=n_steps, n_targets=n_targets,
        init_state=x_init, birth=birth, death=death,
        noise=True
    )


def generate_2dcv_observation(model, truth):
    return sd_generate_observation(
        model=model, truth=truth, noise=True,
        clutter_generator=CubicUniformPoissonClutter(
            lam=poisson_lambda,
            low=-1000,
            high=1000,
            n=n
        ),
        p_d=1.0
    )


if __name__ == "__main__":
    model = generate_2dcv_model()
    truth = generate_2dcv_states(model)
    plot2d_truth(truth=truth, model=model)
    observation = generate_2dcv_observation(model, truth)
    plot2d_observation(observation=observation, truth=truth, model=model)
