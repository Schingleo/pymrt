""" Generating Synthetic Data
"""
import numpy as np


def sd_generate_state(model, n_steps, n_targets, init_state, birth, death,
                      noise=True):
    """Generate synthetic data on target states

    Provide with a model, number of steps to generate, number of targets,
    initial state of each of the targets and the time of the birth and death
    of each target, the function spins out a calculated locations that each
    of those target is in.

    Args:
        model (:obj:`~pymrt.tracking.models.CVModel`): Constant velocity models
        n_steps (:obj:`int`): The number of time steps to simulate.
        n_targets (:obj:`int`): Number of total targets within the ground truth.
        init_state (:obj:`numpy.ndarray`): 2 dimensional float array
            providing the initial state for all targets. The size of
            ``init_state`` is (``n_targets``, ``self.x_dim``).
        birth (:obj:`numpy.ndarray`): 2 dimensional integer array providing
            the time step of birth of each target. The size of ``birth`` is
            (``n_targets``, 1)
        death (:obj:`numpy.ndarray`): 2 dimensional inter array providing
            the time step of death of each target. The size of ``death`` is
            (``n_targets``, 1)
        noise (:obj:`bool`): If true, error is injected into state updates.

    Returns:
        (:obj:`tuple`): Returns tuple of target states at each step, number
            of targets at each step, track ID of corresponding target states.
            For example, at time 0, truth_states[0] is an array of state
            vectors, and truth_no_tracks[0] contains the target ID of those
            state vector. truth_n_targets[0] contains the number of targets
            in the truth_states[0] and truth_no_tracks[0].
    """
    # Check size of each input parameters
    assert(init_state.shape == (n_targets, model.x_dim))
    assert(birth.shape == (n_targets, 1))
    assert(death.shape == (n_targets, 1))
    assert(birth.max() <= n_steps)
    assert(death.max() <= n_steps)

    # Allocate memory for the state for all targets
    truth_states = [[] for i in range(n_steps)]
    truth_n_targets = np.zeros((n_steps,), np.int)
    truth_no_tracks = [[] for i in range(n_steps)]

    # Generate path for each target
    for t in range(n_targets):
        x_t = init_state[t, :]
        # Add init state to ground truth
        k_start = birth[t, 0] - 1
        truth_states[k_start].append(x_t)
        truth_no_tracks[k_start].append(t)
        truth_n_targets[k_start] += 1
        # Generate states for the track
        for k in range(birth[t, 0], death[t, 0]):
            # When target is alive,
            # 1. Generate next state (no noise), and append it to ground truth
            x_t = model.generate_new_state(x_t, noise=noise)
            truth_states[k].append(x_t)
            # 2. Record which target the state belongs to.
            truth_no_tracks[k].append(t)
            # Increase number of targets at timestep k.
            truth_n_targets[k] += 1

    # Returns all truth
    truth = (truth_states, truth_n_targets, truth_no_tracks, n_targets)
    return truth


def sd_generate_observation(model, truth, clutter_generator=None,
                            p_d=1.0, noise=True):
    """Generate noisy observations based on ground truth

    Args:
        model (:obj:`pymrt.tracking.models.CVModel`): Constant velocity model.
        truth (:obj:`tuple`): A tuple of (``truth_states``, ``truth_n_targets``,
            ``truth_no_tracks``, ``n_targets``), where

            * ``n_targets`` is the number of targets in the ground truth.
            * ``truth_states`` is a :obj:`list` of ``n_targets`` target
              tracks. Each target track is a :obj:`list`
              of states vector.
            * ``truth_n_targets`` is the number of targets at each time step.
            * ``truth_no_tracks`` is a list of ``n_targets`` track label
              lists. Each label list of a target contains the index of the
              track at each time step.

        clutter_generator (:obj:``): Clutter process for generating fake
            false-alarms.
        p_d (:obj:`float`): Detection probability.
        noise (:obj:`bool`): If true, error is injected into observation
            generation.
    """
    truth_states, truth_n_targets, truth_no_tracks, n_targets = truth
    # Get total number of steps
    n_steps = len(truth_states)

    # Allocate memory for the state for all targets
    measure_states = [[] for i in range(n_steps)]

    # Generate Measurements
    for step in range(n_steps):
        # Simulate miss-detection
        if truth_n_targets[step] > 0:
            # Generate detected truth
            idx = np.argwhere(
                np.random.uniform(0, 1, (truth_n_targets[step],)) <= p_d
            ).flatten()
            # For each detected target, generate measurement with noise
            for target in idx:
                measure_states[step].append(
                    model.generate_observation(
                        x_prime=truth_states[step][target],
                        noise=noise
                    )
                )
        # Add False Alarm (Clutter)
        measure_states[step] += clutter_generator.generate()
    return measure_states
