import numpy as np


def process_truth(truth, model):
    """ Process truth for plotting

    Args:
        model (:obj:`~pymrt.tracking.models.CVModel`):
    """
    truth_states, truth_n_targets, truth_no_tracks, n_targets = truth
    length = len(truth_states)
    plot_data = []
    for i in range(length):
        num_targets = truth_n_targets[i]
        observable_states = np.zeros((model.n, num_targets))
        for j in range(num_targets):
            # Generate observation
            observable_states[:, j] = model.generate_observation(
                x_prime=truth_states[i][j], noise=False
            ).flatten()
        plot_data.append(observable_states)
    return plot_data


def process_observation(observation, model):
    """ Process observation for plotting
    """
    length = len(observation)
    plot_data = []
    for i in range(length):
        num_measurements = len(observation[i])
        observable_states = np.zeros((model.n, num_measurements))
        for j in range(num_measurements):
            # Generate observation
            observable_states[:, j] = observation[i][j].flatten()
        plot_data.append(observable_states)
    return plot_data


def process_prediction(prediction, model):
    """ Process observation for plotting
    """
    length = len(prediction)
    plot_data = []
    for i in range(length):
        num_measurements = len(prediction[i])
        observable_states = np.zeros((model.n, num_measurements))
        for j in range(num_measurements):
            # Generate observation
            observable_states[:, j] = model.generate_observation(
                x_prime=prediction[i][j], noise=False
            ).flatten()
        plot_data.append(observable_states)
    return plot_data
