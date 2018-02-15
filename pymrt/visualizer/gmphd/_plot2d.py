import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ._data import process_observation, process_truth, process_prediction


def plot2d_data_preparation(prediction=None,
                            observation=None,
                            truth=None,
                            model=None):
    """ Prepare data for 2D plot

    Args:
        prediction (:obj:`list`): List of predictions.
        observation (:obj:`list`): List of observations.
        truth (:obj:`tuple`): A tuple of truth states (by step)
        model (:obj:`~pymrt.tracking.models.CVModel`):

    Returns:
        (:obj:`dict`): Dictionary containing a plot with lines connecting
            observation and truth in the same frame. The dictionary contains
            the following keys:

            * ``fig``: matplotlib Figure object.
            * ``ax``: matplotlib Axes.
            * ``all``: List of all :obj:`~matplotlib.lines.Line2D` drawn in
                the figure.
            * ``observation``: List of :obj:`~matplotlib.lines.Line2D`
                indexed by frame.
            * ``truth``: List of :obj:`~matplotlib.lines.Line2D`
                indexed by frame.
    """
    fig, (ax) = plt.subplots(1, 1)
    obs_line_list = None
    all_lines = []
    if observation is not None:
        obs_line_list = []
        # Process Observation
        plot_data = process_observation(observation=observation, model=model)
        for i in range(len(plot_data)):
            observation_x = plot_data[i][0, :]
            observation_y = plot_data[i][1, :]
            cur_line_group = ax.plot(observation_x, observation_y, marker='s',
                                     ls='None', color='g', markersize=10)
            obs_line_list.append(cur_line_group)
            all_lines += cur_line_group
    # Process Truth if provided
    truth_line_list = None
    if truth is not None:
        truth_line_list = []
        truth_states, truth_n_targets, truth_no_tracks, n_targets = truth
        plot_data = process_truth(truth=truth, model=model)

        for i in range(len(truth_n_targets)):
            truth_x = plot_data[i][0, :]
            truth_y = plot_data[i][1, :]
            cur_line_group = ax.plot(truth_x, truth_y, marker='o',
                                     ls='None', color='r', markersize=10)
            truth_line_list.append(cur_line_group)
            all_lines += cur_line_group
    # Process prediction if provided
    pred_line_list = None
    if prediction is not None:
        pred_line_list = []
        plot_data = process_prediction(prediction=prediction, model=model)

        for i in range(len(plot_data)):
            prediction_x = plot_data[i][0, :]
            prediction_y = plot_data[i][1, :]
            cur_line_group = ax.plot(prediction_x, prediction_y, marker='^',
                                     ls='None', color='b', markersize=10)
            pred_line_list.append(cur_line_group)
            all_lines += cur_line_group

    return {
        'fig': fig,
        'ax': ax,
        'all': all_lines,
        'observation': obs_line_list,
        'truth': truth_line_list,
        'prediction': pred_line_list
    }


def plot2d_truth(model, truth):
    """ Plot truth on 2D plane

    Args:
        truth (:obj:`tuple`): A tuple of truth states (by step)
        model (:obj:`pymrt.tracking.models.CVModel`):
    """
    plot_data = plot2d_data_preparation(truth=truth, model=model)
    truth_line_list = plot_data['truth']
    all_lines = plot_data['all']
    fig = plot_data['fig']
    num_steps = len(truth_line_list)

    # Animation Update Routine
    def truth_plot_init():
        for line_group in truth_line_list:
            for line in line_group:
                line.set_alpha(0)
        return all_lines

    def truth_plot_update(frame):
        for i in range(frame, max(0, frame - 8), -1):
            for line in truth_line_list[i]:
                line.set_alpha(1 - 0.1 * (frame - i))
        return all_lines

    # Start animation
    ani = FuncAnimation(fig, truth_plot_update,
                        frames=range(num_steps),
                        init_func=truth_plot_init, blit=True, interval=500)
    plt.show()


def plot2d_observation(model, observation, truth=None):
    """ Plot truth on 2D plane

    Args:
        observation (:obj:`list`): List of observations.
        truth (:obj:`tuple`): A tuple of truth states (by step)
        model (:obj:`pymrt.tracking.models.CVModel`):
    """
    plot_data = plot2d_data_preparation(
        model=model, observation=observation, truth=truth
    )
    truth_line_list = plot_data['truth']
    obs_line_list = plot_data['observation']
    all_lines = plot_data['all']
    fig = plot_data['fig']
    num_steps = len(obs_line_list)

    # Animation Update Routine
    def plot_init():
        for line in all_lines:
            line.set_alpha(0)
        return all_lines

    def plot_update(frame):
        for i in range(frame, max(0, frame - 8), -1):
            for line in obs_line_list[i]:
                line.set_alpha(1 - 0.1 * (frame - i))
            if truth is not None:
                for line in truth_line_list[i]:
                    line.set_alpha(1 - 0.1 * (frame - i))
        return all_lines

    # Start animation
    ani = FuncAnimation(fig, plot_update,
                        frames=range(num_steps),
                        init_func=plot_init, blit=True, interval=500)
    plt.show()


def plot2d_gmphd_track(model, grid, gm_s_list=None, gm_list_list=None,
                       observation_list=None, prediction_list=None, truth=None,
                       title=None, contours=4, log_plot=True):
    """ Animate GM-PHD Filter result on 2D plot with matplotlib

    Args:
        model (:obj:`pymrt.tracking.models.CVModel`):
        grid (:obj:`numpy.ndarray`): 2D mesh generated by :func:`numpy.mgrid` or
            :func:`numpy.meshgrid`.
        gm_s_list (:obj:`list`): List of PHD scalars at each time step.
        gm_list_list (:obj:`list`): List of Gaussian Mixtures at each time step.
            If ``gm_s_list`` is None, it is used along with ``grid`` to generate
            the PHD scalar at each time step.
        observation_list (:obj:`list`): List of observations at each time step.
        prediction_list (:obj:`list`): List of predictions at each time step.
        truth (:obj:`tuple`): A tuple of truth states (by step)
        title (:obj:`string`): Plot title.
        contours (:obj:`int`): Number of contour surfaces to draw.
        log_plot (:obj:`bool`): Plot ``gm_s`` in log scale.
    """
    if gm_s_list is None:
        if gm_list_list is None:
            raise ValueError("Must provide 3D sampled GM scalar gm_s or a "
                             "Gaussian Mixture list")
        else:
            print('Sampling PHD in 3D space')
            from ...tracking.utils import gm_calculate
            gm_s_list = []
            i = 0
            for gm_list in gm_list_list:
                sys.stdout.write('calculate gm_scalar for step %d' % i)
                gm_s_list.append(gm_calculate(
                    gm_list=gm_list, grid=grid
                ))
                i += 1

    if title is None:
        title = 'PHD'

    plot_data = plot2d_data_preparation(
        prediction=prediction_list,
        observation=observation_list,
        truth=truth,
        model=model
    )

    fig = plot_data['fig']
    ax = plot_data['ax']
    all_lines = plot_data['all']
    obs_line_list = plot_data['observation']
    truth_line_list = plot_data['truth']
    pred_line_list = plot_data['prediction']

    if log_plot:
        contour_s = np.log(gm_s_list[0] + np.finfo(np.float).tiny)
    else:
        contour_s = gm_s_list[0]

    # Add plot for PHD
    plot_data['contour'] = ax.contourf(
        grid[0], grid[1], contour_s,
        alpha=0.5)

    # Animation Update Routine
    def plot_init():
        for line in all_lines:
            line.set_alpha(0)
        return all_lines + plot_data['contour'].collections

    def plot_update(frame):
        for i in range(frame, max(0, frame - 8), -1):
            for line in obs_line_list[i]:
                line.set_alpha(1 - 0.1 * (frame - i))
            for line in pred_line_list[i]:
                line.set_alpha(1 - 0.1 * (frame - i))
            if truth is not None:
                for line in truth_line_list[i]:
                    line.set_alpha(1 - 0.1 * (frame - i))
        for tp in plot_data['contour'].collections:
            tp.remove()
        if log_plot:
            contour_s = np.log(gm_s_list[frame] + np.finfo(np.float).tiny)
        else:
            contour_s = gm_s_list[frame]
        plot_data['contour'] = ax.contourf(
            grid[0], grid[1], contour_s,
            alpha=0.5)
        return all_lines + plot_data['contour'].collections

    # Start animation
    ani = FuncAnimation(fig, plot_update,
                        frames=range(len(gm_s_list)),
                        init_func=plot_init, blit=True, interval=500)
    plt.show()
