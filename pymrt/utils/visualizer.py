""" pymrt.utils.visualizer

This package provides visualization functions for display intermediate
numeric data for multi-resident tracking and activity recognition in CASAS
smart homes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from mayavi import mlab


def animate_2d_mrt_observations_onsite(site, observations, sensors=None):
    """Animate sensor event observations on CASAS site floorplan.
    The animation is developed using matplotlib.

    Args:
        site (:obj:`CASASSite`): CASAS site class containing floorplan and
            sensor information.
        observations (:obj:`list`): List of observations. Each observation is
            a :obj:`list` of index or target name of sensors that are ON,
            PRESENT or OPEN (depends on what type of sensor it is) at each
            time step.
        sensors (:obj:`list`): List of sensors represented as dictionary.
    """
    drawing_data = site._prepare_floorplan()
    # Prepare aux data structure
    if sensors is not None:
        sensor_list = set(sensor['name'] for sensor in sensors)
    else:
        sensor_list = set(sensor['name'] for sensor in site.sensor_list)
    fig, (ax) = plt.subplots(1, 1)
    fig.set_size_inches(18, 18)
    ax.imshow(drawing_data['img'])
    active_patch_list = []
    # Draw Sensor block patches
    for key, patch in drawing_data['sensor_boxes'].items():
        if key in sensor_list:
            patch.set_alpha(0.3)
            ax.add_patch(patch)
            active_patch_list.append(patch)
    # Draw Sensor name
    for key, text in drawing_data['sensor_texts'].items():
        ax.text(*text, color='black',
                backgroundcolor=mcolors.colorConverter.to_rgba('#D3D3D3', 0.7),
                horizontalalignment='center', verticalalignment='top',
                zorder=3)

    def observation_init():
        return active_patch_list

    def observation_update(frame):
        observation = observations[frame]
        activated_sensors = [
            sensor if sensor is str else sensors[sensor]['name'] for sensor in
            observation]
        for key, patch in drawing_data['sensor_boxes'].items():
            patch.set_alpha(0.3)
            if key in activated_sensors:
                patch.set_alpha(1.)
        return active_patch_list

    # Start animation
    ani = FuncAnimation(fig, observation_update,
                        frames=range(len(observations)),
                        init_func=observation_init, blit=True, interval=500)
    plt.show()


def animate_2d_mrt_sequence_onsite(site, sequence, sensors=None):
    """Animate sensor event sequence (only starting part: ON, PRESENT,
    or OPEN) on CASAS site floorplan.
    The animation is developed using matplotlib.

    Args:
        site (:obj:`CASASSite`): CASAS site class containing floorplan and
            sensor information.
        sequence (:obj:`list`): List of sensor activation sequence.
        sensors (:obj:`list`): List of sensors represented as dictionary.
    """
    drawing_data = site._prepare_floorplan()
    # Prepare aux data structure
    if sensors is not None:
        sensor_list = set(sensor['name'] for sensor in sensors)
    else:
        sensor_list = set(sensor['name'] for sensor in site.sensor_list)
    fig, (ax) = plt.subplots(1, 1)
    fig.set_size_inches(18, 18)
    ax.imshow(drawing_data['img'])
    active_patch_list = []
    # Draw Sensor block patches
    for key, patch in drawing_data['sensor_boxes'].items():
        if key in sensor_list:
            patch.set_alpha(0.3)
            ax.add_patch(patch)
            active_patch_list.append(patch)
    # Draw Sensor name
    for key, text in drawing_data['sensor_texts'].items():
        ax.text(*text, color='black',
                backgroundcolor=mcolors.colorConverter.to_rgba('#D3D3D3', 0.7),
                horizontalalignment='center', verticalalignment='top',
                zorder=3)

    def sequence_init():
        return active_patch_list

    def sequence_update(frame):
        sensor = sequence[frame]
        sensor_name = sensor if sensor is str else sensors[sensor]['name']
        for key, patch in drawing_data['sensor_boxes'].items():
            patch.set_alpha(0.3)
            if key == sensor_name:
                patch.set_alpha(1.)
        return active_patch_list

    # Start animation
    ani = FuncAnimation(fig, sequence_update, frames=range(len(sequence)),
                        init_func=sequence_init, blit=True, interval=500)
    plt.show()


def animate_3d_mrt_observations(dataset, embedding, observations):
    """Animate Observation with sensor vector embedding in 3D space using mayavi

    Args:
        dataset (:obj:`CASASDataset`): CASAS dataset class containing
            sensor information.
        embedding (:obj:`numpy.ndarray`): A 2D array of size
            (#sensors, #vec_ndim) containing vector embedding of each sensor.
        observations (:obj:`list`): List of observations. Each observation is
            a :obj:`list` of index or target name of sensors that are ON,
            PRESENT or OPEN (depends on what type of sensor it is) at each
            time step.
    """
    figure = mlab.figure(dataset.data_dict['name'])
    figure.scene.disable_render = True
    points = mlab.points3d(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                           scale_factor=0.03)
    for i, x in enumerate(embedding):
        mlab.text3d(x[0], x[1], x[2], dataset.sensor_list[i]['name'],
                    scale=(0.02, 0.02, 0.02))
    points.glyph.scale_mode = 'scale_by_vector'
    points.mlab_source.dataset.point_data.vectors = np.tile(
        np.ones(embedding.shape[0]), (3, 1))
    color_vector = np.zeros(embedding.shape[0])
    points.mlab_source.dataset.point_data.scalars = color_vector
    mlab.outline(None, color=(.7, .7, .7), extent=[-1, 1, -1, 1, -1, 1])
    ax = mlab.axes(None, color=(.7, .7, .7), extent=[-1, 1, -1, 1, -1, 1],
                   ranges=[-1, 1, -1, 1, -1, 1], nb_labels=6)
    ax.label_text_property.font_size = 3
    ax.axes.font_factor = 0.4
    figure.scene.disable_render = False

    @mlab.animate(delay=250)
    def anim():
        f = mlab.gcf()
        while True:
            for observation in observations:
                color_vector[:] = 0
                color_vector[observation] = 1
                points.mlab_source.dataset.point_data.scalars = color_vector
                yield

    anim()
    mlab.show()


def animate_3d_mrt_sequence(dataset, embedding, sequence):
    """Animate the sensor activation sequence with sensor vector embedding
    in 3D space using mayavi

    Args:
        dataset (:obj:`CASASDataset`): CASAS dataset class containing sensor
            information.
        embedding (:obj:`numpy.ndarray`): A 2D array of size
            (#sensors, #vec_ndim) containing vector embedding of each sensor.
        sequence (:obj:`list`): List of sensor activation sequence.
    """
    figure = mlab.figure(dataset.data_dict['name'])
    figure.scene.disable_render = True
    points = mlab.points3d(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                           scale_factor=0.03)
    for i, x in enumerate(embedding):
        mlab.text3d(x[0], x[1], x[2], dataset.sensor_list[i]['name'],
                    scale=(0.02, 0.02, 0.02))
    points.glyph.scale_mode = 'scale_by_vector'
    points.mlab_source.dataset.point_data.vectors = np.tile(
        np.ones(embedding.shape[0]), (3, 1))
    color_vector = np.zeros(embedding.shape[0])
    points.mlab_source.dataset.point_data.scalars = color_vector
    mlab.outline(None, color=(.7, .7, .7), extent=[-1, 1, -1, 1, -1, 1])
    ax = mlab.axes(None, color=(.7, .7, .7), extent=[-1, 1, -1, 1, -1, 1],
                   ranges=[-1, 1, -1, 1, -1, 1], nb_labels=6)
    ax.label_text_property.font_size = 3
    ax.axes.font_factor = 0.4
    figure.scene.disable_render = False

    @mlab.animate(delay=250)
    def anim():
        f = mlab.gcf()
        while True:
            for sensor in sequence:
                color_vector[:] = 0
                color_vector[sensor] = 1
                points.mlab_source.dataset.point_data.scalars = color_vector
                yield

    anim()
    mlab.show()


def plot_3d_sensor_vector(dataset, embedding):
    """Plot sensor embedding in 3D space using mayavi

    Args:
    """
    figure = mlab.figure('Sensor Embedding (3D)')
    figure.scene.disable_render = True
    points = mlab.points3d(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                           scale_factor=0.03)
    for i, x in enumerate(embedding):
        mlab.text3d(x[0], x[1], x[2], dataset.sensor_list[i]['name'],
                    scale=(0.02, 0.02, 0.02))
    mlab.outline(None, color=(.7, .7, .7), extent=[-1, 1, -1, 1, -1, 1])
    ax = mlab.axes(None, color=(.7, .7, .7), extent=[-1, 1, -1, 1, -1, 1],
                   ranges=[-1, 1, -1, 1, -1, 1], nb_labels=6)
    ax.label_text_property.font_size = 3
    ax.axes.font_factor = 0.4
    figure.scene.disable_render = False
    mlab.show()


def plot_sensor_distance(dataset, distance, sensor, description=""):
    """Plot sensor distance on floorplan
    """
    if sensor is str:
        sensor_id = dataset.sensor_indices_dict[sensor]
    else:
        sensor_id = sensor
    drawing_data = dataset.site._prepare_floorplan()
    # Prepare aux data structure
    sensor_list = set(sensor['name'] for sensor in dataset.sensor_list)
    fig, (ax) = plt.subplots(1, 1)
    fig.set_size_inches(18, 18)
    ax.imshow(drawing_data['img'])
    active_patch_list = []
    # Draw Sensor block patches
    for key, patch in drawing_data['sensor_boxes'].items():
        if key in sensor_list:
            ax.add_patch(patch)
            active_patch_list.append(patch)
    # Draw Sensor name
    for key, text_data in drawing_data['sensor_texts'].items():
        if key in sensor_list:
            text_x, text_y, text = text_data
            temp_distance = distance[
                sensor_id, dataset.sensor_indices_dict[key]]
            text += "\n%.4f" % temp_distance
            temp_color = 'red' if temp_distance == 0 else 'black'
            ax.text(text_x, text_y, text, color=temp_color,
                    backgroundcolor=mcolors.colorConverter.to_rgba('#D3D3D3',
                                                                   0.7),
                    horizontalalignment='center', verticalalignment='top',
                    zorder=3)
    plt.title("%s - %s %s" % (
        dataset.data_dict['name'], dataset.sensor_list[sensor_id]['name'],
        description))
    plt.show()

