import numpy as np
import itertools
from collections import OrderedDict
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import lines as mlines
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MultipleLocator
from matplotlib.animation import FuncAnimation
from pymrt.casas import CASASDataset


def mrt_playback_on_site(mrt_result):
    """Playback mrt results

    The function does a playback on the tracking results (with the respect to
    ground truth) on 2D floor plan. All the information are organized in the
    mrt_result structure, including dataset information, event lists,
    sensor event to resident association, etc. The actual plotting is
    implemented using matplotlib.

    The ``mrt_result`` structure contains the following keys:

    - ``dataset``: a :obj:`~pymrt.casas.CASASDataset` structure.
    - ``observation``: a `list` of observation.
    - ``time``: a list of `datetime` of each observation in observation list.
    - ``ota``: a list of observation to track association.
    - ``steps``: a list of `int` representing the index of observation which
      each entry in `ota` are associated with.

    Args:
        mrt_result (:obj:`dict`): A ``dict`` structure including all
            information about the multi-resident tracking.
    """
    dataset = mrt_result['dataset']
    assert(isinstance(dataset, CASASDataset))

    # Data rename
    obs_seq = mrt_result['observation']
    time_seq = mrt_result['time']
    track_info = mrt_result['track_info']
    track_mta = mrt_result['track_mta']
    steps = mrt_result['steps']

    # Prepare data to draw for each step -
    # The information needed for plot at each step include:
    # 1. Ground truth
    #    - resident location
    #    - resident movement
    # 2. Tracking result
    #    - measurement to track association
    #    - movement of each track
    resident_colors = {
        resident: dataset.get_resident_color(resident)
        for resident in dataset.resident_dict
    }
    # `dict` indexed by sensor ID containing list of tracks and residents
    # associated.
    sensor_truth_dict = {
        sensor['name']: [] for sensor in dataset.sensor_list
    }
    sensor_track_dict = {
        sensor['name']: [] for sensor in dataset.sensor_list
    }
    # Arrows that represent movement of each resident
    truth_path_dict = []
    track_path_dict = []

    # for i, step in enumerate(steps):
    #     for sensor in sensor_truth_dict:
    #         sensor_truth_dict[sensor].append([])
    #         sensor_track_dict[sensor].append([])
    #     # Record resident location
    #     for resident in rma:
    #         if len(rma[resident][step]) != 0:
    #             residents_past_loc[resident] = {
    #                 'loc': rma[resident][step],
    #                 'step': step,
    #                 'datetime': time_seq[step]
    #             }
    #             for sensor in rma[resident][step]:
    #                 sensor_truth_dict[sensor][i].append(resident)
    #     # Record tracking information
    #     for sensor in ota[i]:
    #         sensor_track_dict[sensor][i] = ota[i][sensor]

    floorplan_dict = dataset.site.prepare_floorplan(
        categories=['Motion', 'Door', 'Item']
    )

    fig = plt.figure('Tracking playback')
    # ax0 - floorplan
    # ax1 - ground truth
    # ax2 - tracking
    gs = gridspec.GridSpec(2,2)
    ax0 = plt.subplot(gs[:, 0])
    ax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[1, 1], sharex=ax1)

    # Draw floorplan
    ax0.imshow(floorplan_dict['img'])
    # Draw sensor patches
    active_patch_list = []
    for key, patch in floorplan_dict['sensor_boxes'].items():
        ax0.add_patch(patch)
        active_patch_list.append(patch)
    # Draw sensor names
    for key, text in floorplan_dict['sensor_texts'].items():
        sensor_text = ax0.text(
            *text, color='black',
            backgroundcolor=mcolors.colorConverter.to_rgba('#D3D3D3', 0.7),
            horizontalalignment='center', verticalalignment='top',
            zorder=3
        )

    measurement_info = [
        sensor['name'] for sensor in dataset.sensor_list
    ]

    def sequence_init():
        return active_patch_list + [sensor_text]

    def sequence_update(frame):
        ax1.cla()
        ax2.cla()
        plotted_items = active_patch_list + [sensor_text]
        plotted_items += plot_observation(
            obs_seq=obs_seq, time_seq=time_seq,
            step=steps[frame],
            resident_color=resident_colors,
            measurement_info=measurement_info,
            steps_before=40, steps_total=50, ax=ax1,
            xlabel_shared=True)
        plotted_items += plot_track(
            mta=track_mta, time_seq=time_seq[steps[0]:steps[-1]],
            step=frame,
            measurement_info=measurement_info,
            steps_before=40, steps_total=50, ax=ax2)
        return plotted_items

    ani = FuncAnimation(fig, sequence_update, init_func=sequence_init,
                        frames=range(len(steps)),
                        blit=False, interval=500)
    plt.show()


def mrt_observation_playback(obs_seq, time_seq, dataset):
    """Multi-resident observation playback.

    Args:
        obs_seq (:obj:`list`): List of observation.
        time_seq (:obj:`list`): List of `datetime` associated with each
            sensor message.
        dataset (:obj:`~pymrt.casas.CASASDataset`): The `CASASDataset` object
            containing the information about the dataset.
    """
    assert(isinstance(dataset, CASASDataset))

    floorplan_dict = dataset.site.prepare_floorplan(
        categories=['Motion', 'Door', 'Item']
    )

    fig = plt.figure('MRT Observation Playback')
    # ax0 - Floorplan
    # ax1 - Ground truth graph
    gs = gridspec.GridSpec(5, 1)
    ax0 = plt.subplot(gs[0:4, 0])
    ax1 = plt.subplot(gs[4, 0])

    # Draw floorplan
    ax0.imshow(floorplan_dict['img'])

    # Draw sensor patches
    active_patch_list = []
    for key, patch in floorplan_dict['sensor_boxes'].items():
        ax0.add_patch(patch)
        active_patch_list.append(patch)
    # Draw sensor names
    for key, text in floorplan_dict['sensor_texts'].items():
        sensor_text = ax0.text(
            *text, color='black',
            backgroundcolor=mcolors.colorConverter.to_rgba('#D3D3D3', 0.7),
            horizontalalignment='center', verticalalignment='top',
            zorder=3
        )

    measurement_info = [
        sensor['name'] for sensor in dataset.sensor_list
    ]

    resident_color = {
        resident: dataset.get_resident_color(resident)
        for resident in dataset.resident_dict
    }

    def sequence_init():
        return active_patch_list + [sensor_text]

    def sequence_update(frame):
        ax1.cla()
        plotted_items = active_patch_list + [sensor_text]
        patches = plot_observation(
            obs_seq=obs_seq,
            time_seq=time_seq,
            step=frame,
            resident_color=resident_color,
            measurement_info=measurement_info,
            steps_before=40,
            steps_total=50,
            ax=ax1,
            xlabel_shared=False)
        plotted_items += patches
        return plotted_items

    ani = FuncAnimation(fig, sequence_update, init_func=sequence_init,
                        frames=range(40, len(time_seq)),
                        blit=False, interval=500)
    plt.show()


def plot_observation(obs_seq, time_seq, step,
                     steps_before=20, steps_total=30,
                     resident_color=None,
                     measurement_info=None, ax=None,
                     title=None, with_legend=True,
                     xlabel_shared=False):
    """Plot observation sequence as evolving procedure

    Args:
        obs_seq (:obj:`list`): List of observation.
        time_seq (:obj:`list`): List of `datetime` associated with each
            sensor message.
        step (:obj:`list`): At which step to plot the observation.
        steps_before (:obj:`int`): Number of observations to plot before
            current step.
        steps_total (:obj:`int`): Total number of observations to plot. It has
            to be larger than `steps_before`.
        resident_color (:obj:`dict`): Color string indexed by resident name.
        measurement_info (:obj:`list`): A list of description and name
            corresponding to the measurement.
        ax (:obj:`matplotlib.ax`): Axes to be plotted on.
        title (:obj:`str`): Plot title.
        with_legend (:obj:`bool`): Show legend besides the plot.
        xlabel_shared (:obj:`bool`): If x-axis label is shared. If true, the
            label on x-axis is hidden.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))

    if steps_total < steps_before:
        steps_total = int(1.5 * steps_before)

    # Make a temporary list of all measurements in the window
    step_start = max(0, step - steps_before)
    step_stop = min(len(obs_seq) - 1, step_start + steps_total)

    # Distinct measurement in the window
    measurement_dict = OrderedDict()
    # Resident path in the window
    path_plot = {}
    # Temporary structure for where each resident was at previously
    prev_resident_states = {}
    # Time labels
    time_labels = []
    for s in range(step_start, step_stop):
        # current observation
        observation = obs_seq[s]
        for measurement in observation:
            if measurement not in measurement_dict:
                measurement_dict[measurement] = [observation[measurement]]
            elif observation[measurement] not in measurement_dict[measurement]:
                measurement_dict[measurement].append(observation[measurement])
            detail = observation[measurement]
            for resident in detail['residents']:
                prev_state = prev_resident_states.get(resident, None)
                if prev_state is not None:
                    prev_step = prev_state['step']
                    prev_loc = prev_state['measurement']
                    path_key = (prev_step, prev_loc, s, measurement)
                    if path_key not in path_plot:
                        path_plot[path_key] = [{
                            'track': resident,
                            'probability': 1
                        }]
                    else:
                        path_plot[path_key].append({
                            'track': resident,
                            'probability': 1
                        })
        # Update location for each track
        for measurement in observation:
            for resident in observation[measurement]['residents']:
                prev_resident_states[resident] = {
                    'step': s,
                    'measurement': measurement
                }
        # Update time labels
        time_labels.append('%s [%d]' % (
            time_seq[s].strftime("%m/%d/%Y %H:%M:%S"), s
        ))

    # After all measurements are loaded, calculate values for bar plot
    measurement_y_indices = {
        measurement: i for i, measurement in enumerate(measurement_dict.keys())
    }
    patch_data_list = []
    scatter_x = []
    scatter_y = []
    for measurement in measurement_dict:
        temp_step = step_start
        for detail in measurement_dict[measurement]:
            measurement_y = measurement_y_indices[measurement]
            current_patch_data = {
                'y': measurement_y
            }
            # Start of the patch
            if detail['start'] <= time_seq[step_start]:
                current_patch_data['x0'] = step_start
            else:
                while temp_step < step_stop and \
                        time_seq[temp_step] != detail['start']:
                    temp_step += 1
                current_patch_data['x0'] = temp_step
            scatter_x.append(temp_step)
            scatter_y.append(measurement_y)
            # Always add one when looking for stop
            temp_step += 1
            # Stop of patch
            while temp_step < step_stop and \
                    time_seq[temp_step] <= detail['stop']:
                scatter_x.append(temp_step)
                scatter_y.append(measurement_y)
                temp_step += 1
            if temp_step == step_stop:
                current_patch_data['x1'] = step_stop
            else:
                event_stop_offset = (time_seq[temp_step] -
                                     detail['stop']).total_seconds()
                step_duration = (time_seq[temp_step] -
                                 time_seq[temp_step-1]).total_seconds()
                current_patch_data['x1'] = temp_step \
                    - event_stop_offset / step_duration
            patch_data_list.append(current_patch_data)

    # Draw Patch
    patches = []
    for patch_data in patch_data_list:
        fancybox = mpatches.FancyBboxPatch(
            [patch_data['x0'], patch_data['y'] - 0.2],
            patch_data['x1'] - patch_data['x0'], 0.4,
            boxstyle=mpatches.BoxStyle("Round", pad=0.05)
        )
        patches.append(fancybox)

    patch_collection = PatchCollection(patches, alpha=0.5, facecolors='c')
    ax.add_collection(patch_collection)

    # Draw measurements using circle patches
    # circles = []
    # for i in range(len(scatter_x)):
    #     circle = mpatches.Circle((scatter_x[i], scatter_y[i]),
    #                              radius=0.15, fill=True)
    #     circles.append(circle)
    # circle_collection = PatchCollection(circles, alpha=1, facecolors='r')
    # ax.add_collection(circle_collection)
    scatter_points = ax.plot(scatter_x, scatter_y, ls='None', marker='o',
                             mfc='r', mec='r', ms=5)

    # Draw arrows for each resident path
    resident_arrows = []
    for path_key in path_plot.keys():
        x0, measurement_0, x1, measurement_1 = path_key
        y0 = measurement_y_indices[measurement_0]
        y1 = measurement_y_indices[measurement_1]
        if x1 > step:
            arrow_alpha = 0.3
        else:
            arrow_alpha = 0.9
        transform = 0
        for resident_detail in path_plot[path_key]:
            if y1 == y0:
                arrow_width = 0.06
            else:
                arrow_width = 0.04
            arrow = ax.arrow(
                x0 - 0.1 * transform,
                y0 - 0.1 * transform,
                x1-x0, y1-y0,
                length_includes_head=True,
                shape='full', alpha=arrow_alpha,
                fc=resident_color[resident_detail['track']],
                ec='none',
                width=arrow_width
            )
            resident_arrows.append(arrow)
            transform += 1

    ax.set_xlim(step_start, step_start + steps_total)
    # Only generate 10 x_labels
    x_tick_distance = int(np.floor(steps_total/10.))
    x_ticks_array = list(range(step_start, step_stop, x_tick_distance))
    if x_ticks_array[-1] != step_stop - 1:
        x_ticks_array.append(step_stop - 1)
    x_ticklabels_array = [
        time_labels[i - step_start] for i in x_ticks_array
    ]
    ax.set_xticks(x_ticks_array)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xticklabels(x_ticklabels_array, rotation=45,
                       horizontalalignment='right',
                       verticalalignment='top',
                       visible=not xlabel_shared)
    measurement_list = list(measurement_dict.keys())
    ax.set_ylim(-1, len(measurement_list))
    ax.set_yticks(list(range(len(measurement_list))))
    if measurement_info is not None:
        measurement_labels = [
            '[%d]%s' % (i, measurement_info[i]) for i in measurement_list
        ]
    else:
        measurement_labels = measurement_list
    ax.set_yticklabels(measurement_labels)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='#F0F0F0', which='major', linestyle='--', linewidth=1,
                  alpha=0.5)
    ax.xaxis.grid(color='#D0D0D0', which='minor', linestyle='--', linewidth=1,
                  alpha=0.7)
    ax.xaxis.grid(color='#D0D0D0', which='major', linestyle='--', linewidth=1,
                  alpha=0.7)
    ax.set_xlabel('Time tag [virtual time step]')
    ax.set_ylabel('Sensor ID')
    if title is not None:
        ax.set_title(title)
    if with_legend:
        legend_handles = []
        for resident in prev_resident_states:
            legend_handles.append(
                mpatches.FancyArrowPatch([], [],
                                         color=resident_color[resident],
                                         label=resident)
            )
        legend_handles.append(
            mlines.Line2D([], [], marker='o', color='r', label='Measurement')
        )
        legend_handles.append(
            mpatches.Patch([], color='c', label='Sensor Active Window')
        )
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc=2,
                  borderaxespad=0.)
    return patches + [scatter_points] + resident_arrows


def plot_track(mta, time_seq, step, track_info=None,
               steps_before=20, steps_total=30,
               measurement_info=None, ax=None,
               xlabel_shared=False):
    """Plot tracks identified by the tracking algorithm.

    Args:
        mta (:obj:`list`): List of measurement-to-track association.
        time_seq (:obj:`list`): List of `datetime` associated with each
            sensor message.
        step (:obj:`list`): At which step to plot the observation.
        steps_before (:obj:`int`): Number of observations to plot before
            current step.
        steps_total (:obj:`int`): Total number of observations to plot. It has
            to be larger than `steps_before`.
        measurement_info (:obj:`list`): A list of description and name
            corresponding to the measurement.
        ax (:obj:`matplotlib.ax`): Axes to be plotted on.
        xlabel_shared (:obj:`bool`): If x-axis label is shared. If true, the
            label on x-axis is hidden.
    """
    if track_info is None:
        # Check if the function has static track_info defined
        if hasattr(plot_track, 'track_info'):
            track_info = getattr(plot_track, 'track_info')
        else:
            track_info = {}
            setattr(plot_track, 'track_info', track_info)

    if hasattr(plot_track, 'colors'):
        color_reservior = getattr(plot_track, 'colors')
    else:
        # Prepare color cycle for tracks
        prop_cycle = plt.rcParams['axes.prop_cycle']
        color_reservior = itertools.cycle(list(prop_cycle.by_key()['color']))
        setattr(plot_track, 'colors', color_reservior)

    if ax is None:
        fig, ax = plt.subplots()

    if steps_total < steps_before:
        steps_total = int(1.5 * steps_before)

    # Make a temporary list of all measurements in the window
    step_start = max(0, step - steps_before)
    step_stop = min(len(mta) - 1, step_start + steps_total)

    # Distinct measurement in the window
    measurement_dict = OrderedDict()
    # Resident path in the window
    path_plot = {}
    # Temporary structure for where each track was at previously
    prev_track_states = {}
    # Time labels
    time_labels = []
    for s in range(step_start, step_stop):
        # current observation
        observation = mta[s]
        for measurement in observation:
            if measurement not in measurement_dict:
                measurement_dict[measurement] = [observation[measurement]]
            elif observation[measurement] not in measurement_dict[measurement]:
                measurement_dict[measurement].append(observation[measurement])
            detail = observation[measurement]
            for track, track_detail in detail['tracks'].items():
                prev_state = prev_track_states.get(track, None)
                if prev_state is not None:
                    prev_step = prev_state['step']
                    prev_loc = prev_state['measurement']
                    path_key = (prev_step, prev_loc, s, measurement)
                    if path_key not in path_plot:
                        path_plot[path_key] = [{
                            'track': track,
                            'probability': track_detail['track_weight']
                        }]
                    else:
                         path_plot[path_key].append({
                            'track': track,
                            'probability': track_detail['track_weight']
                        })
        # Update location for each track
        for measurement in observation:
            for track in observation[measurement]['tracks']:
                prev_track_states[track] = {
                    'step': s,
                    'measurement': measurement
                }
        # Update time labels
        time_labels.append('%s [%d]' % (
            time_seq[s].strftime("%m/%d/%Y %H:%M:%S"), s
        ))

    # After all measurements are loaded, calculate values for bar plot
    measurement_y_indices = {
        measurement: i
        for i, measurement in enumerate(measurement_dict.keys())
    }
    patch_data_list = []
    scatter_x = []
    scatter_y = []
    for measurement in measurement_dict:
        temp_step = step_start
        for detail in measurement_dict[measurement]:
            measurement_y = measurement_y_indices[measurement]
            current_patch_data = {
                'y': measurement_y
            }
            # Start of the patch
            if detail['start'] <= time_seq[step_start]:
                current_patch_data['x0'] = step_start
            else:
                while temp_step < step_stop and \
                        time_seq[temp_step] != detail['start']:
                    temp_step += 1
                current_patch_data['x0'] = temp_step
            scatter_x.append(temp_step)
            scatter_y.append(measurement_y)
            # Always add one when looking for stop
            temp_step += 1
            # Stop of patch
            while temp_step < step_stop and \
                    time_seq[temp_step] <= detail['stop']:
                scatter_x.append(temp_step)
                scatter_y.append(measurement_y)
                temp_step += 1
            if temp_step == step_stop:
                current_patch_data['x1'] = step_stop
            else:
                event_stop_offset = (time_seq[temp_step] -
                                     detail['stop']).total_seconds()
                step_duration = (time_seq[temp_step] -
                                 time_seq[temp_step - 1]).total_seconds()
                current_patch_data['x1'] = temp_step \
                                           - event_stop_offset / step_duration
            patch_data_list.append(current_patch_data)

    # Draw Patch
    patches = []
    for patch_data in patch_data_list:
        fancybox = mpatches.FancyBboxPatch(
            [patch_data['x0'], patch_data['y'] - 0.2],
            patch_data['x1'] - patch_data['x0'], 0.4,
            boxstyle=mpatches.BoxStyle("Round", pad=0.05)
        )
        patches.append(fancybox)

    patch_collection = PatchCollection(patches, alpha=0.5, facecolors='c')
    ax.add_collection(patch_collection)

    # Draw measurements using circle patches
    # circles = []
    # for i in range(len(scatter_x)):
    #     circle = mpatches.Circle((scatter_x[i], scatter_y[i]),
    #                              radius=0.15, fill=True)
    #     circles.append(circle)
    # circle_collection = PatchCollection(circles, alpha=1, facecolors='r')
    # ax.add_collection(circle_collection)
    scatter_points = ax.plot(scatter_x, scatter_y, ls='None', marker='o',
                             mfc='r', mec='r', ms=5)

    # Draw arrows for each track path
    track_arrows = []
    for path_key in path_plot.keys():
        x0, measurement_0, x1, measurement_1 = path_key
        y0 = measurement_y_indices[measurement_0]
        y1 = measurement_y_indices[measurement_1]
        if x1 > step:
            arrow_alpha = 0.3
        else:
            arrow_alpha = 0.9
        transform = 0
        for track_detail in path_plot[path_key]:
            if y1 == y0:
                arrow_width = 0.06
            else:
                arrow_width = 0.04
            track = track_detail['track']
            if track in track_info:
                arrow_color = track_info[track]['color']
            else:
                arrow_color = next(color_reservior)
                track_info[track] = {
                    'color': arrow_color
                }
            arrow = ax.arrow(
                x0,
                y0 - 0.1 * transform,
                x1-x0, y1-y0,
                length_includes_head=True,
                shape='full', alpha=arrow_alpha,
                fc=arrow_color,
                ec='none',
                width=arrow_width
            )
            track_arrows.append(arrow)
            transform += 1

    # Add weight annotation for each track
    track_annotations = []
    for s in range(step_start, step_stop):
        for measurement, measurement_detail in mta[s].items():
            transform = 0
            for track, track_detail in measurement_detail['tracks'].items():
                if s > step:
                    track_alpha = 0.3
                else:
                    track_alpha = 0.9
                if track in track_info:
                    track_color = track_info[track]['color']
                else:
                    track_color = next(color_reservior)
                    track_info[track] = {
                        'color': track_color
                    }
                track_text = ax.annotate(
                    'T%d: \n%.3f' % (track, track_detail['track_weight']),
                    xy=(s, measurement_y_indices[measurement] - 0.05
                        - 0.2 * transform),
                    xycoords='data',
                    xytext=(s, measurement_y_indices[measurement] - 0.05
                            - 0.2 * transform),
                    textcoords='data',
                    color=track_color,
                    va='top',
                    ha='center',
                    fontsize=6,
                    alpha=track_alpha
                )
                track_annotations.append(track_text)
                transform += 1

    # Draw ticks and labels
    ax.set_xlim(step_start, step_start + steps_total)
    # Only generate 10 x_labels
    x_tick_distance = int(np.floor(steps_total/10.))
    x_ticks_array = list(range(step_start, step_stop, x_tick_distance))
    if x_ticks_array[-1] != step_stop - 1:
        x_ticks_array.append(step_stop - 1)
    x_ticklabels_array = [
        time_labels[i - step_start] for i in x_ticks_array
    ]
    ax.set_xticks(x_ticks_array)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xticklabels(x_ticklabels_array, rotation=45,
                       horizontalalignment='right',
                       verticalalignment='top',
                       visible=not xlabel_shared)
    measurement_list = list(measurement_dict.keys())
    ax.set_ylim(-1, len(measurement_list))
    ax.set_yticks(list(range(len(measurement_list))))
    if measurement_info is not None:
        measurement_labels = [
            '[%d]%s' % (i, measurement_info[i]) for i in measurement_list
        ]
    else:
        measurement_labels = measurement_list
    ax.set_yticklabels(measurement_labels)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='#F0F0F0', which='major', linestyle='--', linewidth=1,
                  alpha=0.5)
    ax.xaxis.grid(color='#D0D0D0', which='minor', linestyle='--', linewidth=1,
                  alpha=0.7)
    ax.xaxis.grid(color='#D0D0D0', which='major', linestyle='--', linewidth=1,
                  alpha=0.7)

    return patches + [scatter_points] + track_arrows + track_annotations
