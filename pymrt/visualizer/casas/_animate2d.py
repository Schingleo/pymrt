import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation


def animate2d_observations(site, observations, sensors=None):
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
    drawing_data = site.prepare_floorplan()
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


def animate2d_sequence(site, sequence, sensors=None):
    """Animate sensor event sequence (only starting part: ON, PRESENT,
    or OPEN) on CASAS site floorplan.
    The animation is developed using matplotlib.

    Args:
        site (:obj:`CASASSite`): CASAS site class containing floorplan and
            sensor information.
        sequence (:obj:`list`): List of sensor activation sequence.
        sensors (:obj:`list`): List of sensors represented as dictionary.
    """
    drawing_data = site.prepare_floorplan()
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
