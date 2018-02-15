import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_sensor_distance(dataset, distance, sensor, title=""):
    """Plot sensor distance on floorplan

    Args:
        dataset (:obj:`~pymrt.casas.CASASDataset`): CASAS smart home dataset.
        distance (:obj:`numpy.ndarray`): Distance array matrix of shape
            (num_sensor, num_sensor).
        sensor (:obj:`int` or :obj:`string`): The ID or name of the sensor of
            interest.
        title (:obj:`str`): Title of the plot.
    """
    if sensor is str:
        sensor_id = dataset.sensor_indices_dict[sensor]
    else:
        sensor_id = sensor
    drawing_data = dataset.site.prepare_floorplan()
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
                sensor_id, dataset.sensor_indices_dict[key]
            ]
            text += "\n%.4f" % temp_distance
            temp_color = 'red' if temp_distance == 0 else 'black'
            ax.text(text_x, text_y, text, color=temp_color,
                    backgroundcolor=mcolors.colorConverter.to_rgba('#D3D3D3',
                                                                   0.7),
                    horizontalalignment='center', verticalalignment='top',
                    zorder=3)
    plt.title("%s - %s %s" % (
        dataset.data_dict['name'], dataset.sensor_list[sensor_id]['name'],
        title))
    plt.show()
