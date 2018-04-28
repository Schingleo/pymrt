import numpy as np
import matplotlib.pyplot as plt


def sensor_distance_pixelmap(distance_matrix, sensor_list=None,
                             annotation_fontsize=8, legend=True,
                             file_name=None):
    """Draw the distance between sensors as a pixel map.

    The distance between sensors is encoded as intensity of the corresponding
    pixel cell.

    Args:
        distance_matrix (:obj:`numpy.ndarray`): A `ndarray` of shape
            `[num_sensors, num_sensors]` where each element is the euclidean
            or cosine distance between two sensors.
        sensor_list (:obj:`list`): List of sensor dictionary where 'name' key is
            mapped to the sensor ID.
        annotation_fontsize (:obj:`int`): Font size of the annotation on the
            pixel image.
        legend (:obj:`bool`): Plot color bar on the image generated.
        file_name (:obj:`str`): Name of the file to save the drawing to.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(distance_matrix, interpolation='none', cmap='GnBu')
    if annotation_fontsize != 0:
        for i in range(distance_matrix.shape[0]):
            for j in range(distance_matrix.shape[1]):
                ax.annotate('%.2f' % distance_matrix[i, j],
                            fontsize=annotation_fontsize,
                            xy=(i, j), xycoords='data', xytext=(i, j),
                            horizontalalignment='center')
    # Move xaxis tick to the top
    ax.xaxis.tick_top()
    # Move xaxis label to the top as well
    ax.xaxis.set_label_position('top')
    # Each tick at the center
    ax.set_xticks(np.arange(distance_matrix.shape[1]), minor=False)
    ax.set_yticks(np.arange(distance_matrix.shape[1]), minor=False)
    # Prepare sensor label
    if sensor_list is None:
        sensor_labels = ['s_%d' % i for i in range(distance_matrix.shape[1])]
    else:
        sensor_labels = [
            sensor_list[i]['name'] for i in range(distance_matrix.shape[1])
        ]
    ax.set_xticklabels(sensor_labels, horizontalalignment='left',
                       minor=False, rotation=45)
    ax.set_yticklabels(sensor_labels, minor=False, rotation=0)
    if legend:
        plt.colorbar(im)
    plt.tight_layout()
    if file_name is None:
        plt.show()
