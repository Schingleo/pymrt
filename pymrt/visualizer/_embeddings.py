import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from ..casas import CASASDataset
from ..tracking.utils import euclidean_distance_matrix
from ..tracking.utils import cosine_distance_matrix


def sensor_distance_pixelmap(distance_matrix, sensor_list=None,
                             annotation_fontsize=8, legend=True,
                             title=None, filename=None):
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
        title (:obj:`str`): Title of the plot
        filename (:obj:`str`): Name of the file to save the drawing to.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
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
        plt.colorbar(im, fraction=0.046, pad=0.04)
    if title is None:
        fig.suptitle('Sensor distance')
    else:
        fig.suptitle(title)
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_sensor_graph_with_embeddings(dataset, embeddings, threshold,
                                      distance="euclidean", title=None,
                                      filename=None):
    """Plot sensor accessibility graph based on sensor embeddings

    Plot sensor graph (weighted directed graph) using sensor embeddings
    calculated given threshold. The sensor pair whose distance is below the
    given threshold is considered adjacent (connected by an edge in the graph).

    Args:
        dataset (:obj:`~pymrt.casas.CASASDataset`): The dataset the embeddings
            are calculated.
        embeddings (:obj:`numpy.ndarray`): 2D array of sensor vector embeddings
            of shape `[num_sensors, z_dim]`.
        threshold (:obj:`float`): Threshold of the distance between two sensors
            for them to be considered adjacent.
        distance (:obj:`str`): Method used to calculate the distance between
            two sensor embedding vectors, `euclidean` or `cosine`.
        title (:obj:`str`): The name of the graph.
        filename (:obj:`str`): The name of the file to save the graph.
    """
    assert(isinstance(dataset, CASASDataset))
    # Calculate the distance and determine the adjancency matrix between
    # sensors.
    if distance == 'euclidean':
        distance_matrix = euclidean_distance_matrix(embeddings=embeddings)
    else:
        distance_matrix = cosine_distance_matrix(embeddings=embeddings)
    adjacency_matrix = (distance_matrix <= threshold)

    drawing_data = dataset.site.prepare_floorplan()
    sensor_list = [sensor['name'] for sensor in dataset.sensor_list]

    fig, ax = plt.subplots(figsize=(18, 18))
    ax.imshow(drawing_data['img'])
    active_patch_list = []
    # Draw sensor blocks
    for key, patch in drawing_data['sensor_boxes'].items():
        if key in sensor_list:
            ax.add_patch(patch)
            active_patch_list.append(patch)
    # Draw sensor annotations
    for key, text_data in drawing_data['sensor_texts'].items():
        if key in sensor_list:
            ax.text(*text_data, horizontalalignment='center',
                    verticalalignment='top', zorder=3)
    # Draw line connecting targeted sensors
    for i in range(adjacency_matrix.shape[0]):
        sensor_i = sensor_list[i]
        sensor_i_patch = drawing_data['sensor_boxes'][sensor_i]
        for j in range(adjacency_matrix.shape[1]):
            if i != j:
                sensor_j = sensor_list[j]
                sensor_j_patch = drawing_data['sensor_boxes'][sensor_j]
                if adjacency_matrix[i, j]:
                    ax.add_line(mlines.Line2D(
                        xdata=[sensor_i_patch.get_x(), sensor_j_patch.get_x()],
                        ydata=[sensor_i_patch.get_y(), sensor_j_patch.get_y()],
                        color='#D3D3D3', linestyle='-', zorder=1
                    ))
    # Show figure
    if title is None:
        title = dataset.get_name() + \
                ' sensor graph (threshold %.2f)' % threshold
    fig.suptitle = title
    fig.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
