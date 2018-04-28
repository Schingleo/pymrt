import numpy as np
from scipy.spatial import distance


def euclidean_distance_matrix(embeddings):
    """Get euclidean distance matrix based on embeddings

    Args:
        embeddings (:obj:`numpy.ndarray`): A `ndarray` of shape
            `[num_sensors, dim]` that translates each sensor into a vector
            embedding.

    Returns:
        A `ndarray` of shape `[num_sensors, num_sensors]` where each element is
        the euclidean distance between two sensors.
    """
    num_sensors = embeddings.shape[0]
    distance_matrix = np.zeros((num_sensors, num_sensors), dtype=np.float32)

    for i in range(num_sensors):
        for j in range(num_sensors):
            distance_matrix[i, j] = distance.euclidean(
                embeddings[i, :], embeddings[j, :]
            )
    return distance_matrix


def cosine_distance_matrix(embeddings):
    """Get cosine distance matrix based on embeddings

    Args:
        embeddings (:obj:`numpy.ndarray`): A `ndarray` of shape
            `[num_sensors, dim]` that translates each sensor into a vector
            embedding.

    Returns:
        A `ndarray` of shape `[num_sensors, num_sensors]` where each element is
        the cosine distance between two sensors.
    """
    num_sensors = embeddings.shape[0]
    distance_matrix = np.zeros((num_sensors, num_sensors), dtype=np.float32)

    for i in range(num_sensors):
        for j in range(num_sensors):
            distance_matrix[i, j] = distance.cosine(
                embeddings[i, :], embeddings[j, :]
            )
    return distance_matrix
