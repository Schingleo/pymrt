import numpy as np
from mayavi import mlab
from ._plot3d import plot3d_embeddings


def animate3d_observations(dataset, embeddings, observations):
    """Animate Observation with sensor vector embedding in 3D space using mayavi

    Args:
        dataset (:obj:`~pymrt.casas.CASASDataset`): CASAS smart home dataset.
        embeddings (:obj:`numpy.ndarray`): 3D sensor vector embedding.
        observations (:obj:`list`): List of observations. Each observation is
            a :obj:`list` of index or target name of sensors that are ON,
            PRESENT or OPEN (depends on what type of sensor it is) at each
            time step.
    """
    figure = mlab.figure(dataset.data_dict['name'])
    _, points = plot3d_embeddings(dataset, embeddings, figure=figure)
    points.glyph.scale_mode = 'scale_by_vector'
    points.mlab_source.dataset.point_data.vectors = np.tile(
        np.ones(embeddings.shape[0]), (3, 1))
    color_vector = np.zeros(embeddings.shape[0])
    points.mlab_source.dataset.point_data.scalars = color_vector

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


def animate3d_sequence(dataset, embeddings, sequence):
    """Animate the sensor activation sequence with sensor vector embedding
    in 3D space using mayavi

    Args:
        dataset (:obj:`~pymrt.casas.CASASDataset`): CASAS smart home dataset.
        embeddings (:obj:`numpy.ndarray`): 3D sensor vector embedding.
        sequence (:obj:`list`): List of sensor activation sequence.
    """
    figure = mlab.figure(dataset.data_dict['name'])
    _, points = plot3d_embeddings(dataset, embeddings, figure=figure)
    points.glyph.scale_mode = 'scale_by_vector'
    points.mlab_source.dataset.point_data.vectors = np.tile(
        np.ones(embeddings.shape[0]), (3, 1))
    color_vector = np.zeros(embeddings.shape[0])
    points.mlab_source.dataset.point_data.scalars = color_vector

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
