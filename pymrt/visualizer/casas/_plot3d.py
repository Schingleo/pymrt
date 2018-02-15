from mayavi import mlab


def plot3d_embeddings(dataset, embeddings):
    """Plot sensor embedding in 3D space using mayavi.

    Given the dataset and a sensor embedding matrix, each sensor is shown as
    a sphere in the 3D space. Note that the shape of embedding matrix is
    (num_sensors, 3) where num_sensors corresponds to the length of
    ``dataset.sensor_list``. All embedding vectors range between 0 and 1.

    Args:
        dataset (:obj:`~pymrt.casas.CASASDataset`): CASAS smart home dataset.
        embeddings (:obj:`numpy.ndarray`): 3D sensor vector embedding.
    """
    figure = mlab.figure('Sensor Embedding (3D)')
    figure.scene.disable_render = True
    points = mlab.points3d(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                           scale_factor=0.15)
    for i, x in enumerate(embeddings):
        mlab.text3d(x[0], x[1], x[2], dataset.sensor_list[i]['name'],
                    scale=(0.01, 0.01, 0.01))
    mlab.outline(None, color=(.7, .7, .7), extent=[0, 1, 0, 1, 0, 1])
    ax = mlab.axes(None, color=(.7, .7, .7), extent=[0, 1, 0, 1, 0, 1],
                   ranges=[0, 1, 0, 1, 0, 1], nb_labels=6)
    ax.label_text_property.font_size = 3
    ax.axes.font_factor = 0.3
    figure.scene.disable_render = False
    mlab.show()
