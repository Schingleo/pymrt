from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


def sensor_embed_cv_loss(embeddings,
                         inputs,
                         labels,
                         time_spans,
                         velocity=1,
                         name=None):
    """Helper function for velocity_loss functions.

    Computes squared difference compared to the velocity specified.

    Args:
        embeddings: A `Tensor` of shape `[num_sensors, dim]`, or a list of
            `Tensor` objects whose concatenation along dimension 0 has shape
            `[num_sensors, dim]`. The sensor embedding.
        inputs: A `Tensor` of type `int64` and shape `[batch_size, 1]`. The
            input sensor vectors of a batch.
        labels: A `Tensor` of type `int64` and shape `[batch_size, 1]`. The
            target sensor to reach.
        time_spans: A `Tensor` of type `float` and shape `[batch_size, 1]`. The
            time it takes for each input to reach the sensor in labels.
        velocity: A `float`. Targeted velocity.
        name: A name for the operation (optional).

    Returns:
        A `batch_size` 1-D tensor of per-example NCE losses.
    """
    with ops.name_scope(name, "velocity_logits",
                        [embeddings, inputs, labels, time_spans]):
        if labels.dtype != dtypes.int64:
            labels = math_ops.cast(labels, dtypes.int64)
        labels_flat = array_ops.reshape(labels, [-1])
        if inputs.dtype != dtypes.int64:
            inputs = math_ops.cast(inputs, dtypes.int64)
        inputs_flat = array_ops.reshape(inputs, [-1])
    # Acquire the total number of sensors from embedding directly (i.e. classes)
    n_classes = array_ops.shape(embeddings)[0]
    # Acquire the batch size of input tensor
    batch_size = array_ops.shape(inputs_flat)[0]
    # all_classes: `n_classes` size array
    all_classes = math_ops.cast(math_ops.range(0, limit=n_classes, delta=1),
                                dtype=dtypes.int64)
    # input encoded using embedding vectors:
    # 2D `Tensor` of shape `[batch_size, dim]`
    vec_inputs = embedding_ops.embedding_lookup(params=embeddings,
                                                ids=inputs_flat)
    # all embeddings: 2D `Tensor` of shape `[n_classes, dim]`
    vec_all = embedding_ops.embedding_lookup(params=embeddings,
                                             ids=all_classes)

    # Tile inputs and all into 3D array of shape `[batch_size, n_classes, dim]`.
    tiled_all = array_ops.tile(array_ops.expand_dims(vec_all, 0),
                               [batch_size, 1, 1])
    tiled_inputs = array_ops.tile(array_ops.expand_dims(vec_inputs, -1),
                                  multiples=[1, 1, n_classes])
    tiled_inputs = array_ops.transpose(tiled_inputs, perm=[0, 2, 1])

    # Euclidean distance between input and any other sensors
    # 2D `Tensor` of shape `[batch_size, n_classes]`
    tiled_distance = math_ops.reduce_sum(
        math_ops.square(tiled_all - tiled_inputs), axis=2
    )

    # Need to expand time_spans into the same shape as the tiled distance.
    # 2D `Tensor` of shape `[batch_size, n_classes]`
    tiled_time_spans = array_ops.tile(time_spans + 0.1,
                                      multiples=[1, n_classes])

    # Calculate the velocity
    # 2D `Tensor` of shape `[batch_size, n_classes]`
    tiled_velocity = math_ops.divide(tiled_distance,
                                     math_ops.pow(tiled_time_spans, 2))

    # Square of the velocity difference is used as logits for softmax
    # probability measure.
    # 2D `Tensor` of shape `[batchsize, n_classes]`
    tiled_velocity_logits = math_ops.square(
        tiled_velocity - velocity ** 2
    )

    # Since it is only one class can be selected in the softmax probability,
    # so the sparse softmax cross entropy function can be used directly with
    # label not one-hot encoded.
    softmax_losses = nn_ops.sparse_softmax_cross_entropy_with_logits(
        labels=labels_flat, logits=tiled_velocity_logits
    )

    # Internal parameters for debugging use
    local_params = [tiled_distance, tiled_velocity]

    # return summed loss
    return softmax_losses
