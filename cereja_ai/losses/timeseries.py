import tensorflow as tf

__all__ = ['sparse_categorical_crossentropy']


def sparse_categorical_crossentropy(y_true, y_pred):
    """
    Calculates the loss according to the model outputs and the labels of a dataset

    y_true: Labels of a dataset (Tensorflow tensor with format [batch_size, sequence_length, num_classes])
    y_pred: Model outputs (Tensorflow tensor with format [batch_size, sequence_length, num_classes])

    Returns: The calculated loss (Tensorflow tensor with a single value)
    """

    # convert to two dimensions (batch_size * sequence_length, num_classes)
    y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])

    # Convert the labels to 1 dimension (batch_size * sequence_length)
    y_true = tf.reshape(y_true, [-1])

    # loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    # calculate loss
    loss = loss_fn(y_true, y_pred)
    return loss
