import tensorflow as tf

__all__ = ['sparse_categorical_accuracy']


def sparse_categorical_accuracy(y_true, y_pred):
    """
    Calculates the accuracy of model predictions according to true labels.

     Parameters:
         - y_true: True labels of the dataset (TensorFlow tensor with format [batch_size, sequence_length]).
         - y_pred: Model predictions (TensorFlow tensor with format [batch_size, sequence_length, num_classes]).

     Returns:
         - The calculated accuracy (TensorFlow tensor with a single value).
    """
    # Convert forecasts to a one-dimensional format ([batch_size * sequence_length])
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.reshape(y_pred, [-1])

    # Convert the actual labels to a one-dimensional format ([batch_size * sequence_length])
    y_true = tf.cast(tf.reshape(y_true, [-1]), dtype=y_pred.dtype)

    # Calculate the accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), dtype=tf.float32))

    return accuracy
