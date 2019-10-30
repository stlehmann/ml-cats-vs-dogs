from typing import Tuple, Callable, Union, Iterable

import tensorflow as tf


def create_dense_model(
    input_shape: Tuple[int, int]=(64, 64),
    n_hidden: int = 1,
    n_units: int = 7,
    activation: Union[str, Callable]="relu",
    optimizer: Union[str, tf.keras.optimizers.Optimizer]="adam",
    loss: Union[str, tf.keras.losses.Loss]="binary_crossentropy",
    metrics: Iterable[Union[str, tf.keras.metrics.Metric]]=("accuracy",),
) -> tf.keras.Model:
    """Create a dense model.
    :param input_shape: input shape
    :param n_hidden: number of hidden layers
    :param n_units: units per hidden layer
    :param activation: activation function
    :param optimizer: optimizer
    :param loss: loss function
    :param metrics: additional metrics
    :return: model
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_units, activation=activation))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
