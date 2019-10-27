# %%
from __future__ import annotations

from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import numpy as np
import pyprojroot
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

sns.set()

# %%
data_p = pyprojroot.here("data/data.h5", [".here"])


# %%


def create_minibatch_gen(
    file: h5py.File,
    X_key: str,
    y_key: str,
    batch_size: int = 32,
    indices: List[int] = None,
):
    """Create a generator for that yields mini-batches.
    :param dataset: dataset with the training or test data
    :param batch_size: size of a mini-batch
    :param indices: indices that should be used for creating mini-batches, if None then all data will be used
    """
    X = file[X_key]
    y = file[y_key]

    if indices is None:
        indices = np.arange(X.shape[0])

    indices = list(indices)

    n_batches = len(indices) // batch_size
    minibatch_data = []
    minibatch_labels = []
    for i_batch in range(n_batches):
        for ix in indices[(i_batch * batch_size) : ((i_batch + 1) * batch_size)]:
            minibatch_data.append((X[ix] / 255.0).astype(np.float32))
            minibatch_labels.append(y[ix])
        yield (np.array(minibatch_data), np.array(minibatch_labels))


# %%


def create_model(
    input_shape=(64, 64),
    n_hidden: int = 1,
    n_units: int = 7,
    activation="relu",
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(input_shape[0] * input_shape[1], activation=activation))
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_units, activation=activation))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# %%


def train(n_epochs: int = 10, batch_size: int = 32, learning_rate=0.001):
    file = h5py.File(data_p, "r")
    model = create_model(optimizer=tf.keras.optimizers.Adam(lr=learning_rate))
    n_batches = file["train_data"].shape[0] // batch_size

    # validation data
    X_test = (file["test_data"][:] / 255.0).astype(np.float32)
    y_test = file["test_labels"][:]

    history = {"loss": [], "accuracy": [], "val_accuracy": []}

    for i_epoch in range(n_epochs):
        for i_batch, (X_train, y_train) in enumerate(
            create_minibatch_gen(
                file, X_key="train_data", y_key="train_labels", batch_size=32
            )
        ):
            loss, accuracy = model.train_on_batch(X_train, y_train)
            progress = 100.0 * i_batch / n_batches
            print(
                f"[Epoch {i_epoch:3n}] {progress:3.0f}%, loss: {loss:1.4f}, accuracy: {accuracy:1.4f}\r",
                end="",
            )

        _, val_accuracy = model.evaluate(X_test, y_test, verbose=False)
        history["loss"].append(loss)
        history["accuracy"].append(accuracy)
        history["val_accuracy"].append(val_accuracy)
        print(
            f"[Epoch {i_epoch:3n}] loss: {loss:1.3f}, accuracy: {accuracy:1.3f}, val_accuarcy: {val_accuracy:1.3f}"
        )
    file.close()
    return history


# %%

history = train(learning_rate=0.0001, n_epochs=200)
sns.lineplot(data=np.array(history["val_accuracy"]))
plt.xlabel("epochs")
plt.ylabel("val_accuracy")
plt.show()
