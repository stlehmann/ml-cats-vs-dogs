# %%
from __future__ import annotations

from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import numpy as np
import pyprojroot
import tensorflow as tf

sns.set()

# %%
data_p = pyprojroot.here("data/data.h5", [".here"])

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
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_units, activation=activation))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# %%

file = h5py.File(data_p, "r")

X_train = file["train_data"]
y_train = file["train_labels"]
X_test = file["test_data"][:]
y_test = file["test_labels"][:]

model = create_model(optimizer=tf.keras.optimizers.Adam(lr=0.00005), n_units=10, n_hidden=1)

history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    epochs=1000,
    verbose=2,
    batch_size=32,
    shuffle="batch",
)

sns.lineplot(data=np.array(history.history["accuracy"]), label="accuracy")
sns.lineplot(data=np.array(history.history["val_accuracy"]), label="val_accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()

# %%

# index = np.random.choice(np.arange(200))
#
# f = h5py.File(data_p, "r")
# img = f["train_data"][index][:] * 255.0
# plt.imshow(img, cmap="gray")
# plt.show()
#
# print(model.predict(f["train_data"][index].reshape(1, 64, 64)))