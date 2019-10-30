# %%
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import numpy as np
import pyprojroot
import tensorflow as tf

from cats_vs_dogs.models import create_dense_model

sns.set()

# random seed
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %%
data_p = pyprojroot.here("data/data.h5", [".here"])

# %%

file = h5py.File(data_p, "r")

X_train = file["train_data"]
y_train = file["train_labels"]
X_test = file["test_data"][:]
y_test = file["test_labels"][:]

model = create_dense_model(
    optimizer=tf.keras.optimizers.Adam(lr=0.00005),
    n_units=10,
    n_hidden=1,
    metrics=["accuracy"],
)

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
