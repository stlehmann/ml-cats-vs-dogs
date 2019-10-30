# %%
import pathlib
from typing import Tuple

import h5py
import numpy as np
import pyprojroot
from PIL import Image

# %%

data_p = pyprojroot.here("data", [".here"])
train_p = data_p / "original/train"


def get_label(fp: pathlib.Path) -> int:
    """Return label for the given image.

    :param fp: filepath
    :return: 1 if cat, 0 if dog
    """
    try:
        label_s = fp.name.split(".")[0]
    except IndexError:
        label_s = None
    # check if only valid labels
    if label_s not in ("cat", "dog"):
        raise ValueError(f"Unknown label in image {fp}")
    return 1 if label_s == "cat" else 0


def images_to_h5(
    output_p: pathlib.Path,
    count: int = None,
    mode="L",
    size=(128, 128),
    test_size: float = 0.2,
    shuffle: bool = False,
) -> None:

    # path to image and output file
    images_p = list(train_p.glob("*.jpg"))
    output_f = h5py.File(output_p, "w")

    # count of datasets in total
    n_datasets = count or len(images_p)

    # size of train and test data
    assert isinstance(test_size, (float, int))
    if isinstance(test_size, float):
        test_size = int(test_size * n_datasets)
    train_size = n_datasets - test_size

    # shape of train and test data
    train_shape = (train_size, size[0], size[1])
    test_shape = (test_size, size[0], size[1])

    output_f.create_dataset("train_data", train_shape, dtype=np.float32)
    output_f.create_dataset("train_labels", (train_size,), dtype=np.uint8)
    output_f.create_dataset("test_data", test_shape, dtype=np.float32)
    output_f.create_dataset("test_labels", (test_size,), dtype=np.uint8)

    # indices for train and test data
    indices = np.arange(n_datasets)
    if shuffle:
        indices = np.random.permutation(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # train data in hdf5 file
    for i, ix in enumerate(train_indices):
        fp = images_p[ix]
        output_f["train_data"][i, ...] = np.array(
            Image.open(fp).convert(mode).resize(size)
        ) / 255.0
        output_f["train_labels"][i] = get_label(fp)

        if i % max(1, int(train_size / 100.0)) == 0:
            print("Exporting training data...{0}%\r".format(int(100.0 * (i+1) / train_size)), end="")

    print("Exporting training data...done")
    for i, ix in enumerate(test_indices):
        fp = images_p[ix]
        output_f["test_data"][i, ...] = np.array(
            Image.open(fp).convert(mode).resize(size)
        ) / 255.0
        output_f["test_labels"][i] = get_label(fp)

        if i % max(1, int(test_size / 100.0)) == 0:
            print("Exporting test data...{0}%\r".format(int(100.0 * (i+1) / test_size)), end="")
    print("Exporting test data...done")

    output_f.close()


images_to_h5(data_p / "data.h5", count=2000, size=(64, 64), test_size=0.2, shuffle=True)
