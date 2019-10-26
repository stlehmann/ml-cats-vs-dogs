# %%
import pathlib
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyprojroot
from PIL import Image

# %%

data_p = pyprojroot.here("data", [".here"])
train_p = data_p / "original/train"


def array_to_str(data: np.ndarray) -> str:
    return ",".join([str(x) for x in data.tolist()])


def str_to_array(s: str) -> np.ndarray:
    return np.array([np.uint8(x) for x in s.split(",")])


def train_image_gen(count: int = None, mode="L", size=(128, 128), flatten=True) -> Tuple[pathlib.Path, np.ndarray]:
    i = 0
    images_p = list(train_p.glob("*.jpg"))
    for f_p in images_p:
        try:
            label= f_p.name.split(".")[0]
        except IndexError:
            label = None

        # check if only valid labels
        if label not in ("cat", "dog"):
            raise ValueError(f"Unknown label in image {f_p}")

        i += 1
        im = Image.open(f_p)
        im_data = np.array(im.convert(mode).resize(size))
        if flatten:
            im_data = im_data.reshape(size[0]*size[1])
        yield label, im_data

        if count and i >= count:
            return

# %%

with (data_p / "train.csv").open("w") as f:
    f.write("label,data\n")
    for label, img in train_image_gen(5):
        data = '"' + array_to_str(img) + '"'
        f.write(",".join((label, data)) + "\n")

#%%

df = pd.read_csv(data_p / "train.csv")
