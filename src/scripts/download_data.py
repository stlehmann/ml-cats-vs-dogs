"""
Download Dog-vs-Cat dataset from Kaggle.

"""
# %%
from pyprojroot import here
import kaggle
import zipfile

# %%
root_p = here(".", [".here"])
data_p = root_p / "data/original"

# create data dir if not exists
if not data_p.is_dir():
    data_p.mkdir()

# %%
# load Kaggle api and authenticate
api = kaggle.KaggleApi()
api.authenticate()

print("Download data from kaggle...", end="")

api.competition_download_files("dogs-vs-cats", str(data_p))
print("done")

# %%
# extract root archive
archive = zipfile.ZipFile(root_p / "data/dogs-vs-cats.zip")
archive.extractall(data_p)

# %%
# extract train archive
train_archive = zipfile.ZipFile(root_p / "data/train.zip")
train_archive.extractall(data_p)

# %%
# extract test archive
test_archive = zipfile.ZipFile(root_p / "data/test1.zip")
test_archive.extractall(data_p)

# %%
# remove zipfiles
for f in data_p.glob("*.zip"):
    f.unlink()
