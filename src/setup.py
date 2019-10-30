from setuptools import setup


setup(
    name="cats-vs-dogs",
    version="0.0.1",
    packages=["cats_vs_dogs"],
    install_requires=['h5py', 'numpy', 'pyprojroot', 'keras', 'matplotlib', 'seaborn', 'tensorflow'],
    entry_points={
        "console_scripts": [
            "download-data = scripts.download_data.py",
        ]
    }
)