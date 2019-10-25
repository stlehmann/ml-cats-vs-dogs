from setuptools import setup


setup(
    name="ml-cat-recognition",
    version="0.0.1",
    packages=["ml_cat_recognition"],
    install_requires=[],
    entry_points={
        "console_scripts": [
            "download-data = scripts.download_data.py",
        ]
    }
)