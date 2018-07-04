from setuptools import setup, find_packages

setup(
    name="wavenet",
    author="Panagiotis Eustratiadis",
    install_requires=[
        "bleach==1.5.0",  # Pin this because it causes problems.
        "h5py",
        "joblib",
        "keras",
        "matplotlib",
        "numpy",
        "scipy",
        "sklearn",
        "tensorflow"
    ],
    python_requires=">=3.6",
    version="0.0.1",
    packages=find_packages()
)
