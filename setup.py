from setuptools import setup, find_packages

setup(
    name="ppg_pipeline",
    version="1.0.0",
    packages=["ppg_pipeline"],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
    ],
)
