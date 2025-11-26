from setuptools import setup, find_packages

setup(
    name="lowrank-rnn",
    version="0.1.0",
    description="Low-rank RNN training and analysis",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "torch>=1.12.0",
        "scikit-learn>=1.0.0",
    ],
)

