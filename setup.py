from setuptools import setup, find_packages

setup(
    name="flashace",
    version="2.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "e3nn>=0.5.0",
        "ase",
        "numpy",
        "pyyaml"          # Required for reading config.yaml
    ],
    extras_require={
        # Optional compiled acceleration; the codebase gracefully falls back to slower
        # index_add implementations when torch-scatter is unavailable.
        "scatter": ["torch-scatter"],
    },
)
