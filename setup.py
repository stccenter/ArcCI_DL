# setup.py
# !/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='IceClassifier',
    author='Kevin Wang',
    author_email='kevinw@berkeley.edu',
    url='',
    install_requires=[
        'numpy',
        'Pillow',
        'tqdm',
        'scikit-learn',
        'pytorch',
        'torchvision',
        'torchaudio',
        'cudatoolkit=10.2',
        'pytorch-lightning',
        'torchmetrics'
    ],
    packages=find_packages()
)
