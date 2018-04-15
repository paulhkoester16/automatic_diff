from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='automatic_diff',
    version='0.1.0',
    description='naive automatic differentiation via dual numbers',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['numpy']
)