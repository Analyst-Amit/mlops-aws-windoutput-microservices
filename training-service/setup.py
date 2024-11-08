# setup.py in the root directory
from setuptools import find_packages, setup


setup(
    name="training_service",
    version="0.1",
    packages=find_packages(where="app/src"),
    package_dir={"": "app/src"},
    install_requires=[],
    include_package_data=True,
)
