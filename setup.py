from setuptools import setup, find_packages

setup(
    name="DejaVu OSS",
    packages=find_packages(exclude=["tests"]),
)