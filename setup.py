#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    author="Sailik Sengupta",
    author_email="sailiks@asu.edu",
    include_package_data=True,
    license="TBD",
    name="Equilibrium Strategies is General Sum Multi-Agent Reinforcement Learning",
    packages=find_packages("src", exclude=["tests*"]),
    package_dir={"": "src"},
    setup_requires=["pytest-runner"],
    tests_require=["pytest"]
)