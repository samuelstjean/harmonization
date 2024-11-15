from setuptools import find_packages, setup

scripts = ['scripts/harmonization_build_dictionary',
           'scripts/harmonization_from_dictionary']

setup(
    packages=find_packages(),
    scripts=scripts,
)
