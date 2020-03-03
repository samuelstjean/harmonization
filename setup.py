from io import open
from setuptools import find_packages
from numpy.distutils.core import Extension, setup


with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

f_sources = ['glmnet/glmnet.f', 'glmnet/glmnet.pyf']

fflags = ['-fdefault-real-8',
          '-ffixed-form',
          '-O3',
          '-fPIC',
          '-shared']

ext_modules = [Extension('harmonization._glmnet',
                         sources=f_sources,
                         extra_f77_compile_args=fflags,
                         extra_f90_compile_args=fflags)]

scripts = ['scripts/harmonization_build_dictionary',
           'scripts/harmonization_from_dictionary']

setup_requires = ['numpy >= 1.15']

setup(
    name='harmonization',
    version='0.1',
    author='Samuel St-Jean',
    author_email='samuel.st-jean@usherbrooke.ca',
    packages=find_packages(),
    scripts=scripts,
    url='https://github.com/samuelstjean/harmonization',
    license='GPLv2',
    description='Implementation of "Harmonization of diffusion MRI datasets with adaptive dictionary learning".',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=install_requires,
    setup_requires=setup_requires,
    ext_modules=ext_modules,
)
