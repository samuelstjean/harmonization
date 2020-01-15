from setuptools import find_packages
from numpy.distutils.core import Extension, setup

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

f_sources = ['harmonization/glmnet/glmnet.f']

fflags = ['-fdefault-real-8',
          '-ffixed-form',
          '-O3',
          '-fPIC',
          '-shared']

module = Extension('harmonization._glmnet',
                   sources=f_sources,
                   extra_f77_compile_args=fflags,
                   extra_f90_compile_args=fflags)

setup(
    name='harmonization',
    version='0.1',
    author='Samuel St-Jean',
    author_email='samuel@isi.uu.nl',
    packages=find_packages(),
    # scripts=['scripts/dpr', 'scripts/dpr_make_fancy_graph'],
    url='https://github.com/samuelstjean/harmonization',
    license='GPL2',
    description='Implementation of "Harmonization of diffusion MRI datasets with adaptive dictionary learning".',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['numpy>=1.15'],
    ext_modules=[module],
)
