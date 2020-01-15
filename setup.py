from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='harmonization',
    version='0.1',
    author='Samuel St-Jean',
    author_email='samuel@isi.uu.nl',
    packages=find_packages(),
    # scripts=['scripts/dpr', 'scripts/dpr_make_fancy_graph'],
    url='https://github.com/samuelstjean/harmonization',
    license='GPLV2',
    description='Implementation of "Harmonization of diffusion MRI datasets with adaptive dictionary learning".',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['numpy>=1.15'],
)
