import os
import yaml

from glob import glob


# This huge string is dumped directly to a text file, creating the default config as is
default_config = """
# Paths can be relative or absolute, but the root needs to already be created
# Any subfolder (i.e. each subject has its own folder) will be created accordingly
# You *need* to edit the first two lines to specify where your data files are and where the results will be output
# Remember that absolute or relative paths are both valid
path: /root/path/to/all/my/data
outpath: /output/path/to//my/new/data
outfilename: harmonization_dictionary.npy

# If an output dataset already exists, we do not overwrite it by default
overwrite: False

# If globbing is allowed, a star expression needs to be a quoted string or it crashes the reader
# See https://en.wikipedia.org/wiki/Glob_(programming) for more info
# or the examples config.yaml files in the harmonization/tests/datasets folder
glob: True
dataname: '*.nii.gz'

# If we glob, the extension of dataname is replaced to create the filenames of the remaining files
# If not, the filename needs to be supplied and will be loaded from the same folder as the data
bval: _bval
bvec: _bvec
maskname: _brain_mask.nii.gz

block_size: 3, 3, 3, 5
block_up: 3, 3, 3, 5
use_std: False
positivity_D: False
positivity_recon: True
fit_intercept: True
center: True

# bias_correct_std: False
fix_mean: False
use_crossval: False

split_b0s: True
b0_threshold: 20
batchsize: 32
niter: 1500
nlambdas: 100

# ncores can be a positive or negative number, which indicates the number of cores to use or to leave free respectively.
ncores: -1 # -1 will use all cores
"""


def write_config(filename):
    with open(filename, "w") as yaml_file:
        yaml_file.write(default_config)


def read_config(filename):
    with open(filename, "r") as yaml_file:
        yaml_content = yaml.safe_load(yaml_file)
    return yaml_content


def get_filenames(path, use_glob, kwargs):
    datasets = []

    if kwargs['dataname'][-7:] == '.nii.gz':
        kwargs['ext'] = '.nii.gz'
    else:
        kwargs['ext'] = os.path.splitext(kwargs['dataname'])[1]

    # Build the list of all datasets and supporting files
    if use_glob:
        files = os.path.join(path, '**/', kwargs['dataname'])
        for name in sorted(glob(files, recursive=True)):
            dataset = {'data': name,
                       'folder': None,
                       'mask': name.replace(kwargs['ext'], kwargs['maskname']),
                       'bval': name.replace(kwargs['ext'], kwargs['bval']),
                       'bvec': name.replace(kwargs['ext'], kwargs['bvec'])}

            datasets += [dataset]
    else:
        paths = list(os.walk(path))

        if len(paths) == 0:
            topdir = paths[0]
        else:
            topdir = paths[0][0]

        for root, dirs, files in paths:
            for name in files:
                if name == kwargs['dataname']:
                    dataset = {'data': os.path.join(root, kwargs['dataname']),
                               'folder': os.path.relpath(root, start=topdir),
                               'mask': os.path.join(root, kwargs['maskname']),
                               'bval': os.path.join(root, kwargs['bval']),
                               'bvec': os.path.join(root, kwargs['bvec'])}

                    datasets += [dataset]

    if len(datasets) == 0:
        error = f'No datasets found with the given config file at {os.path.realpath(path)}'
        raise ValueError(error)

    return datasets
