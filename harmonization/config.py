from __future__ import print_function, division

import os
import yaml

from glob import iglob


# This huge string is dumped directly to a text file, creating the default config as is
default_config = """
# Paths can be relative or absolute, but the root needs to already be created
# Any subfolder (i.e. each subject has its own folder) will be created accordingly
path: /root/path/to/all/my/data
outpath: /output/path/to//my/new/data
outfilename: harmonization_dictionary.npy

# If globbing is allowed, a star expression needs to be a quoted string or it crashes the reader
glob: True
dataname: '*_.nii.gz'

# If we glob, the extension of dataname is replaced to create the filenames of the remaining files
# If not, the filename needs to be supplied and will be loaded from the same folder as the data
bval: .bval
bvec: .bvec
maskname: _brain_mask.nii.gz

block_size: 3, 3, 3, 5
block_up: 3, 3, 3, 5
use_std: False
positivity: False
fit_intercept: True
center: True

# bias_correct_std: False
fix_mean: False
use_crossval: False

split_b0s: True
b0_threshold: 20
batchsize: 32
niter: 1500

# ncores can be a positive or negative number, which indicates the number of cores to use or to leave free respectively.
# ncores = -1 will use all cores
ncores: -1
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
        files = os.path.join(path, kwargs['dataname'])
        for name in iglob(files):
            dataset = {'data': name,
                       'mask': name.replace(kwargs['ext'], kwargs['maskname']),
                       'bval': name.replace(kwargs['ext'], kwargs['bval']),
                       'bvec': name.replace(kwargs['ext'], kwargs['bvec'])}

            datasets += [dataset]
    else:
        for root, dirs, files in os.walk(path):
            dirs.sort()
            for name in files:
                if name == kwargs['dataname']:
                    dataset = {'data': os.path.join(root, kwargs['dataname']),
                               'mask': os.path.join(root, kwargs['maskname']),
                               'bval': os.path.join(root, kwargs['bval']),
                               'bvec': os.path.join(root, kwargs['bvec'])}

                    datasets += [dataset]

    if len(datasets) == 0:
        raise ValueError('No datasets found with the given config file at {}'.format(path))

    return datasets
