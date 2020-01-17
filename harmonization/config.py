from __future__ import print_function, division

import os
import yaml

from glob import iglob

# This huge string is dumped directly to a text file, creating the default config as is
default_config = """
# Paths can be relative or absolute, but they need to already be created
path: /user/alberto/Samuel/TO_HARMONIZE
outpath: /user/samuel/Samuel/TO_HARMONIZE
outfilename: belgium_norway.npy

# If globbing is allowed, a star expression needs to be a quoted string or it crashes the reader
glob: True
dataname: '*_FP.nii'

# If we glob, the extension of dataname is replaced to create the filenames of the remaining files
# If not, the filename needs to be supplied and will be loaded from the same folder as the data
maskname: _brain_mask.nii.gz
bval: .bval
bvec: .bvec


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
# If it is None or -1, all cores will be used
ncores: 100
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

    return datasets
