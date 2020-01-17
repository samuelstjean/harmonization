from __future__ import print_function, division

import yaml


# This huge string is dumped directly to a text file, creating the default config as is
default_config = """
path: '/user/alberto/Samuel/TO_HARMONIZE'
outpath: '/user/samuel/Samuel/TO_HARMONIZE'
outfilename: 'belgium_norway.npy'

dataname: '*_FP.nii'
maskname: '_brain_mask.nii.gz'
bval: '.bval'
bvec: '.bvec'

glob: True
block_size: '3, 3, 3, 5'
block_up: '3, 3, 3, 5'
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

ncores: 100 # None = uses everything in this case
"""


def write_config(filename):
    with open(filename, "w") as yaml_file:
        yaml_file.write(default_config)


def read_config(filename):
    with open(filename, "r") as yaml_file:
        yaml_content = yaml.safe_load(yaml_file)
    return yaml_content
