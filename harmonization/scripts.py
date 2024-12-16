import os
import sys

import numpy as np

from harmonization.config import read_config, write_config, get_filenames
from harmonization.dictionary import get_global_D, harmonize_my_data


def main_get_global_D():
    if len(sys.argv) == 1:
        usage = 'Everything is set in config.yaml, that is the only input.\nTo create a default config, pass "write /path/to/file/config.yaml" to create a template example'
        print(usage)
        sys.exit(0)

    config = sys.argv[1]

    if len(sys.argv) == 3:
        path = sys.argv[2]
        if config == 'write':
            write_config(path)
            print(f'Default config written at {os.path.abspath(path)}')
            sys.exit(0)
        else:
            error = f'You need to pass the keyword "write" as an argument followed by a filename to write the default config, but you passed {sys.argv[1:]}'
            raise ValueError(error)

    if len(sys.argv) != 2:
        error = f'The only accepted argument is a config.yaml file, but you passed {sys.argv[1:]}'
        raise ValueError(error)

    kwargs = read_config(config)

    # we need to do a few special things for some args
    path = kwargs.pop('path')
    use_glob = kwargs.pop('glob')
    outfilename = kwargs.pop('outfilename')
    kwargs['positivity'] = kwargs.pop('positivity_D')
    kwargs.pop('positivity_recon')

    datasets = get_filenames(path, use_glob, kwargs)

    D = get_global_D(datasets, outfilename, **kwargs)
    np.save(outfilename, D)


def main_harmonize_my_data():
    if len(sys.argv) == 1:
        usage = 'Everything is set in "config.yaml" and is the only accepted input.'
        print(usage)
        sys.exit(0)

    if len(sys.argv) != 2:
        error = f'The only accepted argument is a "config.yaml" file, but you passed {sys.argv[1:]}'
        raise ValueError(error)

    config = sys.argv[1]
    kwargs = read_config(config)

    path = kwargs.pop('path')
    use_glob = kwargs.pop('glob')
    kwargs.pop('positivity_D')
    kwargs['positivity'] = kwargs.pop('positivity_recon')

    datasets = get_filenames(path, use_glob, kwargs)

    # Actually harmonize each dataset by itself
    for dataset in datasets:
        harmonize_my_data(dataset, kwargs)
