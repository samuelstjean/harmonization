from __future__ import print_function, division

import sys
import os

from glob import iglob
from ast import literal_eval

import numpy as np

from harmonization.config import get_arguments, write_config
from harmonization.dictionary import get_global_D


def main():

    if len(sys.argv) == 1:
        usage = 'Usage : everything is set in config.yaml, that is the only input'
        print(usage)
        sys.exit(1)

    config = sys.argv[1]

    if len(sys.argv) == 3:
        path = sys.argv[2]
        if config == 'write':
            write_config(path)
        else:
            error = 'You need to pass the keyword [write] as an argument followed by a filename to write the default config, but you passed {}'.format(sys.argv[1:])
            raise ValueError(error)

    kwargs = get_arguments(config)

    # we need to do a few special things for some args
    path = kwargs.pop('path')
    use_glob = kwargs.pop('glob')
    # dataname = kwargs.pop('dataname')
    outfilename = kwargs.pop('outfilename')
    # kwargs['ncores'] = literal_eval(kwargs['ncores'])
    kwargs['block_size'] = literal_eval(kwargs['block_size'])

    # Allow the all keyword to feed in all the scanners
    # if it's regular scanner, we put it in a list to parse it properly afterwards
    # if scanner.lower() == 'all/st':
    #     scanners = 'GE/st', 'Prisma/st', 'Connectom/st'
    # elif scanner.lower() == 'all/sa':
    #     scanners = 'Prisma/sa', 'Connectom/sa'
    # else:
    #     scanners = [scanner]

    datasets = []
    if use_glob:
        files = os.path.join(path, kwargs['dataname'])
        for name in iglob(files):
            dataset = {'data': name,
                       'mask': name.replace('.nii', kwargs['maskname']),
                       'bval': name.replace('.nii', kwargs['bval']),
                       'bvec': name.replace('.nii', kwargs['bvec'])}

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

    # for dataset in datasets:

        # mask_filename = dataset[:-4] + '_mask.nii.gz'
        # mppca_filename = dataset[:-4] + '_mppca.nii.gz'
        # std_filename = dataset[:-4] + '_std.nii.gz'
        # brain_suffix = dataset[:-4] + '_brain'

        # if os.path.isfile(mask_filename):
        #     print('File already exists! Skipping {}'.format(mask_filename))
        # else:
        #     subprocess.call('bet2 {0} {1} -m -f 0.1'.format(dataset, brain_suffix), shell=True)
        #     subprocess.call('mv {0}_mask.nii.gz {1}'.format(brain_suffix, mask_filename), shell=True)

        # if os.path.isfile(std_filename):
        #     print('File already exists! Skipping {}'.format(std_filename))
        # else:
            # subprocess.call('dwidenoise {} {} -noise {} -mask {}'.format(dataset, mppca_filename, std_filename, mask_filename), shell=True)

    D = get_global_D(datasets, outfilename, **kwargs)

    # filename = outfilename + '_size_{}_std_{}_pos_{}'.format(kwargs['block_size'], kwargs['use_std'], kwargs['positivity'])
    # filename = filename.replace(' ', '').replace('(', '').replace(')', '').replace(',', '_')
    np.save(outfilename, D)


if __name__ == "__main__":
    main()
