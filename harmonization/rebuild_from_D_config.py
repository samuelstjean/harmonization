from __future__ import print_function, division

import sys
import os

from glob import iglob
from harmonization.config import get_arguments
from harmonization.dictionary import harmonize_my_data


def main():

    if len(sys.argv) == 1:
        usage = 'Usage : everything is set in config.yaml, that is the only input'
        # usage = 'Usage : path_data path_D scanner to_scanner block_size block_up use_std positivity'
        print(usage)
        sys.exit(1)

    config = sys.argv[1]
    kwargs = get_arguments(config)

    # path, path_D, scanner, to_scanner, block_size, block_up, use_std, positivity = sys.argv[1:]
    path = kwargs.pop('path')
    # path_D = kwargs['outfilename']
    use_glob = kwargs.pop('glob')

    # outpath = kwargs.pop('outpath')
    # center = kwargs['center']
    # block_up = literal_eval(kwargs['block_up'])
    # block_size = literal_eval(kwargs['block_size'])
    # # use_std = kwargs['use_std']
    # positivity = kwargs['positivity']
    # fit_intercept = kwargs['fit_intercept']
    # center = kwargs['center']
    # fix_mean = kwargs['fix_mean']
    # use_crossval = kwargs['use_crossval']
    # split_b0s = kwargs['split_b0s']
    # b0_threshold = kwargs['b0_threshold']
    # # batchsize = kwargs['batchsize']
    # # niter = kwargs['niter']
    # ncores = kwargs['ncores']

    # path = os.path.expanduser(path)
    # path_D = os.path.expanduser(path_D)

    # block_size = literal_eval(block_size)
    # block_up = literal_eval(block_up)
    # use_std = use_std.lower() == 'true'
    # positivity = positivity.lower() == 'true'

    # ncores = 100
    # # N = 1  # we should find a way to actually estimate that
    # # bias_correct_std = False
    # fix_mean = False
    # # reweighting = True
    # # fit_intercept = False
    # fit_intercept = True
    # # center = False
    # center = True
    # use_crossval = True

    # D = np.load(path_D)
    datasets = []
    # to_datasets = []
    # print(D.min(), D.max())

    # Allow the all keyword to feed in all the scanners
    # if it's regular scanner, we put it in a list to parse it properly afterwards
    # if scanner.lower() == 'all/st':
    #     scanners = 'GE/st', 'Prisma/st', 'Connectom/st'
    # elif scanner.lower() == 'all/sa':
    #     scanners = 'Prisma/sa', 'Connectom/sa'
    # else:
    #     scanners = [scanner]

    # Build the list of all datasets ans supporting files
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
                    filename = os.path.join(root, name)
                    dataset = {'data': filename,
                               'mask': filename.replace('.nii', kwargs['maskname']),
                               'bval': filename.replace('.nii', kwargs['bval']),
                               'bvec': filename.replace('.nii', kwargs['bvec'])}
                    datasets += [dataset]

            # if (name == 'dwi.nii' or name == 'dwi_fw.nii') and to_scanner.lower() in root.lower():
            #     to_datasets += [os.path.join(root, name)]

    # if len(datasets) != len(to_datasets):
    #     raise ValueError('Size mismatch between lists! {}, {}'.format(len(datasets), len(to_datasets)))

    # Actually harmonize each dataset by itself
    for dataset in datasets:
        harmonize_my_data(dataset, kwargs)


if __name__ == "__main__":
    main()
