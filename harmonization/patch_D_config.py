from __future__ import print_function, division

import sys
import os

from glob import iglob
# import subprocess

import numpy as np
import nibabel as nib

# from multiprocessing import cpu_count
from sklearn.feature_extraction.image import extract_patches
from ast import literal_eval
from itertools import cycle

from nlsam.angular_tools import angular_neighbors
from nlsam.denoiser import greedy_set_finder
from create_config import get_arguments, write_config
from tensor_sc import online_DL


def get_global_D(datasets, outfilename, block_size, ncores=None, batchsize=32, niter=500,
                 use_std=False, positivity=False, fit_intercept=True, center=True,
                 b0_threshold=20, split_b0s=True, **kwargs):

    # get the data shape so we can preallocate some arrays
    # we also have to assume all datasets have the same 3D shape obviously
    shape = nib.load(datasets[0]['data']).header.get_data_shape()

    # if len(block_size) < len(shape):
    #     # In the event that we only give out a 3D size and that we have b0s with different TE/TR, we remove those volumes
    #     # Therefore, we need to look at the total number of kept volumes, rather than the shape, to specify the last dimension properly
    #     last_shape = get_indexer(datasets[0]).sum() - 1  # We subtract 1 because block_size adds one b0 to it later down
    #     current_block_size = block_size + (last_shape,)
    #     print('Using full 4D stuff')
    # else:
    current_block_size = block_size

    n_atoms = int(np.prod(current_block_size) * 2)
    b0_block_size = tuple(current_block_size[:-1]) + ((current_block_size[-1] + 1,))
    overlap = b0_block_size
    to_denoise = np.empty(shape[:-1] + (current_block_size[-1] + 1,), dtype=np.float32)

    train_list = []
    variance_large = []

    for filename in datasets:

        print('Now feeding dataset {}'.format(filename['data']))

        # indexer = get_indexer(filename)
        mask = nib.load(filename['mask']).get_fdata(caching='unchanged').astype(np.bool)
        data = nib.load(filename['data']).get_fdata(caching='unchanged').astype(np.float32) * mask[..., None]
        # data = data[..., indexer]
        bvals = np.loadtxt(filename['bval'])
        bvecs = np.loadtxt(filename['bvec'])

        if np.shape(bvecs)[0] == 3:
            bvecs = bvecs.T

        b0_loc = np.where(bvals <= b0_threshold)[0]
        dwis = np.where(bvals > b0_threshold)[0]
        num_b0s = len(b0_loc)

        # We also convert bvecs associated with b0s to exactly (0,0,0), which
        # is not always the case when we hack around with the scanner.
        bvecs = np.where(bvals[:, None] <= b0_threshold, 0, bvecs)

        # Average all b0s if we don't split them in the training set
        if num_b0s > 1 and not split_b0s:
            num_b0s = 1
            data[..., b0_loc] = np.mean(data[..., b0_loc], axis=-1, keepdims=True)

        # Split the b0s in a cyclic fashion along the training data
        # If we only had one, cycle just return b0_loc indefinitely,
        # else we go through all indexes.
        np.random.shuffle(b0_loc)
        split_b0s_idx = cycle(b0_loc)
        sym_bvecs = np.vstack((bvecs, -bvecs))

        neighbors = angular_neighbors(sym_bvecs, current_block_size[-1] - 1) % data.shape[-1]
        neighbors = neighbors[:data.shape[-1]]  # everything was doubled for symmetry

        full_indexes = [(dwi,) + tuple(neighbors[dwi]) for dwi in range(data.shape[-1]) if dwi in dwis]
        indexes = greedy_set_finder(full_indexes)

        # If we have more b0s than indexes, then we have to add a few more blocks since
        # we won't do a full cycle. If we have more b0s than indexes after that, then it breaks.
        if num_b0s > len(indexes):
            the_rest = [rest for rest in full_indexes if rest not in indexes]
            indexes += the_rest[:(num_b0s - len(indexes))]

        if num_b0s > len(indexes):
            error = ('Seems like you still have more b0s {} than available blocks {},'
                     ' either average them or deactivate subsampling.'.format(num_b0s, len(indexes)))
            raise ValueError(error)

        # whole global centering
        if center:
            data -= data.mean(axis=-1, keepdims=True)

        for i, idx in enumerate(indexes):
            b0_loc = tuple((next(split_b0s_idx),))

            # if we mix datasets, then we may need to change the storage array size
            if to_denoise.shape[:-1] != data.shape[:-1]:
                del to_denoise
                to_denoise = np.empty(data.shape[:-1] + (current_block_size[-1] + 1,), dtype=np.float32)

            to_denoise[..., 0] = data[..., b0_loc].squeeze()
            to_denoise[..., 1:] = data[..., idx]

            patches = extract_patches(to_denoise, b0_block_size, overlap)
            axis = tuple(range(patches.ndim//2, patches.ndim))
            mask_patch = np.sum(patches > 0, axis=axis) > np.prod(b0_block_size) // 2
            patches = patches[mask_patch].reshape(-1, np.prod(b0_block_size))

            if use_std:
                try:
                    variance = nib.load(filename['std']).get_data()**2 * mask
                    variance = np.broadcast_to(variance[..., None], data.shape)
                    variance = extract_patches(variance, b0_block_size, overlap)
                    axis = tuple(range(variance.ndim//2, variance.ndim))
                    variance = np.median(variance, axis=axis)[mask_patch].ravel()
                    print('variance shape', variance.shape)
                except IOError:
                    print('Volume {} not found!'.format(filename['std']))
                    variance = [None]
            else:
                variance = [None]

            # check to build with np.r_ the whole list from stringnames instead
            train_list += [patches]
            variance_large += list(variance)
            # train_data.extend(patches)
            # variance_large.extend(variance)

        print('train', len(train_list), data.shape, b0_block_size, overlap, patches.shape)

        del data, mask, patches, variance

    print('Fed everything in')

    lengths = [l.shape[0] for l in train_list]
    train_data = np.empty((np.sum(lengths), np.prod(b0_block_size)))
    print(train_data.shape)

    step = 0
    for i in range(len(train_list)):
        length = lengths[i]
        idx = slice(step, step + length)
        train_data[idx] = train_list[i].reshape(-1, np.prod(b0_block_size))
        step += length

    del train_list

    # if center:
    #     train_data -= train_data.mean(axis=1, keepdims=True)

    # we have variance as a N elements list - so check one element to see if it's an array
    if variance_large[0] is not None:
        variance_large = np.asarray(variance_large).ravel()
    else:
        variance_large = None

    savename = 'Dic_' + outfilename + '_size_{}.npy'.format(block_size).replace(' ', '')

    D = online_DL(train_data, ncores=ncores, positivity=positivity, fit_intercept=fit_intercept, standardize=True,
                  nlambdas=100, niter=niter, batchsize=batchsize, n_atoms=n_atoms, variance=variance_large,
                  progressbar=True, disable_mkl=True, saveback=savename, use_joblib=False)

    return D


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
