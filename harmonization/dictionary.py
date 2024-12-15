from __future__ import print_function, division

import os
import numpy as np
import nibabel as nib

from itertools import cycle
from ast import literal_eval
from time import time

# It's a private function in 0.24 now
try:
    from sklearn.feature_extraction.image import _extract_patches as extract_patches
except ImportError:
    from sklearn.feature_extraction.image import extract_patches

from nlsam.angular_tools import angular_neighbors
from nlsam.denoiser import greedy_set_finder

from harmonization.solver import online_DL, solve_l1
from harmonization.recon import depimp_zoom, reconstruct_from_blocks


def get_global_D(datasets, outfilename, block_size, ncores=None, batchsize=32, niter=500,
                 use_std=False, positivity=False, fit_intercept=True, center=True, nlambdas=100,
                 b0_threshold=20, split_b0s=True, **kwargs):

    # get the data shape so we can preallocate some arrays
    # we also have to assume all datasets have the same 3D shape obviously
    shape = nib.load(datasets[0]['data']).header.get_data_shape()
    current_block_size = literal_eval(block_size)

    n_atoms = int(np.prod(current_block_size) * 2)
    b0_block_size = tuple(current_block_size[:-1]) + ((current_block_size[-1] + 1,))
    overlap = b0_block_size
    to_denoise = np.empty(shape[:-1] + (current_block_size[-1] + 1,), dtype=np.float32)

    train_list = []
    variance_large = []

    for filename in datasets:

        print('Now feeding dataset {}'.format(filename['data']))

        mask = nib.load(filename['mask']).get_fdata(caching='unchanged').astype(np.bool)
        data = nib.load(filename['data']).get_fdata(caching='unchanged').astype(np.float32) * mask[..., None]
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
                except IOError:
                    print('Volume {} not found!'.format(filename['std']))
                    variance = [None]
            else:
                variance = [None]

            # check to build with np.r_ the whole list from stringnames instead
            train_list += [patches]
            variance_large += list(variance)

        del data, mask, patches, variance

    print('Fed everything in')

    lengths = [l.shape[0] for l in train_list]
    train_data = np.empty((np.sum(lengths), np.prod(b0_block_size)))

    step = 0
    for i in range(len(train_list)):
        length = lengths[i]
        idx = slice(step, step + length)
        train_data[idx] = train_list[i].reshape(-1, np.prod(b0_block_size))
        step += length

    del train_list

    # we have variance as a N elements list - so check one element to see if it's an array
    if variance_large[0] is not None:
        variance_large = np.asarray(variance_large).ravel()
    else:
        variance_large = None

    # savename = 'Dic_' + outfilename + '_size_{}.npy'.format(block_size).replace(' ', '')

    D = online_DL(train_data, ncores=ncores, positivity=positivity, fit_intercept=fit_intercept, standardize=True,
                  nlambdas=nlambdas, niter=niter, batchsize=batchsize, n_atoms=n_atoms, variance=variance_large,
                  progressbar=True, use_joblib=True)

    return D


def rebuild(data, mask, D, block_size, block_up, ncores=-1, nlambdas=100,
            positivity=False, variance=None, fix_mean=True, fit_intercept=False, use_crossval=False):

    data = data * mask[..., None]

    if len(block_size) == len(block_up):
        last = block_up[-1]
    else:
        last = block_size[-1]

    factor = np.divide(block_up, block_size)
    new_shape = (int(data.shape[0] * factor[0]),
                 int(data.shape[1] * factor[1]),
                 int(data.shape[2] * factor[2]),
                 last)
    overlap = (1, 1, 1, last)
    new_overlap = overlap

    if block_size == block_up:
        D_depimpe = np.copy(D)
    else:
        D_depimpe = depimp_zoom(D, block_size, block_up)

    blocks = extract_patches(data, block_size, overlap).reshape(-1, np.prod(block_size))
    del data

    # get the variance as blocks
    if variance is not None:
        variance *= mask
        variance = extract_patches(variance, block_size[:-1], overlap[:-1])
        variance = np.asarray(variance).reshape(-1, np.prod(block_size[:-1]))

    # skip empty rows from training since they are probably masked background
    mask = blocks.sum(axis=1) > np.prod(block_size) // 2

    if variance is not None:
        variance = np.median(variance, axis=-1)

        # if we are on an edge, variance can be 0, so truncate those cases as well
        np.logical_and(mask, variance > 0, out=mask)
        variance = variance[mask]

    blocks = blocks[mask]

    tt = time()
    X_small_denoised, alpha, intercept, _ = solve_l1(blocks, D_depimpe, variance=variance, return_all=True, nlambdas=nlambdas, use_joblib=True,
                                                     positivity=positivity, fit_intercept=True, standardize=True, progressbar=True, verbose=5,
                                                     ncores=ncores, use_crossval=use_crossval)

    print('total time was {}'.format(time() - tt))

    # we actually only want alpha and intercept in this step and throw out X if we do upsampling normally
    # for ease of use reason, we now use directly X for the reconstruction, but we should
    # 1. multiply back X
    # 2a. feed X in an empty array with a mask
    #
    #   OR
    #
    # 2b. reconstruct the array and average things internally according to a mask

    tt = time()
    X_final = np.zeros(((mask.shape[0],) + block_up), dtype=np.float32)

    if block_size != block_up:
        X_small_denoised = np.dot(D, alpha).T + intercept

    print('multiply time was {}'.format(time() - tt))
    tt = time()

    X_final[mask] = X_small_denoised.reshape(-1, *block_up)
    recon = reconstruct_from_blocks(X_final, new_shape, block_size, block_up[:-1], new_overlap, weights=None)
    print('recon time was {}'.format(time() - tt))

    del X_final, mask
    return recon


def harmonize_my_data(dataset, kwargs):

    outpath = kwargs['outpath']
    path_D = kwargs['outfilename']
    center = kwargs['center']
    block_up = literal_eval(kwargs['block_up'])
    block_size = literal_eval(kwargs['block_size'])
    positivity = kwargs['positivity']
    fit_intercept = kwargs['fit_intercept']
    fix_mean = kwargs['fix_mean']
    use_crossval = kwargs['use_crossval']
    split_b0s = kwargs['split_b0s']
    b0_threshold = kwargs['b0_threshold']
    ncores = kwargs['ncores']
    nlambdas = kwargs['nlambdas']
    ext = kwargs['ext']

    print('Now rebuilding {}'.format(dataset['data']))
    D = np.load(path_D)
    predicted_D = path_D.replace('.npy', '')

    output_filename = dataset['data'].replace(ext, '_predicted_' + predicted_D + ext)
    output_filename = output_filename.replace(ext, '_recon' + ext)
    output_filename = os.path.join(outpath, os.path.basename(output_filename))

    # Create subfolders tree if it does not exist
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(outpath, exist_ok=True)

    if os.path.isfile(output_filename):
        print('File already exists! Skipping {}'.format(output_filename))
    else:
        vol = nib.load(dataset['data'])
        data = vol.get_fdata(caching='unchanged', dtype=np.float32)
        affine = vol.affine
        header = vol.header

        mask = nib.load(dataset['mask']).get_fdata(caching='unchanged').astype(np.bool)
        bvals = np.loadtxt(dataset['bval'])
        bvecs = np.loadtxt(dataset['bvec'])

        if np.shape(bvecs)[0] == 3:
            bvecs = bvecs.T

        if center:
            # we need to pull down the 3D volume mean for the upsampling part to make sense though
            # We use nanmean because we implicitly exclude voxels with a value of 0 from the mean this way
            data_mean = np.nanmean(np.where(data != 0, data, np.nan), axis=(0, 1, 2))
            data -= data_mean

        variance = None

        # number of blocks is without b0, so we +1 the last dimension
        if len(block_up) < data.ndim:
            current_block_size = block_size[:-1] + (block_size[-1] + 1,)
            current_block_up = block_up + (block_size[-1] + 1,)
        else:
            current_block_size = block_size[:-1] + (block_size[-1] + 1,)
            current_block_up = block_up[:-1] + (block_up[-1] + 1,)

        print('Output filename is {}'.format(output_filename))

        factor = np.divide(current_block_up, current_block_size)

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

        neighbors = angular_neighbors(sym_bvecs, current_block_size[-1] - 2) % data.shape[-1]
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

        # Stuff happens here / we only pimp up dwis, not b0s
        # actually we can't really pimp 4D stuff, SH are there for that
        predicted_size = (int(data.shape[0] * factor[0]),
                          int(data.shape[1] * factor[1]),
                          int(data.shape[2] * factor[2]),
                          data.shape[-1])

        predicted = np.zeros(predicted_size, dtype=np.float32)
        divider = np.zeros(predicted.shape[-1])

        # Put all idx + b0 in this array in each iteration
        to_denoise = np.empty(data.shape[:-1] + (current_block_size[-1],), dtype=np.float64)

        for i, idx in enumerate(indexes, start=1):
            b0_loc = tuple((next(split_b0s_idx),))
            to_denoise[..., 0] = data[..., b0_loc].squeeze()
            to_denoise[..., 1:] = data[..., idx]
            divider[list(b0_loc + idx)] += 1

            print('Now rebuilding volumes {} / block {} out of {}.'.format(b0_loc + idx, i, len(indexes)))
            predicted[..., b0_loc + idx] += rebuild(to_denoise,
                                                    mask,
                                                    D,
                                                    block_size=current_block_size,
                                                    block_up=current_block_up,
                                                    ncores=ncores,
                                                    positivity=positivity,
                                                    fix_mean=fix_mean,
                                                    nlambdas=nlambdas,
                                                    # center=center,
                                                    fit_intercept=fit_intercept,
                                                    use_crossval=use_crossval,
                                                    variance=variance)
        predicted /= divider

        if center:
            predicted[mask] += data_mean

        # clip negatives, which happens at the borders
        predicted.clip(min=0., out=predicted)

        imgfile = nib.Nifti1Image(predicted, affine, header)
        nib.save(imgfile, output_filename)
