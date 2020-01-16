from __future__ import print_function, division

import numpy as np
import nibabel as nib

from itertools import cycle
from time import time

from sklearn.feature_extraction.image import extract_patches

from nlsam.angular_tools import angular_neighbors
from nlsam.denoiser import greedy_set_finder

from harmonization.tensor_sc import online_DL
from harmonization.recon import depimp_zoom, reconstruct_from_blocks
from harmonization.tensor_sc import solve_l1


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


def rebuild(data, mask, D, block_size, block_up, ncores=-1,
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
    print(new_shape, new_overlap, factor)

    if block_size == block_up:
        D_depimpe = np.copy(D)
    else:
        D_depimpe = depimp_zoom(D, block_size, block_up, zoomarray=False)

    blocks = extract_patches(data, block_size, overlap).reshape(-1, np.prod(block_size))
    del data
    # original_shape = blocks.shape

    # # check if recon is correct / only reconstruct old school way from the blocks to be on the safe side maybe?
    # recon = reconstruct_from_blocks(blocks, new_shape, block_size, block_up[:-1], new_overlap, weights=None)
    # return recon

    # blocks = np.asarray(blocks).reshape(-1, np.prod(block_size))

    # get the variance as blocks
    if variance is not None:
        variance *= mask
        print(variance.shape)
        variance = extract_patches(variance, block_size[:-1], overlap[:-1])
        print(variance.shape)
        # axis = list(range(variance.ndim//2, variance.ndim))
        # variance = np.median(variance, axis=axis)
        # print(variance.shape, np.prod(block_size[:-1]), np.prod(block_size), np.prod(variance.shape))
        variance = np.asarray(variance).reshape(-1, np.prod(block_size[:-1]))
        print(variance.shape)
    # skip empty rows from training since they are probably masked background
    mask = blocks.sum(axis=1) > np.prod(block_size) // 2

    if variance is not None:
        variance = np.median(variance, axis=-1)

        # if we are on an edge, variance can be 0, so truncate those cases as well
        np.logical_and(mask, variance > 0, out=mask)
        variance = variance[mask]
        print(variance.shape, np.sum(variance == 0))

    print(blocks.shape)
    blocks = blocks[mask]
    print(blocks.shape, D.shape, D_depimpe.shape, mask.shape, block_size, block_up, new_shape, new_overlap, 'pre l1 stuff')

    # if center:
    #     blocks_mean = blocks.mean(axis=1, keepdims=True)
    #     blocks -= blocks_mean
    tt = time()
    X_small_denoised, alpha, intercept, _ = solve_l1(blocks, D_depimpe, variance=variance, return_all=True, nlambdas=100, use_joblib=True,
                                                     positivity=positivity, fit_intercept=True, standardize=True, progressbar=True,
                                                     method='fork', ncores=ncores, use_crossval=use_crossval)

    print(X_small_denoised.shape, D_depimpe.shape, alpha.shape, intercept.shape)
    # print(np.min(alpha), np.max(alpha), np.abs(alpha).min(), np.abs(alpha).max())
    # print(np.min(intercept), np.max(intercept), np.abs(intercept).min(), np.abs(intercept).max())
    print('total time was {}'.format(time() - tt))
    # if fix_mean:
    #     mean = blocks.mean(axis=1)
    # else:
    #     mean = None

    # if center:
    #     intercept += blocks_mean

    # # reconstructor = None  # should we put block_up from the original?
    # tt = time()
    # recon = reconstruct_from_indexes(alpha, D, intercept, new_shape, new_overlap, mask, block_size, block_up)  #
    # print('reconstruction took {}'.format(time() - tt))
    # return recon

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
