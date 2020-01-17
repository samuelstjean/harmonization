from __future__ import print_function, division

import numpy as np

from itertools import product

from scipy.ndimage.interpolation import zoom
from sklearn.feature_extraction.image import extract_patches
from harmonization.solver import online_DL, solve_l1


def upsampler_3D(data, variance=None, block_size=(3,3,3), block_up=(5,5,5),
                 mask=None, dtype=np.float64, ncores=-1, params=None):

    factor = np.array(block_up) / np.array(block_size)[:-1]

    if mask is None:
        mask = np.ones(data.shape[:3])

    if data.ndim == 4:
        overlap = (1, 1, 1, data.shape[-1])
        new_overlap = overlap
        new_size = np.array(block_up, dtype=np.int16)
        new_shape = (np.array(data.shape[:-1]) * factor).astype(np.int16)

        new_size = tuple(new_size) + (data.shape[-1],)
        new_shape = tuple(new_shape) + (data.shape[-1],)
    else:
        overlap = (1, 1, 1)

        new_size = np.ceil(np.array(block_size) * factor).astype(np.int16)
        new_shape = np.ceil(np.array(data.shape) * factor).astype(np.int16)
        new_overlap = np.ceil(np.array(overlap) * factor).astype(np.int16)

    mask_col = extract_patches(mask, new_size[:-1], overlap[:-1])
    dims = tuple(range(mask_col.ndim//2, mask_col.ndim))
    shape = mask_col.shape[mask_col.ndim//2:]
    train_idx = np.sum(mask_col, axis=dims) > (np.prod(shape)//2)
    trainer = extract_patches(data, new_size, overlap)

    train_data = trainer[train_idx]
    axis = tuple(range(1, train_data.ndim))

    if variance is None:
        variance_large = None
    else:
        mask_large = extract_patches(mask, new_size[:-1], overlap[:-1])
        shape = mask_large.shape[mask_large.ndim//2:]
        dims = tuple(range(mask_large.ndim//2, mask_large.ndim))
        variance_mask = np.sum(mask_large, axis=dims) > (np.prod(shape)//2)
        variance_large = extract_patches(variance, new_size[:-1], overlap[:-1])[variance_mask]
        axis = tuple(range(1, variance_large.ndim))
        variance_large = np.median(variance_large, axis=axis)

    n_atoms = int(np.prod(block_size) * 2)

    D = online_DL(train_data, ncores=ncores, positivity=True, fit_intercept=True, standardize=True,
                  nlambdas=100, niter=150, batchsize=256, n_atoms=n_atoms, variance=variance_large, progressbar=True)

    params['D'] = D

    D_depimpe = depimp_zoom(D, block_size, block_up)
    del train_data, mask_col, shape

    mask_small = extract_patches(mask, block_size[:-1], new_overlap[:-1])
    dims = tuple(range(mask_small.ndim//2, mask_small.ndim))
    shape = mask_small.shape[mask_small.ndim//2:]

    train_small = np.sum(mask_small, axis=dims) > (np.prod(shape)//2)
    X_small_full = extract_patches(data, block_size, new_overlap)
    X_small = X_small_full[train_small]
    axis = tuple(range(1, X_small.ndim))

    if variance is not None:
        mask_small = extract_patches(mask, block_size[:-1], new_overlap[:-1])
        shape = mask_small.shape[mask_small.ndim//2:]
        dims = tuple(range(mask_small.ndim//2, mask_small.ndim))
        variance_mask = np.sum(mask_small, axis=dims) > (np.prod(shape)//2)
        variance_small = extract_patches(variance, block_size[:-1], new_overlap[:-1])[variance_mask]
        axis = range(1, variance_small.ndim)
        variance_small = np.median(variance_small, axis=axis)

    X_small_denoised, alpha, intercept = solve_l1(X_small, D_depimpe, variance=variance_small, return_all=True,
                                                  positivity=True, nlambdas=100, fit_intercept=True, standardize=True, progressbar=True)
    del X_small_denoised

    indexes = train_small.ravel()
    upsampled = np.zeros((indexes.shape[0],) + new_size, dtype=np.float32)
    full_weights = np.ones(indexes.shape[0], dtype=np.float32)
    weights = 1.
    stuff = np.dot(D, alpha).T + intercept

    upsampled[indexes] = stuff.reshape((stuff.shape[0],) + new_size)
    full_weights[indexes] = weights
    del stuff, weights, indexes

    output = reconstruct_from_blocks(upsampled, new_shape, block_size, block_up, new_overlap, weights=full_weights)
    return output


def depimp_zoom(D, block_size, block_up, order=1):

    block_size = np.array(block_size)
    block_up = np.array(block_up)

    if (len(block_up) == len(block_size)):
        factor = block_size / block_up
        zoomer = (*factor, 1)
    else:
        factor = block_size[:-1] / block_up
        zoomer = (*factor, 1, 1)

    # if we have a 4D block array and different last dimension, subsample it
    # if (len(block_up) == len(block_size)) and (block_up[-1] != block_size[-1]):
    size = tuple(block_up)

    if (len(block_up) - 1) == len(block_size):
        size = size + (block_size[-1],)

    reshaped = D.reshape(size + (-1,))
    D_depimpe = zoom(reshaped, zoomer, order=order).reshape(np.prod(block_size), -1)

    return D_depimpe


def reconstruct_from_indexes(alpha, D, intercept, new_shape, new_overlap, mask,
                             block_size, block_up, small_recon=None, patch_mean=None):

    if len(block_size) == len(block_up):
        factor = np.divide(block_up, block_size)
        last = (block_up[-1],)
        block_up = block_up[:-1]
    else:
        last = (block_size[-1],)
        factor = tuple(np.divide(block_up, block_size[:-1])) + last

    i_h, i_w, i_l = new_shape[:3]
    p_h, p_w, p_l = block_up

    img = np.zeros(new_shape, dtype=np.float32)
    img = np.pad(img, [(0, p_h), (0, p_w), (0, p_l), (0, 0)], 'constant', constant_values=(0, 1))
    div = np.full(new_shape, 1e-15, dtype=np.float32)
    div = np.pad(div, [(0, p_h), (0, p_w), (0, p_l), (0, 0)], 'constant', constant_values=(0, 1))

    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    n_l = i_l - p_l + 1

    frac, nup = np.modf(np.divide(block_up, block_size[:-1]))
    int_frac = np.ceil(frac).astype(np.int32)
    nup = np.array(nup + int_frac, dtype=np.int32)

    valid = (np.arange(0, block_up[0], nup[0]),
             np.arange(0, block_up[1], nup[1]),
             np.arange(0, block_up[2], nup[2]))

    plus = [None, None, None]

    for idx in range(3):
        if len(valid[idx]) < block_size[idx]:
            candidate = np.arange(block_up[idx])[(len(valid[idx]) - block_size[idx]):]

            # if we already have filled the last index (like in 4/5), we have to find a valid location for it
            while any(np.in1d(candidate, valid[idx])):
                candidate -= 1

            plus[idx] = candidate
        else:
            plus[idx] = ()

    valid = (np.append(valid[0], plus[0]),
             np.append(valid[1], plus[1]),
             np.append(valid[2], plus[2]))

    step = ([slice(i, i + p_h) for i in range(0, n_h + int_frac[0], new_overlap[0]) if i % block_up[0] in valid[0]],
            [slice(j, j + p_w) for j in range(0, n_w + int_frac[1], new_overlap[1]) if j % block_up[1] in valid[1]],
            [slice(k, k + p_l) for k in range(0, n_l + int_frac[2], new_overlap[2]) if k % block_up[2] in valid[2]])

    ijk = product(*step)
    stuff = np.zeros(D.shape[0], dtype=np.float64)

    p = 0
    weights = 1
    print(stuff.shape, D.shape, alpha.shape, 'stuff')

    for idx, (i, j, k) in enumerate(ijk):
        if mask[idx]:

            stuff[:] = np.dot(D, alpha[:, p]) + intercept[p]

            img[i, j, k] += np.reshape(stuff * weights, block_up + last)
            div[i, j, k] += weights
            p += 1

    return img[:-p_h, :-p_w, :-p_l] / div[:-p_h, :-p_w, :-p_l]


def reconstruct_from_blocks(patches, image_size, block_size, block_up, new_overlap, weights=None):

    i_h, i_w, i_l = image_size[:3]

    if len(patches.shape[1:]) == 3:
        p_h, p_w, p_l = patches.shape[1:]
    else:
        p_h, p_w, p_l = patches.shape[1:-1]

    img = np.zeros(image_size, dtype=np.float32)
    img = np.pad(img, [(0, p_h), (0, p_w), (0, p_l), (0, 0)], 'constant', constant_values=(0, 1))
    div = np.full(image_size, 1e-15, dtype=np.float32)
    div = np.pad(div, [(0, p_h), (0, p_w), (0, p_l), (0, 0)], 'constant', constant_values=(0, 1))

    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    n_l = i_l - p_l + 1

    frac, nup = np.modf(np.divide(block_up, block_size[:-1]))
    int_frac = np.ceil(frac).astype(np.int32)
    nup = np.array(nup + int_frac, dtype=np.int32)

    valid = (np.arange(0, block_up[0], nup[0]),
             np.arange(0, block_up[1], nup[1]),
             np.arange(0, block_up[2], nup[2]))

    plus = [None, None, None]

    for idx in range(3):
        if len(valid[idx]) < block_size[idx]:
            candidate = np.arange(block_up[idx])[(len(valid[idx]) - block_size[idx]):]

            # if we already have filled the last index (like in 4/5), we have to find a valid location for it
            while any(np.in1d(candidate, valid[idx])):
                candidate -= 1

            plus[idx] = candidate
        else:
            plus[idx] = ()

    valid = (np.append(valid[0], plus[0]),
             np.append(valid[1], plus[1]),
             np.append(valid[2], plus[2]))

    step = ([slice(i, i + p_h) for i in range(0, n_h + int_frac[0], new_overlap[0]) if i % block_up[0] in valid[0]],
            [slice(j, j + p_w) for j in range(0, n_w + int_frac[1], new_overlap[1]) if j % block_up[1] in valid[1]],
            [slice(k, k + p_l) for k in range(0, n_l + int_frac[2], new_overlap[2]) if k % block_up[2] in valid[2]])

    ijk = list(product(*step))

    for p, (i, j, k) in zip(patches, ijk):
        img[i, j, k] += p
        div[i, j, k] += 1

    out = img[:-p_h, :-p_w, :-p_l] / div[:-p_h, :-p_w, :-p_l]
    return out
