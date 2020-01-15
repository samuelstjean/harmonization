from __future__ import print_function, division

import numpy as np

from time import time
from itertools import product
from multiprocessing import cpu_count

from scipy.ndimage.interpolation import zoom
from tensor_sc import online_DL, solve_l1
from sklearn.feature_extraction.image import extract_patches


def upsampler_3D(data, variance=None, block_size=(3,3,3), block_up=(5,5,5),
                 mask=None, dtype=np.float64, ncores=None, params=None):

    if ncores is None:
        ncores = cpu_count()

    factor = np.array(block_up) / np.array(block_size)[:-1]

    print(block_size, factor, block_up)

    if mask is None:
        mask = np.ones(data.shape[:3])

    if data.ndim == 4:
        overlap = (1, 1, 1, data.shape[-1])
        # overlap = tuple(block_size[:-1]) + (data.shape[-1],)
        # new_overlap = tuple(block_size[:-1]) + (data.shape[-1],)
        # new_overlap = tuple(block_up) + (data.shape[-1],)
        new_overlap = overlap
        # new_overlap = (2, 2, 2, data.shape[-1])

        # new_size = np.array(block_size[:-1]) * factor
        new_size = np.array(block_up, dtype=np.int16)
        new_shape = (np.array(data.shape[:-1]) * factor).astype(np.int16)
        # pad_shape = np.array(data.shape[:-1]) + (np.mod(data.shape[:-1], block_size[:-1]).astype(np.bool) * np.array(block_size[:-1])) - np.mod(data.shape[:-1], block_size[:-1])
        # new_shape = (factor * pad_shape).astype(np.int16)
        # new_shape = np.ceil(np.array(data.shape[:-1]) * new_size // np.array(block_size[:-1]) * new_size).astype(np.int16)
        # new_shape = (new_shape * new_size).astype(np.int16)
        # print(new_shape, data.shape[:-1], factor, 'shape')
        # new_shape = (int(data.shape[0] + 2*nup),
        #              int(data.shape[1] + 2*nup),
        #              int(data.shape[2] + 2*nup))
        # new_shape = np.array(new_shape)

        # new_overlap = np.array(overlap[:-1]) * factor
        # new_overlap = np.array(overlap[:-1])
        # new_overlap = np.array([2,2,2,data.shape[-1]])

        new_size = tuple(new_size) + (data.shape[-1],)
        new_shape = tuple(new_shape) + (data.shape[-1],)
        # new_overlap = tuple(new_overlap.astype(np.int16)) + (data.shape[-1],)
    else:
        overlap = (1, 1, 1)

        new_size = np.ceil(np.array(block_size) * factor).astype(np.int16)
        new_shape = np.ceil(np.array(data.shape) * factor).astype(np.int16)
        new_overlap = np.ceil(np.array(overlap) * factor).astype(np.int16)

    print(block_size, data.shape, overlap, factor)
    print(new_size, new_shape, new_overlap, factor)
    print('train 1', mask.shape, data.shape, new_size[:-1])
    # print(mask.flags)
    # 1/0
    # mask_col = extract_patches(np.broadcast_to(mask[..., None], data.shape), new_size, overlap)
    mask_col = extract_patches(mask, new_size[:-1], overlap[:-1])
    print('train 2', mask_col.shape)
    # 1/0
    dims = tuple(range(mask_col.ndim//2, mask_col.ndim))
    shape = mask_col.shape[mask_col.ndim//2:]
    train_idx = np.sum(mask_col, axis=dims) > (np.prod(shape)//2)
    print(np.sum(mask_col, axis=dims).max(), np.prod(shape), shape)
    # 1/0
    print('train 3', train_idx.shape, train_idx.max(), train_idx.sum(), mask.sum(), mask_col.sum())
    trainer = extract_patches(data, new_size, overlap)
    X_upsampled_shape = np.prod(trainer.shape[:trainer.ndim//2]), np.prod(trainer.shape[trainer.ndim//2:])
    print(X_upsampled_shape, trainer.shape)
    # 1/0
    # trainer_grad = extract_patches(np.sum(np.gradient(data), axis=0), new_size, overlap)

    # train_data = np.concatenate((trainer[train_idx], trainer_grad[train_idx]))
    # train_data = trainer_grad[train_idx]
    train_data = trainer[train_idx]
    # train_idx = train_idx.ravel()
    # print(trainer.shape, train_idx.shape, trainer[train_idx].shape)
    # 1/0
    print(dims, train_data.shape)
    axis = tuple(range(1, train_data.ndim))
    # train_data -= train_data.mean(axis=axis, keepdims=True)

    print('train 4', train_data.shape, trainer.shape, data.shape, new_size)
    print('Constructing big D matrix')
    t = time()

    if variance is None:
        variance_large = None
    else:
        mask_large = extract_patches(mask, new_size[:-1], overlap[:-1])
        shape = mask_large.shape[mask_large.ndim//2:]
        dims = tuple(range(mask_large.ndim//2, mask_large.ndim))
        variance_mask = np.sum(mask_large, axis=dims) > (np.prod(shape)//2)
        # print('variance shape', variance.shape, 0, variance_mask.shape)
        # print(extract_patches(variance, new_size[:-1], overlap[:-1]).shape)
        # print(new_overlap[:-1], new_size[:-1], overlap[:-1])
        # 1/0
        variance_large = extract_patches(variance, new_size[:-1], overlap[:-1])[variance_mask]
        # 1/0
        # variance_large = extract_patches(np.broadcast_to(variance[..., None], data.shape), new_size, overlap)[train_idx]
        print('variance shape', variance_large.shape, train_idx.shape, dims, data.shape)
        axis = tuple(range(1, variance_large.ndim))
        variance_large = np.median(variance_large, axis=axis)
        print('variance shape', variance_large.shape)

    # if False:#'D' in params:
    #     print('found big D, skipping')
    #     D = params['D'].copy()
    # else:
    n_atoms = int(np.prod(block_size) * 2)
    # n_atoms = int(np.prod(new_size) * 2)

    D = online_DL(train_data, ncores=ncores, positivity=True, fit_intercept=True, standardize=True,
                  nlambdas=100, niter=150, batchsize=256, n_atoms=n_atoms, variance=variance_large, progressbar=True)

    print(D.shape)
    params['D'] = D

    print('mean D', np.abs(D).mean())
    print('The D is done, time {}'.format(time() - t), D.shape)

    # D_depimpe = depimp_mean(D, block_size, factor)
    # print(D_depimpe.shape, D.shape, block_size, factor, 'shape 1')

    D_depimpe = depimp_zoom(D, block_size, block_up)
    print(D_depimpe.shape, D.shape, block_size, block_up, factor, 'shape 1')
    del train_data, mask_col, shape

    t = time()
    # padding = tuple(np.array(block_up) - np.array(block_size[:-1])) + (0,)
    # padding = ((padding[0], padding[0]),
    #            (padding[1], padding[1]),
    #            (padding[2], padding[2]),
    #            (padding[3], padding[3]))
    # broad_mask = np.broadcast_to(mask[..., None], data.shape) #np.pad(np.broadcast_to(mask[..., None], data.shape), padding, mode='constant')
    # mask_small = extract_patches(broad_mask, block_size, new_overlap)
    mask_small = extract_patches(mask, block_size[:-1], new_overlap[:-1])
    dims = tuple(range(mask_small.ndim//2, mask_small.ndim))
    shape = mask_small.shape[mask_small.ndim//2:]

    # train_small = np.sum(mask_small, axis=dims) > (np.prod(shape)//2)
    train_small = np.sum(mask_small, axis=dims) > (np.prod(shape)//2)
    # print(padding, data.shape)
    # data = data#np.pad(data, padding, mode='constant')
    # print(padding, data.shape)

    X_small_full = extract_patches(data, block_size, new_overlap)
    # X_small_shape = (np.prod(X_small_full.shape[:X_small_full.ndim//2]), np.prod(shape))
    X_small = X_small_full[train_small]

    # axis = tuple(range(1, X_small.ndim//2 + 1))
    # alpha = np.zeros((D.shape[1], X_small_full.shape[0]))
    # print(X_small_full.shape, D_depimpe.shape, D.shape, 'shape 2')
    # dims = tuple(range(X_small.ndim//2, X_small.ndim))
    axis = tuple(range(1, X_small.ndim))
    X_mean = X_small.mean(axis=axis)[:, None]
    # X_small -= X_mean
    # print(X_small.shape, X_mean.shape, axis, 'axis')
    # return 1
    # mkl.set_num_threads(1)
    if variance is not None:
        # broad_mask = mask #np.pad(mask, padding[:-1], mode='constant')
        # mask_small = extract_patches(broad_mask, block_size[:-1], new_overlap[:-1])
        mask_small = extract_patches(mask, block_size[:-1], new_overlap[:-1])
        shape = mask_small.shape[mask_small.ndim//2:]
        dims = tuple(range(mask_small.ndim//2, mask_small.ndim))
        print(mask.shape, dims, mask_small.shape)
        variance_mask = np.sum(mask_small, axis=dims) > (np.prod(shape)//2)
        print('variance shape', variance.shape, 0)
        # broad_variance = variance #np.pad(variance, padding[:-1], mode='constant')
        variance_small = extract_patches(variance, block_size[:-1], new_overlap[:-1])[variance_mask]
        print('variance shape', variance_small.shape, variance_mask.shape, dims)
        axis = range(1, variance_small.ndim)
        variance_small = np.median(variance_small, axis=axis)
        print('variance shape', variance_small.shape)

    X_small_denoised, alpha, intercept = solve_l1(X_small, D_depimpe, variance=variance_small, return_all=True,
                                                  positivity=True, nlambdas=100, fit_intercept=True, standardize=True, progressbar=True)

    # reconstruct_by_indexes = False
    # if reconstruct_by_indexes:
    #     mask = train_small.ravel()
    #     return reconstruct_from_indexes(alpha, D, intercept, new_shape, mask, block_size, block_up)

    # X_small_denoised = np.zeros(10)
    # alpha = np.zeros((D.shape[1], X_small.shape[0]))
    # intercept = np.zeros((alpha.shape[1], 1))
    # mkl.set_num_threads(ncores)
    # Xhat, alpha, intercept
    print('total time : {}'.format(time() - t), train_small.shape, D.shape, alpha.shape)
    print('mean alpha', np.abs(alpha).mean())
    # print(X_small_denoised.shape, X_small.shape, X_upsampled_shape, D_depimpe.shape)
    # 1/0
    del X_small_denoised
    # upsampled = np.zeros((X_small_shape[0], D.shape[0]), dtype=np.float32)
    indexes = train_small.ravel()
    upsampled = np.zeros((indexes.shape[0],) + new_size, dtype=np.float32)
    full_weights = np.ones(indexes.shape[0], dtype=np.float32)
    print(train_small.shape, indexes.shape)
    # weights = 1. / (1. + np.sum(alpha != 0, axis=0, dtype=np.float32))
    weights = 1.
    stuff = np.dot(D, alpha).T + intercept

    # fix offset to enforce original mean consistency
    # offset = X_mean - stuff.mean(axis=-1, keepdims=True)
    # stuff += offset

    print(train_small.shape, indexes.shape, new_shape)
    print(upsampled[indexes].shape, upsampled.shape)

    upsampled[indexes] = stuff.reshape((stuff.shape[0],) + new_size)
    full_weights[indexes] = weights
    del stuff, weights, offset, indexes

    output = reconstruct_from_blocks(upsampled, new_shape, block_size, block_up, new_overlap, weights=full_weights)
    return output


# def pimp(D, block_size, factor, order=1):

#     if len(factor) == len(block_size) - 1:
#         factor = tuple(factor) + (1,)

#     size = np.prod(block_size)
#     size_up = np.prod(np.array(block_size) * factor)
#     num = D.shape[0] // size

#     unpacked = np.zeros(block_size, dtype=np.float64)
#     unpacked_up = np.zeros(np.array(block_size) * factor, dtype=np.float64)

#     D_up = np.zeros((size_up, D.shape[1]), dtype=np.float64)

#     for j in range(D.shape[1]):
#         for i in range(num):
#             unpacked[:] = D[i:(i + 1)*size, j].reshape(block_size)
#             zoom(unpacked, factor, output=unpacked_up, order=order)
#             D_up[i:(i + 1)*size_up, j] = unpacked_up.reshape(size_up)

#     return D_up


# def segmented_dot(a, b_sparse, step=10000):

#     out = np.zeros((a.shape[0], b_sparse.shape[1]), dtype=b_sparse.dtype)
#     temp = np.zeros((b_sparse.shape[0], step), dtype=b_sparse.dtype)
#     for i in range(0, b_sparse.shape[1], step):

#         if i+step > b_sparse.shape[1]:
#             step = b_sparse.shape[1] % step
#             temp = np.zeros((b_sparse.shape[0], step), dtype=b_sparse.dtype)

#         b_sparse[:,i:i+step].toarray(out=temp)
#         np.dot(a, temp, out=out[:, i:i+step])

#     return out


# this should try to downsample in the same way we recombine indexes with factors per dimension
def depimp_slicing(D, block_size, block_up):

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

    valid = (tuple(np.append(valid[0], plus[0]).astype(np.int16)),
             tuple(np.append(valid[1], plus[1]).astype(np.int16)),
             tuple(np.append(valid[2], plus[2]).astype(np.int16)))

    # D_depimpe = np.zeros((np.prod(block_size[:3]), D.shape[1]), dtype=np.float64)
    D_depimpe = D.reshape(block_up + (-1, D.shape[-1]))[valid[0], valid[1], valid[2]]
    # print(valid)
    # for i in range(D_depimpe.shape[1]):
        # out = D[:, i].reshape(block_up + (-1,))[valid[0], valid[1], valid[2]]
        # print(out.shape, D.shape, D_depimpe.shape)
        # D_depimpe[:, i] = out.ravel()

    return D_depimpe


def depimp_cut(D, block_size, factor):
    D_depimpe = np.zeros((D.shape[0]//factor**3, D.shape[1]), dtype=np.float64)
    block_size = np.array(block_size)
    size = block_size * factor

    for i in range(D_depimpe.shape[1]):
        out = D[:, i].reshape(size)[::factor[0], ::factor[1], ::factor[2]]
        D_depimpe[:, i] = out.ravel()

    return D_depimpe


def depimp_mean(D, block_size, factor):

    block_size = np.array(block_size)

    # if not isinstance(factor, np.ndarray):
    #     factor = np.array([factor, factor, factor])
    #     print(factor)
    if len(factor) < len(block_size):
        factor = np.concatenate((factor, [1]))
    # factor = factor.astype(np.int16)
    size = tuple(np.array((block_size[0] * factor[0],
                           block_size[1] * factor[1],
                           block_size[2] * factor[2]), dtype=np.int16))

    resizer = (block_size[0], factor[0], block_size[1], factor[1], block_size[2], factor[2])
    # print(size)
    print(D.reshape(size + (-1,)).shape, size, 'line1')
    print(resizer, np.prod(resizer), 'line2')

    D_reshaped = D.reshape(size + (-1,)).reshape(resizer + (-1,))
    D_depimpe = np.mean(D_reshaped, axis=(1,3,5), dtype=np.float64)
    print(D_depimpe.shape, D_reshaped.shape, np.prod(block_size), block_size, D.shape, 'line3')
    print(np.asfortranarray(D_depimpe.reshape(np.prod(block_size), -1)).shape, 'line 4')

    return D_depimpe.reshape(np.prod(block_size), -1)


def depimp_zoom(D, block_size, block_up, order=1, zoomarray=False):

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

    print(D.shape, reshaped.shape, zoomer, size, block_up, block_size, 'zoomer')
    if zoomarray:
        print(reshaped.shape, tuple(block_size) + (D.shape[1],), np.prod(block_size))
        D_depimpe = zoomArray(reshaped, tuple(block_size) + (D.shape[1],), order=order).reshape(np.prod(block_size), -1)
    else:
        D_depimpe = zoom(reshaped, zoomer, order=order).reshape(np.prod(block_size), -1)

    print(D_depimpe.shape, D.shape, size, D.reshape(size + (-1,)).shape, zoomer, 'the shapes')
    return D_depimpe


# def depimp_lanczos(D, block_size, block_up):
#     from scipy.ndimage.filters import convolve

#     def kernel(x, a=2):
#         if np.abs(x) < a:
#             return np.sinc(x) * np.sinc(x / a)
#         return 0


def zoomArray(inArray, finalShape, sameSum=False, zoomFunction=zoom, **zoomKwargs):
    """
    Stolen from https://stackoverflow.com/questions/34047874/scipy-ndimage-interpolation-zoom-uses-nearest-neighbor-like-algorithm-for-scalin

    Normally, one can use scipy.ndimage.zoom to do array/image rescaling.
    However, scipy.ndimage.zoom does not coarsegrain images well. It basically
    takes nearest neighbor, rather than averaging all the pixels, when
    coarsegraining arrays. This increases noise. Photoshop doesn't do that, and
    performs some smart interpolation-averaging instead.

    If you were to coarsegrain an array by an integer factor, e.g. 100x100 ->
    25x25, you just need to do block-averaging, that's easy, and it reduces
    noise. But what if you want to coarsegrain 100x100 -> 30x30?

    Then my friend you are in trouble. But this function will help you. This
    function will blow up your 100x100 array to a 120x120 array using
    scipy.ndimage zoom Then it will coarsegrain a 120x120 array by
    block-averaging in 4x4 chunks.

    It will do it independently for each dimension, so if you want a 100x100
    array to become a 60x120 array, it will blow up the first and the second
    dimension to 120, and then block-average only the first dimension.

    Parameters
    ----------

    inArray: n-dimensional numpy array (1D also works)
    finalShape: resulting shape of an array
    sameSum: bool, preserve a sum of the array, rather than values.
             by default, values are preserved
    zoomFunction: by default, scipy.ndimage.zoom. You can plug your own.
    zoomKwargs:  a dict of options to pass to zoomFunction.
    """
    inArray = np.asarray(inArray, dtype=np.float64)
    inShape = inArray.shape
    assert len(inShape) == len(finalShape)
    mults = []  # multipliers for the final coarsegraining
    for i in range(len(inShape)):
        if finalShape[i] < inShape[i]:
            mults.append(int(np.ceil(inShape[i] / finalShape[i])))
        else:
            mults.append(1)
    # shape to which to blow up
    tempShape = tuple([i * j for i, j in zip(finalShape, mults)])

    # stupid zoom doesn't accept the final shape. Carefully crafting the
    # multipliers to make sure that it will work.
    zoomMultipliers = np.array(tempShape) / np.array(inShape) + 0.0000001
    assert zoomMultipliers.min() >= 1

    # applying scipy.ndimage.zoom
    rescaled = zoomFunction(inArray, zoomMultipliers, **zoomKwargs)

    for ind, mult in enumerate(mults):
        if mult != 1:
            sh = list(rescaled.shape)
            assert sh[ind] % mult == 0
            newshape = sh[:ind] + [sh[ind] // mult, mult] + sh[ind + 1:]
            rescaled.shape = newshape
            rescaled = np.mean(rescaled, axis=ind + 1)
    assert rescaled.shape == finalShape

    if sameSum:
        extraSize = np.prod(finalShape) / np.prod(inShape)
        rescaled /= extraSize
    return rescaled


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

            # # fix offset to enforce original mean consistency
            # if patch_mean is not None:
            #     stuff += patch_mean[p] - stuff.mean(axis=-1, keepdims=True)

            # if small_recon is not None:
            #     norm0 = np.sum(alpha[:, p] != 0)
            #     error = np.sqrt(np.sum((stuff - small_recon[p])**2))
            #     # weights = np.exp(-error/norm0)
            # # else:
            #     # weights = 1. / (1. + np.sum(alpha[:, p] != 0, axis=0, dtype=np.float64))
            #     # weights = 1

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
    print(img.shape, div.shape)
    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    n_l = i_l - p_l + 1

    # if weights is None:
    #     weights = np.broadcast_to(1., patches.shape[0])

    print(patches.shape, image_size, block_size, block_up, new_overlap)
    frac, nup = np.modf(np.divide(block_up, block_size[:-1]))
    int_frac = np.ceil(frac).astype(np.int32)
    nup = np.array(nup + int_frac, dtype=np.int32)
    print(frac, nup, block_size, block_up, n_h, n_w, n_l)
    print(range(nup[0], n_h + nup[0]), block_up[0]-nup[0], range(block_size[0]))

    valid = (np.arange(0, block_up[0], nup[0]),
             np.arange(0, block_up[1], nup[1]),
             np.arange(0, block_up[2], nup[2]))

    print(valid, block_up, block_size)
    print(len(valid[0]) - block_size[0], len(valid[1]) - block_size[1], len(valid[2]) - block_size[2])

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

    # valid = (np.append(valid[0], np.arange(block_up[0], 0, -1)[:(len(valid[0]) - block_size[0])]),
    #          np.append(valid[1], np.arange(block_up[1], 0, -1)[:(len(valid[1]) - block_size[1])]),
    #          np.append(valid[2], np.arange(block_up[2], 0, -1)[:(len(valid[2]) - block_size[2])]))

    # nup = np.array(nup - int_frac, dtype=np.int32)
    # print(nup)

    # step = ([i for i in range(n_h) if i % (block_up[0]) in valid[0]],
    #         [j for j in range(n_w) if j % (block_up[1]) in valid[1]],
    #         [k for k in range(n_l) if k % (block_up[2]) in valid[2]])

    step = ([slice(i, i + p_h) for i in range(0, n_h + int_frac[0], new_overlap[0]) if i % block_up[0] in valid[0]],
            [slice(j, j + p_w) for j in range(0, n_w + int_frac[1], new_overlap[1]) if j % block_up[1] in valid[1]],
            [slice(k, k + p_l) for k in range(0, n_l + int_frac[2], new_overlap[2]) if k % block_up[2] in valid[2]])

    # step = ([i for i in range(n_h + 1) if i % block_up[0] in valid[0]],
    #         [j for j in range(n_w + 1) if j % block_up[1] in valid[1]],
    #         [k for k in range(n_l + 1) if k % block_up[2] in valid[2]])

    # print(valid)
    # print(step)

    # step = ([i for i in range(nup[0], n_h + nup[0]) if i % block_up[0]-nup[0] in range(block_size[0])],
    #         [j for j in range(nup[1], n_w + nup[1]) if j % block_up[1]-nup[1] in range(block_size[1])],
    #         [k for k in range(nup[2], n_l + nup[2]) if k % block_up[2]-nup[2] in range(block_size[2])])

    ijk = list(product(*step))

    print(patches.shape[0], len(ijk))
    # print(step)

    for p, (i, j, k) in zip(patches, ijk):
        # img[i:i + p_h, j:j + p_w, k:k + p_l] += patches[p] * weights[p]
        # div[i:i + p_h, j:j + p_w, k:k + p_l] += weights[p]

        img[i, j, k] += p
        div[i, j, k] += 1

    # for p, (i, j, k) in zip(range(patches.shape[0]), ijk):
    #     img[i:i + p_h, j:j + p_w, k:k + p_l] += patches[p] * weights[p]
    #     div[i:i + p_h, j:j + p_w, k:k + p_l] += weights[p]
        # print(p,i,j,k, p_h, p_w, p_l, n_h, n_w, n_l)

    # unpad = img / div
    # unpad = unpad[:-p_h, :-p_w, :-p_l]
    out = img[:-p_h, :-p_w, :-p_l] / div[:-p_h, :-p_w, :-p_l]
    return out
