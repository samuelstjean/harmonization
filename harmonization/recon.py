import numpy as np

from itertools import product
from scipy.ndimage.interpolation import zoom


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


def reconstruct_from_blocks(patches, image_size, block_size, block_up, new_overlap, weights=None):

    i_h, i_w, i_l = image_size[:3]

    if len(patches.shape[1:]) == 3:
        p_h, p_w, p_l = patches.shape[1:]
    else:
        p_h, p_w, p_l = patches.shape[1:-1]

    img = np.zeros(image_size, dtype=np.float32)
    img = np.pad(img, [(0, p_h), (0, p_w), (0, p_l), (0, 0)], 'constant', constant_values=(0, 0))
    div = np.full(image_size, 1e-15, dtype=np.float32)
    div = np.pad(div, [(0, p_h), (0, p_w), (0, p_l), (0, 0)], 'constant', constant_values=(1e-15, 1e-15))

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
