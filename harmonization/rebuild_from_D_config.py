from __future__ import print_function, division

import sys
import os

import numpy as np
import nibabel as nib

from itertools import cycle
from ast import literal_eval
from glob import iglob


from harmonization.config import get_arguments

from nlsam.angular_tools import angular_neighbors
from nlsam.denoiser import greedy_set_finder


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
    outpath = kwargs.pop('outpath')
    path_D = kwargs.pop('outfilename')
    use_glob = kwargs.pop('glob')

    center = kwargs['center']
    block_up = literal_eval(kwargs['block_up'])
    block_size = literal_eval(kwargs['block_size'])
    # use_std = kwargs['use_std']
    positivity = kwargs['positivity']
    fit_intercept = kwargs['fit_intercept']
    center = kwargs['center']
    fix_mean = kwargs['fix_mean']
    use_crossval = kwargs['use_crossval']
    split_b0s = kwargs['split_b0s']
    b0_threshold = kwargs['b0_threshold']
    # batchsize = kwargs['batchsize']
    # niter = kwargs['niter']
    ncores = kwargs['ncores']

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

    D = np.load(path_D)
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

    # print(scanner)
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
                if name == filename:
                    dataset = os.path.join(root, name)
                    datasets += [dataset]

            # if (name == 'dwi.nii' or name == 'dwi_fw.nii') and to_scanner.lower() in root.lower():
            #     to_datasets += [os.path.join(root, name)]

    # if len(datasets) != len(to_datasets):
    #     raise ValueError('Size mismatch between lists! {}, {}'.format(len(datasets), len(to_datasets)))

    for dataset in datasets:

        print('Now rebuilding {}'.format(dataset['data']))
        predicted_D = path_D.replace('.npy', '')
        output_filename = dataset['data'].replace('.nii', '_predicted_' + predicted_D + '.nii.gz')
        output_filename = output_filename.replace('.nii.gz', '_recon.nii.gz')
        output_filename = os.path.join(outpath, os.path.basename(output_filename))

        # if center:
        #     output_filename = output_filename.replace('_predicted_', '_predicted_center_')

        # if use_crossval:
        #     output_filename = output_filename.replace('_predicted_', '_predicted_with_cv2_')
        # else:
        #     output_filename = output_filename.replace('_predicted_', '_predicted_with_aic_')

        # if use_std:
        #     output_filename = output_filename.replace('_predicted_', '_predicted_with_std_')

        # if positivity:
        #     output_filename = output_filename.replace('_predicted_', '_predicted_pos_')

        # if fix_mean:
        #     output_filename = output_filename.replace('_predicted_', '_predicted_meanfix_')

        # if not fit_intercept:
        #     output_filename = output_filename.replace('_predicted_', '_predicted_no_intercept_')

        if os.path.isfile(output_filename):
            print('File already exists! Skipping {}'.format(output_filename))
        else:
        # if True:
            vol = nib.load(dataset['data'])

            # once we have loaded the data, replace the name if we used a fw dataset since everything else will match
            # dataset = dataset.replace('_fw', '')
            # to_dataset = to_dataset.replace('_fw', '')

            # indexer = get_indexer(dataset)
            data = vol.get_fdata(caching='unchanged').astype(np.float32)
            affine = vol.affine
            header = vol.header

            mask = nib.load(dataset['mask']).get_fdata(caching='unchanged').astype(np.bool)
            # std = nib.load(dataset.replace('.nii', '_std.nii.gz')).get_data()
            bvals = np.loadtxt(dataset['bval'])
            bvecs = np.loadtxt(dataset['bvec'])

            if np.shape(bvecs)[0] == 3:
                bvecs = bvecs.T

            # data = data[:, :, 30:36]
            # mask = mask[:, :, 30:36]
            # if use_std:
            #     std = std[:, :, 30:36]

            if center:
                # # data_mean = np.mean(data, axis=-1, keepdims=True)
                # with warnings.catch_warnings():
                #     warnings.filterwarnings('ignore')
                #     data_mean = np.nanmean(np.where(data != 0, data, np.nan), axis=-1, keepdims=True)

                # # print(np.sum(np.isfinite(data_mean)), np.sum(np.isnan(data_mean)))
                # data_mean[np.isnan(data_mean)] = 0

                # we need to pull down the 3D volume mean for the upsampling part to make sense though
                data_mean = np.nanmean(np.where(data != 0, data, np.nan), axis=(0, 1, 2), keepdims=True)
                data -= data_mean
                # print('centering removed {}'.format(data_mean.mean()))
                # 1/0

            # if use_std:
            #     if bias_correct_std:
            #         print('this is now disabled somehow :/')
            #         # print(std.shape, data.shape)
            #         # std = np.broadcast_to(std[..., None], data.shape)
            #         # mask_4D = np.broadcast_to(mask[..., None], data.shape)
            #         # std = np.median(corrected_sigma(data, std, mask_4D, N), axis=-1)
            #     variance = std**2
            # else:
            # std = None
            variance = None

            # number of blocks is without b0, so we +1 the last dimension
            if len(block_up) < data.ndim:
                # last_size = int(D.shape[0] / np.prod(block_size))
                current_block_size = block_size[:-1] + (block_size[-1] + 1,)
                current_block_up = block_up + (block_size[-1] + 1,)
            else:
                current_block_size = block_size[:-1] + (block_size[-1] + 1,)
                current_block_up = block_up[:-1] + (block_up[-1] + 1,)

            print('Output filename is {}'.format(output_filename))
            print(D.shape, current_block_size, current_block_up)

            factor = np.divide(current_block_up, current_block_size)

            # split_b0s = True
            # b0_threshold = 20
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

            print(data.shape, predicted.shape, factor)

            # Put all idx + b0 in this array in each iteration
            to_denoise = np.empty(data.shape[:-1] + (current_block_size[-1],), dtype=np.float64)
            print(to_denoise.shape)

            for i, idx in enumerate(indexes, start=1):
                b0_loc = tuple((next(split_b0s_idx),))
                to_denoise[..., 0] = data[..., b0_loc].squeeze()
                to_denoise[..., 1:] = data[..., idx]
                divider[list(b0_loc + idx)] += 1

                print('Now denoising volumes {} / block {} out of {}.'.format(b0_loc + idx, i, len(indexes)))
                predicted[..., b0_loc + idx] += rebuild(to_denoise,
                                                        mask,
                                                        D,
                                                        block_size=current_block_size,
                                                        block_up=current_block_up,
                                                        ncores=ncores,
                                                        positivity=positivity,
                                                        fix_mean=fix_mean,
                                                        # center=center,
                                                        fit_intercept=fit_intercept,
                                                        use_crossval=use_crossval,
                                                        variance=variance)
                # break
            predicted /= divider

            if center:
                predicted += data_mean
                # el cheapo mask after upsampling
                predicted[predicted == data_mean] = 0

            # clip negatives, which happens at the borders
            predicted.clip(min=0., out=predicted)

            # header voxel size is all screwed up, so replace with the destination header and affine by the matching dataset
            # to_affine = nib.load(to_dataset).affine
            # to_header = nib.load(to_dataset).header
            # header['pixdim'] = to_header['pixdim']

            imgfile = nib.Nifti1Image(predicted, affine, header)
            nib.save(imgfile, output_filename)

            # # subsample to common bvals/bvecs/tr/te
            # to_data = nib.load(to_dataset).get_data()
            # to_bvals = np.loadtxt(to_dataset.replace('.nii', '.bval'))
            # to_bvecs = np.loadtxt(to_dataset.replace('.nii', '.bvec'))

            # to_indexer = get_indexer(to_dataset)
            # to_data = to_data[..., to_indexer]
            # to_bvals = to_bvals[to_indexer]
            # to_bvecs = to_bvecs[to_indexer]

            # np.savetxt(to_dataset.replace('dwi.nii', 'dwi_subsample.bval'), to_bvals, fmt='%1.4f')
            # np.savetxt(to_dataset.replace('dwi.nii', 'dwi_subsample.bvec'), to_bvecs, fmt='%1.4f')

            # # match to the new bvecs
            # rotated = match_bvecs(predicted, bvals, bvecs, to_bvals, to_bvecs).clip(min=0.)
            # imgfile = nib.Nifti1Image(rotated, to_affine, to_header)
            # nib.save(imgfile, output_filename.replace('predicted', 'rotated'))

            # # match to the new bvecs with nnls
            # rotated_nnls = match_bvecs(predicted, bvals, bvecs, to_bvals, to_bvecs, use_nnls=True)
            # imgfile = nib.Nifti1Image(rotated_nnls, to_affine, to_header)
            # nib.save(imgfile, output_filename.replace('predicted', 'rotated_nnls'))

            del data, predicted, vol, imgfile, mask, variance
            del to_denoise, divider


if __name__ == "__main__":
    main()
