import os.path
import tempfile

from harmonization.config import write_config, read_config, get_filenames


def test_read_write_config():
    folder = tempfile.gettempdir()
    file = 'temp_write.yaml'
    filepath = os.path.join(folder, file)

    write_config(filepath)
    assert os.path.isfile(filepath)
    read_config(filepath)


def test_get_filenames():
    root = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(root, 'datasets')

    kwargs1 = {'dataname': 'dwi.nii.gz',
               'maskname': 'dwi_mask.nii.gz',
               'bval': 'dwi.bval',
               'bvec': 'dwi.bvec'}

    kwargs2 = {'dataname': 'dwi.nii.gz',
               'maskname': '_mask.nii.gz',
               'bval': '.bval',
               'bvec': '.bvec'}

    keys = ['data', 'mask', 'bval', 'bvec']
    filenames = ['dwi.nii.gz', 'dwi_mask.nii.gz', 'dwi.bval', 'dwi.bvec']

    for glob, kwargs in zip([False, True],
                            [kwargs1, kwargs2]):
        if glob:
            subjs = ['subj1', 'subj2']
            subjpath = path
        else:
            subjs = ['subj1']
            subjpath = os.path.join(path, 'subj1')

        datasets = get_filenames(subjpath, glob, kwargs)

        for subj, dataset in zip(subjs, datasets):
            for key, filename in zip(keys, filenames):
                assert dataset[key] == os.path.join(path, subj, filename)
