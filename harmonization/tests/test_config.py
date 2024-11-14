import os.path
import tempfile

from harmonization.config import write_config, read_config, get_filenames

folder = tempfile.gettempdir()
file = 'temp_write.yaml'

def test_write_config():
    write_config(os.path.join(folder, file))

def test_read_config():
    read_config(os.path.join(folder, file))

def test_get_filenames():
    path = os.path.dirname(os.path.realpath(__file__))
    glob = False
    kwargs = {'dataname': 'dwi.nii.gz',
              'maskname': 'dwi_mask.nii.gz',
              'bval': 'dwi.bval',
              'bvec': 'dwi.bvec'}

    get_filenames(path, glob, kwargs)
