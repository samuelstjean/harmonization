import os.path
import subprocess
import pytest

path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(path, 'datasets', 'config.yaml')
filename_upscale = os.path.join(path, 'datasets', 'config_upscale.yaml')

files = [filename, filename_upscale]
commands = []

for fname in files:
    commands += [f'harmonization_get_global_D {fname}',
                 f'harmonization_harmonize_my_data {fname}']

@pytest.mark.parametrize('command', commands)
def test_scripts(command):
    subprocess.run([command], shell=True, check=True)
