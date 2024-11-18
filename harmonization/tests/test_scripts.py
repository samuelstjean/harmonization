import os.path
import subprocess
import pytest

path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(path, 'datasets', 'config.yaml')

commands = [f'harmonization_get_global_D {filename}',
            f'harmonization_harmonize_my_data {filename}']

@pytest.mark.parametrize('command', commands)
def test_scripts(command):
    subprocess.run([command], shell=True, check=True)
