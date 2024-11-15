import subprocess
import pytest

commands = ['harmonization_get_global_D config.yaml',
            'harmonization_harmonize_my_data config.yaml']

@pytest.mark.parametrize('command', commands)
def test_scripts(command):
    subprocess.run([command], shell=True, check=True)
