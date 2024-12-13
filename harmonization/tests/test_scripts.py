import subprocess
import tempfile
import pytest

from pathlib import Path

cwd =  Path(__file__).parents[0]

filename = Path('datasets') / 'config.yaml'
filename_upscale = Path('datasets') /'config_upscale.yaml'
temppath = Path(tempfile.gettempdir()) / 'tester.yaml'

files = [filename, filename_upscale]
commands = ['harmonization_harmonize_my_data',
            'harmonization_get_global_D',
            ('harmonization_get_global_D', 'write', temppath)]

for fname in files:
    commands += [('harmonization_get_global_D', fname),
                 ('harmonization_harmonize_my_data', fname)]

@pytest.mark.parametrize('command', commands)
def test_scripts(command):
    subprocess.run(command, check=True, cwd=cwd)
