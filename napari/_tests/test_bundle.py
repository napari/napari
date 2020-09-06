import configparser
from pathlib import Path

import pytest
import tomlkit

import napari

root_dir = Path(napari.__file__).parent.parent
setup_file = root_dir / 'setup.cfg'


@pytest.mark.skipif(
    not setup_file.is_file(),
    reason='Bundle not tested in source or wheel distributions',
)
def test_bundle_requirements():
    """Test that briefcase requirements are superset of setup.cfg requirements.
    """
    parser = configparser.ConfigParser()
    parser.read(setup_file)
    requirements = parser.get("options", "install_requires").splitlines()
    requirements = {r.split('#')[0].strip() for r in requirements if r}

    with open(root_dir / 'pyproject.toml', 'r') as f:
        pyproject = f.read()

    toml = tomlkit.parse(pyproject)
    toml_recs = set(toml['tool']['briefcase']['app']['napari']['requires'])
    assert requirements.issubset(toml_recs)
