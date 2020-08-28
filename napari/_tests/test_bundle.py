import configparser
from pathlib import Path

import tomlkit

import napari


def test_bundle_requirements():
    """Test that briefcase requirements are superset of setup.cfg requirements.
    """
    parser = configparser.ConfigParser()
    root_dir = Path(napari.__file__).parent.parent
    parser.read(root_dir / 'setup.cfg')
    requirements = parser.get("options", "install_requires").splitlines()
    requirements = {r.split('#')[0].strip() for r in requirements if r}

    with open(root_dir / 'pyproject.toml', 'r') as f:
        pyproject = f.read()

    toml = tomlkit.parse(pyproject)
    toml_recs = set(toml['tool']['briefcase']['app']['napari']['requires'])
    assert requirements.issubset(toml_recs)
