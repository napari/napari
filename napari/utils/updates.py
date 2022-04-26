"""Napari update utilities."""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union
from urllib import request

from ..plugins.pypi import get_package_versions
from .misc import is_dev, parse_version, running_as_constructor_app

InstallerTypes = Literal['pip', 'conda']
LETTERS_PATTERN = re.compile(r'[a-zA-Z]')


def _get_napari_pypi_versions() -> List[str]:
    """Get the versions of the napari pypi package."""
    return get_package_versions('napari')


def _get_napari_conda_versions(nightly: bool = False) -> List[str]:
    """Get the versions of the napari conda package."""
    data = {}
    if nightly:
        url = 'https://api.anaconda.org/package/napari/napari'
    else:
        url = 'https://api.anaconda.org/package/conda-forge/napari'

    with request.urlopen(url) as response:
        try:
            data = json.loads(response.read().decode())
        except Exception:
            pass

    return data.get('versions', [])


def _get_installed_versions() -> List[str]:
    """Check the current conda prefix for installed versions."""
    versions = []
    path = Path(sys.prefix)
    envs_folder = path.parent
    if envs_folder.parts[-1] == "envs":
        # Check environment name is starts with napari
        env_paths = [
            p
            for p in envs_folder.iterdir()
            if p.stem.rsplit('-')[0] == 'napari'
        ]
        for env_path in env_paths:
            conda_meta_folder = envs_folder / env_path / 'conda-meta'
            napari_file = conda_meta_folder / 'napari'
            if (
                conda_meta_folder.exists()
                and napari_file.exists()
                and napari_file.is_file()
            ):
                for p in conda_meta_folder.iterdir():
                    if p.suffix == '.json':
                        # Check environment contains a napari package
                        parts = p.stem.rsplit('-')
                        if len(parts) == 3 and parts[-3] == 'napari':
                            versions.append(tuple(parts[1:]))
    return versions


def _is_stable_version(version: Union[Tuple[str], str]) -> bool:
    """
    Return ``True`` if version is stable.

    Stable version examples: ``0.4.12``, ``0.4.1``.
    Non-stable version examples: ``0.4.15beta``, ``0.4.15rc1``, ``0.4.15dev0``.
    """
    if not isinstance(version, tuple):
        version = version.split('.')

    return not LETTERS_PATTERN.search(version[-1])


def check_updates(
    stable: bool = True,
    nightly: bool = False,
    installer: InstallerTypes = None,
) -> Dict:
    """Check for napari bundle updates.

    Parameters
    ----------
    stable : bool, optional
        If ``True``, check for stable versions. Default is ``True``.
    nightly : bool, optional
        If ``True``, check for nightly versions. Only applicable to conda
        installer. Default is ``False``.
    installer : InstallerTypes, optional
        Installer type. If ``None`` is provided, the isntaller will be
        detected based on the system. Default is ``None``.

    Returns
    -------
    dict
        Dictionary containing the current and latest versions, found
        installed versions and the installer type used.
    """
    try:
        from napari._version import __version__
    except ImportError:
        __version__ = 'dev'

    versions = []
    if installer is None:
        if is_dev():
            versions = [__version__]
            installer = 'dev'
        elif running_as_constructor_app():
            installer = 'conda'
            versions = _get_napari_conda_versions(nightly=nightly)
        else:
            installer = 'pip'
            versions = _get_napari_pypi_versions()
    elif installer == 'pip':
        versions = _get_napari_pypi_versions()
    elif installer == 'conda':
        versions = _get_napari_conda_versions(nightly=nightly)

    if stable:
        versions = list(filter(_is_stable_version, versions))

    update = False
    latest_version = versions[-1] if versions else None
    installed_versions_builds = _get_installed_versions()
    installed_versions = [vb[0] for vb in installed_versions_builds]
    if __version__ != 'dev':
        update = (
            parse_version(latest_version) > parse_version(__version__)
            and latest_version not in installed_versions
        )

    return {
        "release": versions,
        "current": __version__,
        "latest": latest_version,
        "found": installed_versions,
        "installer": installer,
        "update": update,
    }
