import os
from enum import auto
from pathlib import Path

from napari.utils.compat import StrEnum


class Environment(StrEnum):
    pip = auto()
    uv = auto()
    conda = auto()
    pixi = auto()


def get_venv_path() -> Path | None:
    """Get the path to the current virtual environment, if any.

    Returns
    -------
    Path | None
        The path to the current virtual environment, or None if not in a venv.
    """
    venv = os.environ.get('VIRTUAL_ENV')
    return Path(venv) if venv else None


def get_conda_path() -> Path | None:
    """Get the path to the current conda environment, if any.

    Returns
    -------
    Path | None
        The path to the current conda environment, or None if not in conda.
    """
    conda = os.environ.get('CONDA_PREFIX')
    return Path(conda) if conda else None


def chek_if_uv_venv() -> bool:
    """Check if we are in a uv virtual environment.

    Returns
    -------
    bool
        True if we are in a uv virtual environment, False otherwise.
    """
    venv_path = get_venv_path()
    if venv_path is None:
        return False
    if not (venv_path / 'pyvenv.cfg').exists():
        return False
    return 'uv =' in (venv_path / 'pyvenv.cfg').read_text()


def detect_environment() -> Environment:
    if get_venv_path():
        if chek_if_uv_venv():
            return Environment.uv
        return Environment.pip
    if get_conda_path() is not None:
        return Environment.conda
    return Environment.pip
