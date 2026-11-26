"""Dependency check shared by the CLI and the optional hot-reload dock."""

from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version


def _require_qtreload() -> None:
    """Raise an installation hint if the required qtreload version is missing."""
    try:
        if Version(version('qtreload')) >= Version('0.3.0'):
            return
    except PackageNotFoundError:
        pass
    raise ImportError(
        'Hot reload requires qtreload >= 0.3.0. '
        'Install or upgrade with pip install -U "napari[reload]".'
    )
