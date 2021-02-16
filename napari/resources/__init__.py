import warnings
from pathlib import Path
from typing import List, Optional

ICON_PATH = (Path(__file__).parent / 'icons').resolve()
ICON_NAMES = {x.stem for x in ICON_PATH.iterdir()}


def get_icon_path(name: str) -> str:
    """Return path to an SVG in the theme icons."""
    if name not in ICON_NAMES:
        raise ValueError(
            f"unrecognized icon name: {name!r}. Known names: {ICON_NAMES}"
        )
    return str(ICON_PATH / f'{name}.svg')


def get_stylesheet(extra: Optional[List[str]] = None) -> str:
    """For backward compatibility"""
    warnings.warn(
        "Moved to module napari._qt.qt_resources. Will be removed after version 0.4.6.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    from .._qt.qt_resources import get_stylesheet as qt_get_stylesheet

    return qt_get_stylesheet(extra)
