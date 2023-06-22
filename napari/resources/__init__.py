from typing import List, Optional

from napari.resources._icons import (
    ICON_PATH,
    ICONS,
    get_colorized_svg,
    get_icon_path,
)
from napari.utils.translations import trans

__all__ = ['get_colorized_svg', 'get_icon_path', 'ICON_PATH', 'ICONS']


def get_stylesheet(extra: Optional[List[str]] = None) -> str:
    """For backward compatibility"""
    import warnings

    warnings.warn(
        trans._(
            "Moved to module napari._qt.qt_resources. Will be removed after version 0.4.6.",
            deferred=True,
        ),
        category=DeprecationWarning,
        stacklevel=2,
    )
    from napari._qt.qt_resources import get_stylesheet as qt_get_stylesheet

    return qt_get_stylesheet(extra)
