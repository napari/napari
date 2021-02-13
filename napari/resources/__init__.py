import warnings
from typing import List, Optional


def get_stylesheet(extra: Optional[List[str]] = None) -> str:
    """For backward compatibility"""
    warnings.warn(
        "Moved to module napari._qt.qt_resources. Will be removed after version 0.4.6.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    from .._qt.qt_resources import get_stylesheet as qt_get_stylesheet

    return qt_get_stylesheet(extra)
