from enum import auto

from napari.utils.compat import StrEnum


class TriangulationBackend(StrEnum):
    """Enum-like class to specify which triangulation backend to use."""

    bermuda = auto()
    partsegcore = auto()
    triangle = auto()
    none = auto()
