"""Layers are the viewable objects that can be added to a viewer.

Custom layers must inherit from Layer and pass along the
`visual node <https://vispy.org/api/vispy.scene.visuals.html>`_
to the super constructor.
"""
import inspect as _inspect
from typing import Set

from napari.layers.base import Layer
from napari.layers.image import Image
from napari.layers.labels import Labels
from napari.layers.points import Points
from napari.layers.shapes import Shapes
from napari.layers.surface import Surface
from napari.layers.tracks import Tracks
from napari.layers.vectors import Vectors
from napari.utils.misc import all_subclasses as _all_subcls

# isabstact check is to exclude _ImageBase class
NAMES: Set[str] = {
    subclass.__name__.lower()
    for subclass in _all_subcls(Layer)
    if not _inspect.isabstract(subclass)
}

__all__ = [
    'Image',
    'Labels',
    'Layer',
    'Points',
    'Shapes',
    'Surface',
    'Tracks',
    'Vectors',
    'NAMES',
]
