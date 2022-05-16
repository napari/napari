"""Hello, napari docs world!

Layers are the viewable objects that can be added to a viewer.

Custom layers must inherit from Layer and pass along the
`visual node <https://vispy.org/api/vispy.scene.visuals.html>`_
to the super constructor.
"""
import inspect as _inspect

from ..utils.misc import all_subclasses as _all_subcls
from .base import Layer
from .image import Image
from .labels import Labels
from .points import Points
from .shapes import Shapes
from .surface import Surface
from .tracks import Tracks
from .vectors import Vectors

# isabstact check is to exclude _ImageBase class
NAMES = {
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
