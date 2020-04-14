"""Layers are the viewable objects that can be added to a viewer.

Custom layers must inherit from Layer and pass along the
`visual node <http://vispy.org/scene.html#module-vispy.scene.visuals>`_
to the super constructor.
"""

from ..utils.misc import all_subclasses
from .base import Layer
from .image import Image
from .labels import Labels
from .points import Points
from .shapes import Shapes
from .surface import Surface
from .vectors import Vectors

NAMES = [subclass.__name__.lower() for subclass in all_subclasses(Layer)]
del all_subclasses
