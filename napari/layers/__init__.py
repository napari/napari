"""Layers are the viewable objects that can be added to a viewer.

Custom layers must inherit from Layer and pass along the
`visual node <http://vispy.org/scene.html#module-vispy.scene.visuals>`_
to the super constructor.
"""


from .base import Layer
from .image import Image
from .points import Points
from .vectors import Vectors
from .surface import Surface
from .shapes import Shapes
from .labels import Labels
