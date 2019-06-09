"""Layers are the viewable objects that can be added to a viewer.

Custom layers must inherit from Layer and pass along the
`visual node <http://vispy.org/scene.html#module-vispy.scene.visuals>`_
to the super constructor.
"""


from ._base_layer import Layer
from ._image_layer import Image
from ._markers_layer import Markers
from ._vectors_layer import Vectors
from ._shapes_layer import Shapes
from ._labels_layer import Labels
from ._pyramid_layer import Pyramid
from .util import create_qt_properties, create_qt_controls
