"""Layers is the list of individual layer objects that have been added to the
viewer. Individual layer objects inherit from Layer and pass along the
`visual node <http://vispy.org/scene.html#module-vispy.scene.visuals>`
to the super constructor.
"""
from .model import Layers
from ._base_layer import Layer
from ._image_layer import Image
from ._markers_layer import Markers
from ._vectors_layer import Vectors
from ._shapes_layer import Shapes
from ._labels_layer import Labels
from ._register import add_to_viewer
