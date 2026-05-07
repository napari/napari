"""
Limit Layer Modes
=================

Show example how to limit the available modes for a layer using subclassing.

.. tags:: layers
"""


import numpy as np

from napari import Viewer, run
from napari.layers import Labels


class OwnLabels(Labels):
    def support_mode(self, mode: str):
        return super().support_mode(mode) and mode != 'erase'

viewer = Viewer()
layer = OwnLabels(np.zeros((512, 512), dtype='uint8'), name='labels')
viewer.add_layer(layer)

run()
