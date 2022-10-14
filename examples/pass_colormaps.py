"""
Pass colormaps
==============

Add named or unnamed vispy colormaps to existing layers.

.. tags:: visualization-basic
"""

import numpy as np
from skimage import data
import napari


histo = data.astronaut() / 255
rch, gch, bch = np.transpose(histo, (2, 0, 1))

v = napari.Viewer()

rlayer = v.add_image(
    rch, name='red channel', colormap='red', blending='additive'
)
glayer = v.add_image(
    gch, name='green channel', colormap='green', blending='additive'
)
blayer = v.add_image(
    bch, name='blue channel', colormap='blue', blending='additive'
)

if __name__ == '__main__':
    napari.run()
