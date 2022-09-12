"""
Set colormaps
=============

Add named or unnamed vispy colormaps to existing layers.

.. tags:: visualization-basic
"""

import numpy as np
import vispy.color
from skimage import data
import napari


histo = data.astronaut() / 255
rch, gch, bch = np.transpose(histo, (2, 0, 1))
red = vispy.color.Colormap([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
green = vispy.color.Colormap([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
blue = vispy.color.Colormap([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

v = napari.Viewer()

rlayer = v.add_image(rch, name='red channel')
rlayer.blending = 'additive'
rlayer.colormap = 'red', red
glayer = v.add_image(gch, name='green channel')
glayer.blending = 'additive'
glayer.colormap = green  # this will appear as [unnamed colormap]
blayer = v.add_image(bch, name='blue channel')
blayer.blending = 'additive'
blayer.colormap = {'blue': blue}

if __name__ == '__main__':
    napari.run()
