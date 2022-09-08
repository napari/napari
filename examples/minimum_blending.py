"""
Minimum blending
================

Demonstrates how to use the `minimum` blending mode with inverted colormaps.
`minimum` blending uses the minimum value of each R, G, B channel for each pixel.
`minimum` blending can be used to yield multichannel color images on a white 
background, when the channels have inverted colormaps assigned.
An inverted colormap is one where white [1, 1, 1] is used to represent the lowest 
values, as opposed to the more conventional black [0, 0, 0].
"""

import numpy as np
from vispy.color import Colormap
from skimage import data
import napari

# First, generate two inverted colormaps that go from white [1, 1, 1] 
# to the chosen color.
# These are from ChrisLUTs: https://github.com/cleterrier/ChrisLUTs
# and correspond to the `green` and `magenta` colormaps used by default
# when opening this sample from the File menu.
I_Green = Colormap([[1, 1, 1], [0, 1, 0]])
I_Magenta = Colormap([[1, 1, 1], [1, 0, 1]])

# create a viewer
viewer = napari.Viewer()

# Add the cells3d example image, using the two inverted colormaps
# and minimum blending mode. Note that the bottom-most layer
# must be translucent or opaque to prevent blending with the canvas.
viewer.add_image(data.cells3d(),
                            name=["membrane", "nuclei"],
                            channel_axis=1,
                            contrast_limits = [[1110, 23855], [1600, 50000]],
                            colormap = [("I_Magenta", I_Magenta), ("I_Green", I_Green)], 
                            blending= ["translucent_no_depth", "minimum"]
                            )

if __name__ == '__main__':
    napari.run()
