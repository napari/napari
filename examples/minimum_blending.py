"""
Minimum blending
================

Demonstrates how to use the `minimum` blending mode with inverted colormaps.
`minimum` blending uses the minimum value of each R, G, B channel for each pixel.
`minimum` blending can be used to yield multichannel color images on a white 
background, when the channels have inverted colormaps assigned.
An inverted colormap is one where white [1, 1, 1] is used to represent the lowest 
values, as opposed to the more conventional black [0, 0, 0]. For example, try the
colormaps prefixed with *I*, such as *I Forest* or *I Bordeaux*, from 
ChrisLUTs: https://github.com/cleterrier/ChrisLUTs .

.. tags:: visualization-basic
"""

from skimage import data

import napari

# create a viewer
viewer = napari.Viewer()

# Add the cells3d example image, using the two inverted colormaps
# and minimum blending mode. Note that the bottom-most layer
# must be translucent or opaque to prevent blending with the canvas.
viewer.add_image(data.cells3d(),
                            name=["membrane", "nuclei"],
                            channel_axis=1,
                            contrast_limits = [[1110, 23855], [1600, 50000]],
                            colormap = ["I Purple", "I Orange"], 
                            blending= ["translucent_no_depth", "minimum"]
                            )

if __name__ == '__main__':
    napari.run()
