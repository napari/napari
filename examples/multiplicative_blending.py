"""
Multiplicative blending
=======================

Use ``multiplicative`` blending to overlay a probability map on an image
without hiding the image underneath.

Each blending mode decides how the color of a layer is combined with the color
that is already on the canvas. ``multiplicative`` multiplies the two and adds
their alpha values, so a pixel that is white in the upper layer, [1, 1, 1],
leaves the pixel below it unchanged, while a colored pixel tints it towards
that color. To take advantage of this, the upper layer needs an *inverted*
colormap, in which the lowest values are white instead of black, such as the
colormaps whose name starts with *I* from ChrisLUTs:
https://github.com/cleterrier/ChrisLUTs .

This is what makes it possible to show where a detector fired without hiding
the data it was run on: where the probability is low, the map is white and the
image below it is unchanged. With the default ``translucent`` blending mode,
the same map would cover the image with white wherever the probability is low.

.. tags:: visualization-basic
"""

import numpy as np
from skimage import data
from skimage.filters import gaussian, threshold_otsu

import napari

###############################################################################
# We start from a real image: one optical section of the nuclei channel of the
# ``cells3d`` dataset.

nuclei = data.cells3d()[30, 1]

###############################################################################
# The map to overlay is meant to be the output of a detector, but is simulated
# here by smoothing a thresholded version of the image, so that it is high
# wherever nuclei are dense. A real application would use the output of a
# trained classifier instead. The values are then rescaled to [0, 1], the
# interval a probability is expressed in, so that the map spans the whole
# colormap.

nuclei_mask = nuclei > threshold_otsu(nuclei)
density = gaussian(nuclei_mask.astype(np.float32), sigma=5)
density /= density.max()

###############################################################################
# Finally we display the image with the density map on top of it. The map uses
# an inverted colormap and ``multiplicative`` blending, so its white background
# leaves the image untouched and the nuclei are tinted only where the density
# is high. The bottom layer is ``opaque`` so that it is not blended with the
# black canvas.

viewer, _ = napari.imshow(
    nuclei,
    name='nuclei',
    colormap='gray',
    blending='opaque',
)
viewer.add_image(
    density,
    name='nuclei density',
    colormap='I Orange',
    blending='multiplicative',
)

if __name__ == '__main__':
    napari.run()
