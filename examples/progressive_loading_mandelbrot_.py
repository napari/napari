"""
Progressive loading: Mandelbrot set
===================================

Display an extremely large (multi-gigapixel) multiscale image whose chunks
are computed on the fly, using napari's experimental progressive loading.

A single multiscale image layer is added to the viewer. As you pan and
zoom, the chunks of the resolution level napari selects for display are
fetched on a background thread, nearest to the center of view first, while
coarser data is shown as a backdrop. Try zooming deep into the boundary of
the Mandelbrot set.

.. tags:: experimental
"""

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)
from napari.experimental._progressive_loading_datasets import (
    mandelbrot_dataset,
)

# 14 levels means the highest resolution level is 8.4 million pixels wide.
# Increase max_levels for an even deeper zoom.
dataset = mandelbrot_dataset(max_levels=14)

viewer = napari.Viewer()
layer = add_progressive_loading_image(
    dataset['arrays'],
    viewer=viewer,
    contrast_limits=(0, 255),
    colormap='twilight_shifted',
)

if __name__ == '__main__':
    napari.run()
