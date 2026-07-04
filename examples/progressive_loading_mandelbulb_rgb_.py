"""
Progressive loading: 3D RGB Mandelbulb
=======================================

Display a multiscale 3D RGB volume using napari's experimental progressive
loading.  The Mandelbulb escape-time is mapped to an RGB colormap so every
voxel is an (R, G, B) triplet (uint8).  This exercises the RGB texture path
in 3D progressive loading.

Chunks are generated lazily (on first access), so startup is instant.

.. tags:: experimental
"""

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)
from napari.experimental._progressive_loading_datasets import (
    mandelbulb_rgb_dataset,
)

dataset = mandelbulb_rgb_dataset(max_levels=5, tilesize=32, maxiter=64)

viewer = napari.Viewer()
viewer.dims.ndisplay = 3
layer = add_progressive_loading_image(
    dataset['arrays'],
    viewer=viewer,
    name='Mandelbulb RGB',
    rgb=True,
    rendering='attenuated_mip',
)

if __name__ == '__main__':
    napari.run()
