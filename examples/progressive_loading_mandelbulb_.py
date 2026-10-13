"""
Progressive loading: 3D Mandelbulb
==================================

Display a multiscale 3D volume whose chunks are computed on the fly, using
napari's experimental progressive loading in 3D.

While napari itself renders the coarsest level of a multiscale image in
3D, progressive loading automatically selects the data level from the
camera zoom: zoom in and finer chunks stream in progressively, with
coarser data shown as a backdrop in the meantime. Chunks are prioritized
by distance to the camera, so the volume fills in from the front of the
view. The resolution selector in the layer controls can still pin an
explicit level; set it back to "Auto" to resume zoom-driven selection.

.. tags:: experimental
"""

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)
from napari.experimental._progressive_loading_datasets import (
    mandelbulb_dataset,
)

dataset = mandelbulb_dataset(max_levels=5, tilesize=32, maxiter=64)

viewer = napari.Viewer()
viewer.dims.ndisplay = 3
layer = add_progressive_loading_image(
    dataset['arrays'],
    viewer=viewer,
    contrast_limits=(0, 64),
    colormap='magma',
    rendering='attenuated_mip',
)

if __name__ == '__main__':
    napari.run()
