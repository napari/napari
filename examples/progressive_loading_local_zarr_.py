"""
Progressive loading: local multiscale zarr
==========================================

Build a multiscale mandelbulb zarr on disk, then view it with
progressive loading. This separates the data-loading path from
on-the-fly chunk generation, making it a realistic test of local
zarr I/O performance.

The first run builds ``mandelbulb.zarr`` (~150 MB, takes a minute or
two with numba). Subsequent runs reuse the existing store.

.. tags:: experimental
"""

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)
from napari.experimental._progressive_loading_datasets import (
    local_zarr_dataset,
)

dataset = local_zarr_dataset('mandelbulb.zarr')

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
