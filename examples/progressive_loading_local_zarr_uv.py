# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "napari[pyqt5]",
#     "zarr>=3.1.6",
#     "numba",
# ]
#
# [tool.uv.sources]
# napari = { git = "https://github.com/kephale/napari", rev = "cc1bf8bc" }
# ///
"""
Progressive loading: local multiscale zarr via uv
=================================================

Self-contained launcher for the local-zarr progressive-loading demo.
The inline script metadata above lets ``uv run`` install napari from a
pinned commit of this branch and resolve every dependency without a
clone or a virtualenv::

    uv run progressive_loading_local_zarr_uv.py

The first run builds ``mandelbulb.zarr`` (~150 MB) in the working
directory using numba-accelerated chunk generation. Subsequent runs
reuse the existing store and launch immediately.

Edit ``rev`` in the header to A/B any two commits.

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
