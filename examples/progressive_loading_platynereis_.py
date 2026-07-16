"""
Progressive loading: Platynereis whole-worm EM
===============================================

Browse a whole 6-day-old *Platynereis dumerilii* worm imaged by
serial-section electron microscopy at 10 nm XY / 25 nm Z. This is
the canonical OME-Zarr community demo dataset — 11 416 x 25 916 x
27 499 voxels with a 10-level pyramid.

The data is highly anisotropic (Z resolution ~2.5x coarser than XY),
which exercises the level-of-detail selection logic. Served over
plain HTTPS from EMBL — no S3 credentials needed.

.. tags:: experimental
"""

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)
from napari.experimental._progressive_loading_datasets import open_ome_zarr

URL = 'https://s3.embl.de/i2k-2020/platy-raw.ome.zarr'

arrays, scale, _translate = open_ome_zarr(URL, num_levels=10)

viewer = napari.Viewer()
viewer.dims.ndisplay = 3

layer = add_progressive_loading_image(
    arrays,
    viewer=viewer,
    name='Platynereis (serial-section EM)',
    contrast_limits=(0, 255),
    colormap='green',
    scale=scale,
    rendering='attenuated_mip',
)

if __name__ == '__main__':
    napari.run()
