"""
Progressive loading: remote OME-Zarr
====================================

Browse a large remote OME-Zarr image with napari's experimental
progressive loading. Only the chunks needed for the current view are
downloaded, nearest to the center of view first, with coarser resolution
data shown while they stream in.

This example uses a multiscale light-sheet timelapse of zebrafish
development from `zebrahub.org <https://zebrahub.org>`_ (Royer Lab, CZ
Biohub). It requires network access; chunks are cached in memory.

.. tags:: experimental
"""

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)
from napari.experimental._progressive_loading_datasets import open_ome_zarr

URL = 'https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS002.ome.zarr/'

arrays, scale, _translate = open_ome_zarr(URL, num_levels=4)

viewer = napari.Viewer()
layer = add_progressive_loading_image(
    arrays,
    viewer=viewer,
    contrast_limits=(0, 1000),
    colormap='gray',
    scale=scale,
)

if __name__ == '__main__':
    napari.run()
