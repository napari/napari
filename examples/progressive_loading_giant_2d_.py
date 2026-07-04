"""
Progressive loading: 350-gigapixel zebrafish embryo EM
======================================================

Browse a 380 928 x 921 600 pixel (~350 gigapixel) sagittal section of
a zebrafish (*Danio rerio*) embryo with a Google Maps-like pan-and-zoom
experience. The image is a serial-section electron microscopy montage
at **1.6 nm/pixel** from the IDR (idr0053, Faas et al., *J. Cell
Biol.* 2012 — "Virtual nanoscopy").

The 8-level multiscale pyramid is served over HTTPS from the Image
Data Resource; only the 1024x1024 chunks overlapping your view are
fetched. Try zooming all the way in to see ultrastructural detail,
then zoom out to see the entire embryo.

Requires network access (public HTTPS, no credentials).

.. tags:: experimental
"""

import contextlib

with contextlib.suppress(ModuleNotFoundError):
    import napari_colormaps  # noqa: F401 - registers colormaps

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)
from napari.experimental._progressive_loading_datasets import open_ome_zarr

# 5D (1,1,1,H,W) IDR v0.1 data; open_ome_zarr squeezes to 2D
URL = 'https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/4495402.zarr/'

arrays, _scale, _translate = open_ome_zarr(URL, num_levels=8, zarr_format=2)

viewer = napari.Viewer()

layer = add_progressive_loading_image(
    arrays,
    viewer=viewer,
    name='Zebrafish embryo EM (350 Gpx)',
    contrast_limits=(0, 255),
    colormap='cyan',
)

if __name__ == '__main__':
    napari.run()
