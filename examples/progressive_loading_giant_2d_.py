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

import dask.array as da
try:
    import napari_colormaps  # noqa: F401 - registers colormaps
except ModuleNotFoundError:
    pass
import zarr
from zarr.experimental.cache_store import CacheStore
from zarr.storage import FsspecStore, MemoryStore

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)

URL = 'https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/4495402.zarr/'
NUM_LEVELS = 8


def open_giant_2d():
    """Open the IDR giant 2D image through an in-memory cache.

    The raw arrays are 5D (1,1,1,H,W); we squeeze to 2D via dask so
    the progressive loader sees a plain (H, W) image.
    """
    store = CacheStore(
        FsspecStore.from_url(URL),
        cache_store=MemoryStore(),
        max_size=int(4e9),
    )
    group = zarr.open_group(store, mode='r', zarr_format=2)
    ms = dict(group.attrs)['multiscales'][0]
    datasets = ms['datasets']
    arrays = [
        da.from_zarr(group[d['path']]).squeeze()
        for d in datasets[:NUM_LEVELS]
    ]
    return arrays


arrays = open_giant_2d()

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
