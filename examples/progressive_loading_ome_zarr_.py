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

import zarr
from zarr.experimental.cache_store import CacheStore
from zarr.storage import FsspecStore, MemoryStore

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)

URL = 'https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS002.ome.zarr/'
NUM_LEVELS = 4


def open_zebrahub():
    """Open the zebrahub multiscale levels through an in-memory cache."""
    store = CacheStore(
        FsspecStore.from_url(URL),
        cache_store=MemoryStore(),
        max_size=int(4e9),
    )
    group = zarr.open_group(store, mode='r')
    arrays = [group[str(level)] for level in range(NUM_LEVELS)]
    ms = dict(group.attrs)['multiscales'][0]
    scale = ms['datasets'][0]['coordinateTransformations'][0]['scale']
    return arrays, scale


arrays, scale = open_zebrahub()

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
