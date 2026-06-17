# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "napari[pyqt5]",
#     "zarr>=3.1.6",
#     "fsspec>=2023.10.0",
#     "aiohttp",
#     "requests",
# ]
#
# [tool.uv.sources]
# napari = { git = "https://github.com/kephale/napari", branch = "progressive-loading-rebase" }
# ///
"""
Progressive loading: zebrahub via uv
====================================

Self-contained launcher for the zebrahub progressive-loading demo: the
inline script metadata above lets ``uv run`` install napari from a
pinned commit of this branch and resolve every dependency without a
clone or a virtualenv::

    uv run progressive_loading_zebrahub_uv.py

Edit ``rev`` in the header to launch any other commit/branch/tag; uv
caches the built wheel per rev, so switching between two commits for an
A/B is fast after the first build of each.

Same dataset as :ref:`progressive_loading_ome_zarr`: a multiscale
light-sheet timelapse of zebrafish development from
`zebrahub.org <https://zebrahub.org>`_ (Royer Lab, CZ Biohub). Requires
network access; chunks are cached in memory.

Knobs worth A/B-ing (set in the environment before launch)::

    NAPARI_PROGRESSIVE_TILE_MAX_BYTES_3D=16e6|24e6|33e6  # sharpness/fluidity dial
    NAPARI_PROGRESSIVE_TILE_MARGIN_3D=1.25               # pan slack
    NAPARI_PROGRESSIVE_FETCH_WORKERS=2|4

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
