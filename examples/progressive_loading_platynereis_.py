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

import napari_colormaps  # noqa: F401 - registers colormaps
import zarr
from zarr.experimental.cache_store import CacheStore
from zarr.storage import FsspecStore, MemoryStore

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)

URL = 'https://s3.embl.de/i2k-2020/platy-raw.ome.zarr'
NUM_LEVELS = 10


def open_platynereis():
    """Open the Platynereis volume through an in-memory cache."""
    store = CacheStore(
        FsspecStore.from_url(URL),
        cache_store=MemoryStore(),
        max_size=int(4e9),
    )
    group = zarr.open_group(store, mode='r')
    ms = dict(group.attrs)['multiscales'][0]
    datasets = ms['datasets']
    arrays = [group[d['path']] for d in datasets[:NUM_LEVELS]]
    scale = datasets[0]['coordinateTransformations'][0]['scale']
    return arrays, scale


arrays, scale = open_platynereis()

viewer = napari.Viewer()
viewer.dims.ndisplay = 3

layer = add_progressive_loading_image(
    arrays,
    viewer=viewer,
    name='Platynereis (serial-section EM)',
    contrast_limits=(0, 255),
    colormap='bioluminescent',
    scale=scale,
    rendering='attenuated_mip',
)

if __name__ == '__main__':
    napari.run()
