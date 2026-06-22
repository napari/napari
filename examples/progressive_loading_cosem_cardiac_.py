"""
Progressive loading: COSEM zebrafish heart
==========================================

Stream through the largest volume in the COSEM collection — a zebrafish
heart imaged by FIB-SEM at ~5 nm isotropic resolution. The full volume
is 40 152 x 19 814 x 20 629 voxels (~16 billion voxels) with a 15-level
multiscale pyramid, making it an ideal stress test for progressive
loading.

Only the chunks visible in the current camera frustum are downloaded,
nearest-first, with coarser data shown as a backdrop while fine tiles
stream in. Try zooming into a sarcomere or mitochondrion.

Requires network access and anonymous S3 (no credentials needed).

.. tags:: experimental
"""

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

BUCKET = (
    's3://janelia-cosem-datasets'
    '/jrc_zf-cardiac-1/jrc_zf-cardiac-1.zarr/recon-1/em/fibsem-uint8'
)
NUM_LEVELS = 15


def open_cardiac():
    """Open the zebrafish cardiac volume through an in-memory cache."""
    store = CacheStore(
        FsspecStore.from_url(BUCKET, storage_options={'anon': True}),
        cache_store=MemoryStore(),
        max_size=int(4e9),
    )
    group = zarr.open_group(store, mode='r')
    arrays = [group[f's{level}'] for level in range(NUM_LEVELS)]
    ms = dict(group.attrs)['multiscales'][0]
    scale = ms['datasets'][0]['coordinateTransformations'][0]['scale']
    return arrays, scale


arrays, scale = open_cardiac()

viewer = napari.Viewer()
viewer.dims.ndisplay = 3

layer = add_progressive_loading_image(
    arrays,
    viewer=viewer,
    name='Zebrafish heart (FIB-SEM)',
    contrast_limits=(0, 255),
    colormap='turbo',
    scale=scale,
    rendering='attenuated_mip',
)

if __name__ == '__main__':
    napari.run()
