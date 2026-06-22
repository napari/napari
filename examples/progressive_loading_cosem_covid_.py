"""
Progressive loading: COSEM SARS-CoV-2 infected cell
=====================================================

Browse a SARS-CoV-2 infected Vero (CCL-81) cell imaged by FIB-SEM
at ~5 nm resolution (6 450 x 5 000 x 3 750 voxels, uint16). Viral
particles are visible at full resolution inside the cell.

This dataset uses uint16 intensity values (unlike the uint8 COSEM
EM datasets), exercising the 16-bit texture upload path.

Requires network access and anonymous S3.

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
    '/jrc_ccl81-covid-1/jrc_ccl81-covid-1.zarr/recon-1/em/fibsem-uint16'
)
NUM_LEVELS = 5


def open_covid():
    """Open the COVID-infected cell volume through an in-memory cache."""
    store = CacheStore(
        FsspecStore.from_url(BUCKET, storage_options={'anon': True}),
        cache_store=MemoryStore(),
        max_size=int(2e9),
    )
    group = zarr.open_group(store, mode='r')
    arrays = [group[f's{level}'] for level in range(NUM_LEVELS)]
    ms = dict(group.attrs)['multiscales'][0]
    scale = ms['datasets'][0]['coordinateTransformations'][0]['scale']
    return arrays, scale


arrays, scale = open_covid()

viewer = napari.Viewer()
viewer.dims.ndisplay = 3

layer = add_progressive_loading_image(
    arrays,
    viewer=viewer,
    name='SARS-CoV-2 cell (FIB-SEM)',
    contrast_limits=(0, 65535),
    colormap='inferno',
    scale=scale,
    rendering='attenuated_mip',
)

if __name__ == '__main__':
    napari.run()
