"""
Progressive loading: COSEM labels overlay
==========================================

Load a COSEM FIB-SEM volume with organelle segmentation labels
using progressive loading. Both the EM image and the labels stream
in at the resolution matching the current view.

This example uses the ``jrc_mus-liver`` dataset from
`OpenOrganelle / COSEM <https://openorganelle.janelia.org>`_
(Janelia Research Campus). It requires network access; chunks are
cached in memory.

The labels layer uses ``add_progressive_loading_labels`` to render
mitochondria instance segmentations from groundtruth crop124.

.. tags:: experimental
"""

import zarr
from zarr.experimental.cache_store import CacheStore
from zarr.storage import FsspecStore, MemoryStore

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
    add_progressive_loading_labels,
)

BUCKET = 's3://janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.zarr/recon-1'
EM_PATH = f'{BUCKET}/em/fibsem-uint8'
LABEL_PATH = f'{BUCKET}/labels/groundtruth/crop124/all'
NUM_EM_LEVELS = 5
NUM_LABEL_LEVELS = 5


def open_cosem(path, num_levels):
    """Open a COSEM multiscale group through an in-memory cache."""
    store = CacheStore(
        FsspecStore.from_url(path, anon=True),
        cache_store=MemoryStore(),
        max_size=int(2e9),
    )
    group = zarr.open_group(store, mode='r')
    arrays = [group[f's{level}'] for level in range(num_levels)]
    ms = dict(group.attrs)['multiscales'][0]
    scale = ms['datasets'][0]['coordinateTransformations'][0]['scale']
    return arrays, scale


em_arrays, em_scale = open_cosem(EM_PATH, NUM_EM_LEVELS)
label_arrays, label_scale = open_cosem(LABEL_PATH, NUM_LABEL_LEVELS)

viewer = napari.Viewer()
viewer.dims.ndisplay = 3

add_progressive_loading_image(
    em_arrays,
    viewer=viewer,
    name='EM (FIB-SEM)',
    contrast_limits=(0, 255),
    colormap='gray',
    scale=em_scale,
    rendering='attenuated_mip',
)

add_progressive_loading_labels(
    label_arrays,
    viewer=viewer,
    name='organelles (crop124)',
    scale=label_scale,
)

if __name__ == '__main__':
    napari.run()
