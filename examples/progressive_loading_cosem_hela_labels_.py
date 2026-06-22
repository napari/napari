"""
Progressive loading: COSEM HeLa cell with organelle labels
===========================================================

Load a whole HeLa cell imaged by FIB-SEM (6 368 x 1 600 x 12 000
voxels, 4 x 4 x 5.24 nm) together with organelle instance
segmentation labels from the COSEM project. Both the EM intensity
and the label overlay stream in progressively.

The labels demonstrate ``add_progressive_loading_labels`` on a
real segmentation (crop155, 800³ voxels, 6 levels), with
mitochondria, ER, Golgi, and other organelles visible as you
navigate through the cell.

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
    add_progressive_loading_labels,
)

DATASET = 's3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1'
EM_PATH = f'{DATASET}/em/fibsem-uint8'
LABEL_PATH = f'{DATASET}/labels/groundtruth/crop155/all'
NUM_EM_LEVELS = 6
NUM_LABEL_LEVELS = 6


def open_cosem(path, num_levels):
    """Open a COSEM multiscale group through an in-memory cache."""
    store = CacheStore(
        FsspecStore.from_url(path, storage_options={'anon': True}),
        cache_store=MemoryStore(),
        max_size=int(2e9),
    )
    group = zarr.open_group(store, mode='r')
    arrays = [group[f's{level}'] for level in range(num_levels)]
    ms = dict(group.attrs)['multiscales'][0]
    transforms = ms['datasets'][0]['coordinateTransformations']
    scale = transforms[0]['scale']
    translate = [0.0] * len(scale)
    for t in transforms:
        if t.get('type') == 'translation':
            translate = t['translation']
            break
    return arrays, scale, translate


em_arrays, em_scale, em_translate = open_cosem(EM_PATH, NUM_EM_LEVELS)
label_arrays, label_scale, label_translate = open_cosem(
    LABEL_PATH, NUM_LABEL_LEVELS
)

viewer = napari.Viewer()
viewer.dims.ndisplay = 3

add_progressive_loading_image(
    em_arrays,
    viewer=viewer,
    name='HeLa EM (FIB-SEM)',
    contrast_limits=(0, 255),
    colormap='gray',
    scale=em_scale,
    translate=em_translate,
    rendering='attenuated_mip',
)

add_progressive_loading_labels(
    label_arrays,
    viewer=viewer,
    name='organelles (crop4)',
    scale=label_scale,
    translate=label_translate,
)

if __name__ == '__main__':
    napari.run()
