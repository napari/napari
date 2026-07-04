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

import contextlib

with contextlib.suppress(ModuleNotFoundError):
    import napari_colormaps  # noqa: F401 - registers colormaps

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
    add_progressive_loading_labels,
)
from napari.experimental._progressive_loading_datasets import open_ome_zarr

DATASET = 's3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1'
EM_PATH = f'{DATASET}/em/fibsem-uint8'
LABEL_PATH = f'{DATASET}/labels/groundtruth/crop155/all'

em_arrays, em_scale, em_translate = open_ome_zarr(EM_PATH, num_levels=6)
label_arrays, label_scale, label_translate = open_ome_zarr(
    LABEL_PATH, num_levels=6
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
