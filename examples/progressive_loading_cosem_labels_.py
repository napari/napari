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

import sys

try:
    import s3fs  # noqa: F401
except ImportError:
    print('This example requires s3fs: pip install s3fs')
    sys.exit(0)

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
    add_progressive_loading_labels,
)
from napari.experimental._progressive_loading_datasets import open_ome_zarr

BUCKET = 's3://janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.zarr/recon-1'
EM_PATH = f'{BUCKET}/em/fibsem-uint8'
LABEL_PATH = f'{BUCKET}/labels/groundtruth/crop124/all'

em_arrays, em_scale, em_translate = open_ome_zarr(EM_PATH, num_levels=5)
label_arrays, label_scale, label_translate = open_ome_zarr(
    LABEL_PATH, num_levels=5
)

viewer = napari.Viewer()
viewer.dims.ndisplay = 3

add_progressive_loading_image(
    em_arrays,
    viewer=viewer,
    name='EM (FIB-SEM)',
    contrast_limits=(0, 255),
    colormap='gray',
    scale=em_scale,
    translate=em_translate,
    rendering='attenuated_mip',
)

add_progressive_loading_labels(
    label_arrays,
    viewer=viewer,
    name='organelles (crop124)',
    scale=label_scale,
    translate=label_translate,
)

if __name__ == '__main__':
    napari.run()
