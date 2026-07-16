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

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)
from napari.experimental._progressive_loading_datasets import open_ome_zarr

BUCKET = (
    's3://janelia-cosem-datasets'
    '/jrc_zf-cardiac-1/jrc_zf-cardiac-1.zarr/recon-1/em/fibsem-uint8'
)

arrays, scale, translate = open_ome_zarr(BUCKET, num_levels=15)

viewer = napari.Viewer()
viewer.dims.ndisplay = 3

layer = add_progressive_loading_image(
    arrays,
    viewer=viewer,
    name='Zebrafish heart (FIB-SEM)',
    contrast_limits=(0, 255),
    colormap='turbo',
    scale=scale,
    translate=translate,
    rendering='attenuated_mip',
)

if __name__ == '__main__':
    napari.run()
