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

import contextlib

with contextlib.suppress(ModuleNotFoundError):
    import napari_colormaps  # noqa: F401 - registers colormaps

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)
from napari.experimental._progressive_loading_datasets import open_ome_zarr

BUCKET = (
    's3://janelia-cosem-datasets'
    '/jrc_ccl81-covid-1/jrc_ccl81-covid-1.zarr/recon-1/em/fibsem-uint16'
)

arrays, scale, translate = open_ome_zarr(BUCKET, num_levels=5)

viewer = napari.Viewer()
viewer.dims.ndisplay = 3

layer = add_progressive_loading_image(
    arrays,
    viewer=viewer,
    name='SARS-CoV-2 cell (FIB-SEM)',
    contrast_limits=(0, 65535),
    colormap='inferno',
    scale=scale,
    translate=translate,
    rendering='attenuated_mip',
)

if __name__ == '__main__':
    napari.run()
