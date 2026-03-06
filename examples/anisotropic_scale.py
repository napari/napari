"""
Anisotropic data with scale
============================

Display a 3D image with anisotropic voxel spacing using the ``scale``
parameter so that the volume appears with correct proportions.

Microscopy data is often anisotropic: the voxel spacing along the Z axis
is larger than along X and Y.  Without ``scale``, napari treats every
voxel as a unit cube and the volume looks compressed along Z.
Setting ``scale`` to the real voxel dimensions corrects this.

Toggle the visibility of each layer to compare the difference.

.. tags:: visualization-nD, layers
"""

import numpy as np
from skimage import data

import napari

# Subsample Z by 4x to simulate strongly anisotropic data
cells = data.cells3d()
nuclei = np.ascontiguousarray(cells[::4, 1])

# Z voxel spacing is ~4.5x larger than XY after subsampling
scale = (4.5, 1, 1)

viewer = napari.Viewer(ndisplay=3)

viewer.add_image(
    nuclei,
    name='no scale',
    rendering='mip',
    blending='additive',
    colormap='magenta',
)

viewer.add_image(
    nuclei,
    name='with scale',
    rendering='mip',
    blending='additive',
    colormap='green',
    scale=scale,
)

viewer.camera.angles = (-25, 25, -140)
viewer.camera.zoom = 1.5

if __name__ == '__main__':
    napari.run()
