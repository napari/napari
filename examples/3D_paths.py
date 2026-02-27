"""
3D Paths
========

Display two path shapes on top of a 3D image layer. One path spans multiple
Z slices (red) and one is flat on the Z=0 plane (blue).

With ``out_of_slice_display=True``, the 3D path is visible even in
2D view — it is shown on every slice that falls within its Z range.
Switch to 2D view and scroll through the Z slider to see both paths.

.. tags:: visualization-advanced, layers
"""

import numpy as np
from skimage import data

import napari

blobs = data.binary_blobs(
    length=128, blob_size_fraction=0.05, n_dim=3, volume_fraction=0.05
)

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(blobs.astype(float))

# sample vector coord-like data
path = np.array([np.array([[0, 0, 0], [0, 10, 10], [0, 5, 15], [20, 5, 15],
    [56, 70, 21], [127, 127, 127]]),
    np.array([[0, 0, 0], [0, 10, 10], [0, 5, 15], [0, 5, 15],
        [0, 70, 21], [0, 127, 127]])])

print('Path', path.shape)
layer = viewer.add_shapes(
    path,
    shape_type='path',
    edge_width=4,
    edge_color=['red', 'blue'],
    out_of_slice_display=True,
)

if __name__ == '__main__':
    napari.run()
