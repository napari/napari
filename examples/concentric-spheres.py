"""
Concentric spheres
==================

Display concentric spheres in 3D.

.. tags:: visualization-nD
"""

import numpy as np
from skimage import morphology

import napari

b0 = morphology.ball(5)

b1 = morphology.ball(10)

b0p = np.pad(b0, 5)

viewer = napari.Viewer(ndisplay=3)

# viewer.add_labels(b0)
viewer.add_labels(b0p)
viewer.add_labels(b1 * 2)

viewer.add_points([[10, 10, 10]], size=1)

if __name__ == '__main__':
    napari.run()
