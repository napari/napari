"""
Display one markers layer ontop of one 4-D image layer using the
add_markers and add_image APIs, where the markes are visible as nD objects
accross the dimensions, specified by their size
"""

import numpy as np
from skimage import data
from napari import ViewerApp
from napari.util import app_context


with app_context():
    # TODO: change axis back to 0 once dims displayed in sane order
    blobs = np.stack([data.binary_blobs(length=128, blob_size_fraction=0.05,
                                        n_dim=3, volume_fraction=f)
                      for f in np.linspace(0.05, 0.5, 10)], axis=-1)
    viewer = ViewerApp(blobs.astype(float))
    # add the markers
    markers = np.array([[100, 100, 0, 0], [50, 120, 0, 0], [100, 40, 1, 0],
                       [110, 20, 100, 2], [400, 100, 10, 8]])
    viewer.add_markers(markers, size=[10, 10, 6, 0], face_color='blue',
                       n_dimensional=True)
