"""
Display one points layer ontop of one 4-D image layer using the
add_points and add_image APIs, where the markes are visible as nD objects
accross the dimensions, specified by their size
"""

import numpy as np
from skimage import data
import napari


with napari.gui_qt():
    blobs = np.stack(
        [
            data.binary_blobs(
                length=128, blob_size_fraction=0.05, n_dim=3, volume_fraction=f
            )
            for f in np.linspace(0.05, 0.5, 10)
        ],
        axis=0,
    )
    viewer = napari.view_image(blobs.astype(float))

    # add the points
    coords = np.array(
        [
            [0, 0, 100, 100],
            [0, 1, 50, 120],
            [1, 0, 100, 40],
            [2, 10, 110, 100],
            [9, 8, 80, 100],
        ]
    )
    text = ['(0, 0)', '(0, 1)', '(1, 0)', '(2, 10)', '(9, 8)']
    text_data = (coords, text)
    viewer.add_text(text_data)

    viewer.layers[1].text_color = 'green'
    viewer.layers[1].font_size = 20
