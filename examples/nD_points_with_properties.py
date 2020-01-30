"""
Display one points layer ontop of one 4-D image layer using the
add_points and add_image APIs, where the markes are visible as nD objects
accross the dimensions, specified by their size
"""

from math import ceil

import numpy as np
from skimage import data
import napari


with napari.gui_qt():
    blobs = data.binary_blobs(
                length=100, blob_size_fraction=0.05, n_dim=3, volume_fraction=0.05
            )
    viewer = napari.view_image(blobs.astype(float))

    # create the points
    points = []
    for z in range(blobs.shape[0]):
        points += [
                [z, 25, 25],
                [z, 25, 75],
                [z, 75, 25],
                [z, 75, 75]
        ]

    # create the property for setting the face and edge color.
    face_property = np.array(
        [True, True, True, True, False, False, False, False] * int((blobs.shape[0] / 2))
    )
    edge_property = np.array(['A', 'B', 'C', 'D', 'E'] * int(len(points) / 5))

    properties = {
        'face_property': face_property,
        'edge_property': edge_property,
    }

    points_layer = viewer.add_points(
        points,
        properties=properties,
        size=3,
        edge_width=5,
        edge_color='edge_property',
        face_color='face_property',
        n_dimensional=False,
    )

    # change the face color cycle
    points_layer.face_color_cycle = ['white', 'black']

    # change the edge_color cycle.
    # there are 4 colors for 5 categories, so 'c' will be recycled
    points_layer.edge_color_cycle = ['c', 'm', 'y', 'k']
