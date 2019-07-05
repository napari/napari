"""
Display one 4-D image layer using the add_image API
"""

import numpy as np
from skimage import data
import napari
from napari.util import gui_qt


with gui_qt():
    blobs = data.binary_blobs(
        length=128, blob_size_fraction=0.05, n_dim=3, volume_fraction=0.1
    ).astype(float)

    viewer = napari.view(blobs.astype(float))

    # create one random rectangle per "plane"
    planes = np.tile(np.arange(128).reshape((128, 1, 1)), (1, 2, 1))
    corners = np.random.uniform(0, 128, size=(128, 2, 2))
    shapes = np.concatenate((planes, corners), axis=2)

    base_cols = ['red', 'green', 'blue', 'white', 'yellow', 'magenta', 'cyan']
    colors = np.random.choice(base_cols, size=128)

    layer = viewer.add_shapes(
        np.array(shapes),
        shape_type='rectangle',
        face_color=colors,
        name='sliced',
    )

    masks = layer.to_masks(mask_shape=(128, 128, 128))
    labels = layer.to_labels(labels_shape=(128, 128, 128))
    shape_array = np.array(layer.to_list())

    print(
        f'sliced: nshapes {layer.nshapes}, mask shape {masks.shape}, '
        f'labels_shape {labels.shape}, array_shape, {shape_array.shape}'
    )

    corners = np.random.uniform(0, 128, size=(2, 2))
    layer = viewer.add_shapes(
        corners, shape_type='rectangle', name='broadcasted'
    )

    masks = layer.to_masks(mask_shape=(128, 128, 128))
    labels = layer.to_labels(labels_shape=(128, 128, 128))
    shape_array = np.array(layer.to_list())

    print(
        f'broadcast: nshapes {layer.nshapes}, mask shape {masks.shape}, '
        f'labels_shape {labels.shape}, array_shape, {shape_array.shape}'
    )
