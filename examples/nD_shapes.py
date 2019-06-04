"""
Display one 4-D image layer using the add_image API
"""

import numpy as np
from skimage import data
import napari
from napari.util import app_context


with app_context():
    blobs = data.binary_blobs(length=128, blob_size_fraction=0.05, n_dim=3,
                              volume_fraction=.1).astype(float)

    viewer = napari.view(blobs.astype(float))

    # Create random rectangles
    shapes = [[[i] + list(128*np.random.random(2)),
               [i] + list(128*np.random.random(2))] for i in range(128)]

    base_cols = ['red', 'green', 'blue', 'white', 'yellow', 'magenta', 'cyan']
    colors = [np.random.choice(base_cols) for i in range(128)]

    layer = viewer.add_shapes(np.array(shapes), shape_type='rectangle',
                              face_color=colors, name='sliced')

    masks = layer.to_masks(mask_shape=(128, 128, 128))
    labels = layer.to_labels(labels_shape=(128, 128, 128))
    shape_array = np.array(layer.to_list())

    print('sliced nshapes', layer.nshapes,
          'mask shape', masks.shape,
          'labels_shape', labels.shape,
          'array_shape', shape_array.shape)

    layer = viewer.add_shapes(np.array(shapes[0]), shape_type='rectangle',
                              broadcast=True, name='broadcasted')

    print('broadcast nshapes', layer.nshapes,
          'mask shape', masks.shape,
          'labels_shape', labels.shape,
          'array_shape', shape_array.shape)
