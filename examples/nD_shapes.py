"""
nD shapes
=========

Display one 4-D image layer using the ``add_image`` API

.. tags:: visualization-nD
"""

import numpy as np
from skimage import data
import napari


blobs = data.binary_blobs(
    length=128, blob_size_fraction=0.05, n_dim=3, volume_fraction=0.1
).astype(float)

viewer = napari.view_image(blobs.astype(float))

# create one random polygon per "plane"
planes = np.tile(np.arange(128).reshape((128, 1, 1)), (1, 5, 1))
np.random.seed(0)
corners = np.random.uniform(0, 128, size=(128, 5, 2))
shapes = np.concatenate((planes, corners), axis=2)

base_cols = ['red', 'green', 'blue', 'white', 'yellow', 'magenta', 'cyan']
colors = np.random.choice(base_cols, size=128)

layer = viewer.add_shapes(
    np.array(shapes),
    shape_type='polygon',
    face_color=colors,
    name='sliced',
)

masks = layer.to_masks(mask_shape=(128, 128, 128))
labels = layer.to_labels(labels_shape=(128, 128, 128))
shape_array = np.array(layer.data)

print(
    f'sliced: nshapes {layer.nshapes}, mask shape {masks.shape}, '
    f'labels_shape {labels.shape}, array_shape, {shape_array.shape}'
)

corners = np.random.uniform(0, 128, size=(2, 2))
layer = viewer.add_shapes(corners, shape_type='rectangle', name='broadcasted')

masks = layer.to_masks(mask_shape=(128, 128))
labels = layer.to_labels(labels_shape=(128, 128))
shape_array = np.array(layer.data)

print(
    f'broadcast: nshapes {layer.nshapes}, mask shape {masks.shape}, '
    f'labels_shape {labels.shape}, array_shape, {shape_array.shape}'
)

if __name__ == '__main__':
    napari.run()
