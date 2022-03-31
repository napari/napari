"""

"""
import napari
import numpy as np
from skimage import data

viewer = napari.Viewer(ndisplay=3)

# add a volume
blobs = data.binary_blobs(
    length=64, volume_fraction=0.1, n_dim=3
).astype(np.float32)

plane_layer = viewer.add_image(
    blobs,
    rendering='average',
    name='plane',
    depiction='plane',
    opacity=0.5,
    plane={'position': (32, 32, 32), 'normal': (1, 1, 1), 'thickness': 10}
)

labels_colormap = {
    0: np.array([0., 0., 0., 0.], dtype=np.float32),
    None: np.array([0., 0., 0., 1.], dtype=np.float32)
}
labels_layer = viewer.add_labels(np.zeros_like(blobs).astype(int), color=labels_colormap)
labels_layer.n_edit_dimensions = 3
labels_layer.experimental_linked_image_layer = plane_layer

napari.run()
