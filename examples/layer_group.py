import napari
import numpy as np
from skimage import data


points = np.random.rand(10, 3) * 128
colors = np.random.rand(10, 3)

blobs = data.binary_blobs(length=128, volume_fraction=0.05, n_dim=3)

viewer = napari.Viewer(ndisplay=3)

pl = napari.layers.Points(points, face_color=colors)
il = napari.layers.Image(blobs)

viewer.add_layer([pl, il])

napari.run()
