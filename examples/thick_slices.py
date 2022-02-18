import napari
import numpy as np
import skimage

points = np.random.rand(1000, 4) * 300
vec = (np.random.rand(1000, 4) - 0.5) * 30
vec = np.stack([points, vec], axis=1)
color = np.random.rand(1000, 3)
img = skimage.data.binary_blobs(50, n_dim=4)

viewer = napari.Viewer()

viewer.add_image(img, scale=[6, 6, 6, 6])
viewer.add_points(points, face_color=color)
viewer.add_vectors(vec, edge_color=color)

viewer.dims.thickness = 50, 50, 1, 1

napari.run()
