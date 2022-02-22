import napari
import numpy as np
import skimage


N = 100
ndim = 4
scale = 300

pts = np.random.rand(N, ndim) * scale
vec = (np.random.rand(N, ndim) - 0.5) * scale / 10
vec = np.stack([pts, vec], axis=1)

color = np.random.rand(N, 3)

img = skimage.data.binary_blobs(50, n_dim=ndim)

viewer = napari.Viewer()

viewer.add_image(img, scale=[scale / 50] * ndim)
viewer.add_points(pts, face_color=color)
viewer.add_vectors(vec, edge_color=color)

viewer.dims.set_thickness(axis=(0, 1), value=(50, 50))

napari.run()
