import numpy as np
import napari

image_data = np.random.rand(1000, 512, 512)

np.random.seed(0)
n = 10_000_000
points_data = image_data.shape * np.random.rand(n, 3)

viewer = napari.Viewer()
viewer.add_image(image_data)
viewer.add_points(points_data)

if __name__ == '__main__':
    napari.run()
