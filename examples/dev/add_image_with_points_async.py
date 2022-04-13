# A simple driving example to test async slicing.

from skimage import data
import numpy as np
import napari

image_data = data.brain()

# 10 million points behaves undesirably on my macbook pro on main (with sync)
np.random.seed(0)
n = 10_000_000
points_data = image_data.shape * np.random.rand(n, 3)

viewer = napari.Viewer()
viewer.add_image(image_data)
viewer.add_points(points_data)

if __name__ == '__main__':
    napari.run()
