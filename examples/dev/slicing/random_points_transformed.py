from skimage import data
import numpy as np
import napari

np.random.seed(0)
n = 1000
data = 100 * np.random.rand(n, 3)

translate = [0, 10, -20]
scale = [1, 2, 3]
viewer = napari.view_points(data, translate=translate, scale=scale)

if __name__ == '__main__':
    napari.run()
