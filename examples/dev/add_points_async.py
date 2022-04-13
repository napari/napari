# A simple driving example to test async slicing.

from skimage import data
import numpy as np
import napari

# 10 million points behaves undesirably on my macbook pro on main (with sync)
np.random.seed(0)
n = 10_000_000
data = 1000 * np.random.rand(n, 3)
viewer = napari.view_points(data)

if __name__ == '__main__':
    napari.run()
