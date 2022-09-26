import argparse
from skimage import data
import numpy as np
import napari

"""
Stress the points layer by generating a large number of points.
"""

parser = argparse.ArgumentParser()
parser.add_argument("n", type=int, default=10_000_000)
args = parser.parse_args()

np.random.seed(0)
n = args.n
data = 1000 * np.random.rand(n, 3)
viewer = napari.view_points(data)

if __name__ == '__main__':
    napari.run()
