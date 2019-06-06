"""
Displays an image pyramid
"""

from skimage import data
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.transform import pyramid_gaussian
import napari
from napari.util import app_context
import numpy as np


# create pyramid from astronaut image
astronaut = rgb2gray(data.astronaut())
base = np.tile(astronaut, (4, 4))
pyramid = list(pyramid_gaussian(base, downscale=2, max_layer=4,
                                multichannel=False))
pyramid = [np.array([p * (abs(5 - i) + 1) / 6 for i in range(10)])
           for p in pyramid]
pyramid = [img_as_ubyte(p) for p in pyramid]
print('pyramid level shapes: ', [p.shape for p in pyramid])

with app_context():
    # create the viewer
    viewer = napari.Viewer()

    # add image pyramid
    viewer.add_pyramid(pyramid, clim_range=[0, 255])
