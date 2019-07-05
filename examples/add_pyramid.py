"""
Displays an image pyramid
"""

from skimage import data
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.transform import pyramid_gaussian
import napari
from napari.util import gui_qt
import numpy as np


# create pyramid from astronaut image
astronaut = rgb2gray(data.astronaut())
base = np.tile(astronaut, (16, 16))
pyramid = list(
    pyramid_gaussian(base, downscale=2, max_layer=5, multichannel=False)
)
pyramid = [img_as_ubyte(p) for p in pyramid]
print('pyramid level shapes: ', [p.shape for p in pyramid])

with gui_qt():
    # create the viewer
    viewer = napari.Viewer()

    # add image pyramid
    viewer.add_pyramid(pyramid, clim_range=[0, 255])
