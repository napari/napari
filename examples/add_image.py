"""
Display one image using the add_image API.
"""

from skimage import data
import napari
from scipy.fftpack import fftn, fftshift
import numpy as np

with napari.gui_qt():
    # create the viewer with an image

    viewer = napari.view_image(np.log(fftshift(fftn(data.astronaut().mean(-1)))))
