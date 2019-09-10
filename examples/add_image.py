"""
Display one image using the add_image API.
"""

from skimage import data
from skimage.color import rgb2gray
import napari
import os.path


with napari.gui_qt():
    # create the viewer with an image
    viewer = napari.view(
        astronaut=rgb2gray(data.astronaut()), title='napari example'
    )
    viewer.title = os.path.splitext(os.path.basename(__file__))[0]
