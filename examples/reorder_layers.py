"""
Add four image layers using the add_image API and then reorder them using the
layers swap method and remove one using the layers pop method
"""

from skimage import data
from skimage.color import rgb2gray
from napari import Window, Viewer
from napari.util import app_context


with app_context():
    # create the viewer and window
    viewer = Viewer()
    window = Window(viewer)
    # add the first image
    viewer.add_image(rgb2gray(data.astronaut()))
    # add the second image
    viewer.add_image(data.camera())
    # add the third image
    viewer.add_image(data.coins())
    # add the fourth image
    viewer.add_image(data.moon())
    # Remove the camera
    viewer.layers.remove(viewer.layers[1])
