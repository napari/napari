"""
Display one image using the add_image API.
"""

image_web_url = 'https://github.com/napari/napari/raw/master/docs/source/img/napari_logo.png'
image_local_filename = '../docs/source/img/napari_logo.png'

from skimage import data
import napari

with napari.gui_qt():
    # create the viewer
    viewer = napari.Viewer()

    # open locally stored image
    layers = viewer.open(image_local_filename)
    print("Image " + layers[0].source + " loaded.")

    # open image from web
    layers = viewer.open(image_web_url)
    print("Image " + layers[0].source + " loaded.")
