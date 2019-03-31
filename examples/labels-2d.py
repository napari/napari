"""
Display a labels layer above of an image layer using the add_labels and
add_image APIs
"""

from skimage import data
from skimage.color import rgb2gray
from skimage.segmentation import slic
from napari import ViewerApp
from napari.util import app_context


with app_context():
    astro = data.astronaut()

    # initialise viewer with astro image
    viewer = ViewerApp(astronaut=rgb2gray(astro), multichannel=False)
    viewer.layers[0].colormap = 'gray'

    # add the labels
    # we add 1 because SLIC returns labels from 0, which we consider background
    labels = slic(astro, multichannel=True, compactness=20) + 1
    label_layer = viewer.add_labels(labels, name='segmentation')
    print(f'The color of label 5 is {label_layer.label_color(5)}')
