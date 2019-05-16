"""
Display a labels layer above of an image layer using the add_labels and
add_image APIs
"""

from imageio import imread
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square, remove_small_objects
from napari import ViewerApp
from napari.util import app_context


coins = imread('imageio:coins.png')[50:-50, 50:-50]

with app_context():

    # apply threshold
    thresh = threshold_otsu(coins)
    bw = closing(coins > thresh, square(4))

    # remove artifacts connected to image border
    cleared = remove_small_objects(clear_border(bw), 20)

    # label image regions
    label_image = label(cleared)

    # initialise viewer with astro image
    viewer = ViewerApp(coins=coins, multichannel=False)
    viewer.layers[0].colormap = 'gray'

    # add the labels
    label_layer = viewer.add_labels(label_image, name='segmentation')
