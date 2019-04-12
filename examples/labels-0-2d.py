"""
Display a labels layer above of an image layer using the add_labels and
add_image APIs
"""

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square, remove_small_objects
from napari import ViewerApp
from napari.util import app_context


with app_context():
    image = data.coins()[50:-50, 50:-50]

    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(4))

    # remove artifacts connected to image border
    cleared = remove_small_objects(clear_border(bw), 20)

    # label image regions
    label_image = label(cleared)

    # initialise viewer with astro image
    viewer = ViewerApp(coins=image, multichannel=False)
    viewer.layers[0].colormap = 'gray'

    # add the labels
    label_layer = viewer.add_labels(label_image, name='segmentation')
