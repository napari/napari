"""
Add labels with features
========================

Display a labels layer with various features

.. tags:: layers, analysis
"""


import numpy as np
from skimage import data
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import closing, remove_small_objects, square
from skimage.segmentation import clear_border

import napari

image = data.coins()[50:-50, 50:-50]

# apply threshold
thresh = threshold_otsu(image)
bw = closing(image > thresh, square(4))

# remove artifacts connected to image border
cleared = remove_small_objects(clear_border(bw), 20)

# label image regions
label_image = label(cleared)

# initialise viewer with coins image
viewer = napari.view_image(image, name='coins', rgb=False)

# get the size of each coin (first element is background area)
label_areas = np.bincount(label_image.ravel())[1:]

# split coins into small or large
size_range = max(label_areas) - min(label_areas)
small_threshold = min(label_areas) + (size_range / 2)
coin_sizes = np.where(label_areas > small_threshold, 'large', 'small')

label_features = {
    'row': ['none']
    + ['top'] * 4
    + ['bottom'] * 4,  # background is row: none
    'size': ['none'] + list(coin_sizes),  # background is size: none
}

color = {1: 'white', 2: 'blue', 3: 'green', 4: 'red', 5: 'yellow'}

# add the labels
label_layer = viewer.add_labels(
    label_image,
    name='segmentation',
    features=label_features,
    color=color,
)

if __name__ == '__main__':
    napari.run()
