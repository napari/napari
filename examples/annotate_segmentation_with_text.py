"""
Annotate segmentation with text
===============================

Perform a segmentation and annotate the results with
bounding boxes and text

.. tags:: analysis
"""
import numpy as np
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops_table
from skimage.morphology import closing, square, remove_small_objects
import napari


def segment(image):
    """Segment an image using an intensity threshold determined via
    Otsu's method.

    Parameters
    ----------
    image : np.ndarray
        The image to be segmented

    Returns
    -------
    label_image : np.ndarray
        The resulting image where each detected object labeled with a unique integer.
    """
    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(4))

    # remove artifacts connected to image border
    cleared = remove_small_objects(clear_border(bw), 20)

    # label image regions
    label_image = label(cleared)

    return label_image


def make_bbox(bbox_extents):
    """Get the coordinates of the corners of a
    bounding box from the extents

    Parameters
    ----------
    bbox_extents : list (4xN)
        List of the extents of the bounding boxes for each of the N regions.
        Should be ordered: [min_row, min_column, max_row, max_column]

    Returns
    -------
    bbox_rect : np.ndarray
        The corners of the bounding box. Can be input directly into a
        napari Shapes layer.
    """
    minr = bbox_extents[0]
    minc = bbox_extents[1]
    maxr = bbox_extents[2]
    maxc = bbox_extents[3]

    bbox_rect = np.array(
        [[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]]
    )
    bbox_rect = np.moveaxis(bbox_rect, 2, 0)

    return bbox_rect


def circularity(perimeter, area):
    """Calculate the circularity of the region

    Parameters
    ----------
    perimeter : float
        the perimeter of the region
    area : float
        the area of the region

    Returns
    -------
    circularity : float
        The circularity of the region as defined by 4*pi*area / perimeter^2
    """
    circularity = 4 * np.pi * area / (perimeter ** 2)

    return circularity


# load the image and segment it
image = data.coins()[50:-50, 50:-50]
label_image = segment(image)

# create the features dictionary
features = regionprops_table(
    label_image, properties=('label', 'bbox', 'perimeter', 'area')
)
features['circularity'] = circularity(
    features['perimeter'], features['area']
)

# create the bounding box rectangles
bbox_rects = make_bbox([features[f'bbox-{i}'] for i in range(4)])

# specify the display parameters for the text
text_parameters = {
    'string': 'label: {label}\ncirc: {circularity:.2f}',
    'size': 12,
    'color': 'green',
    'anchor': 'upper_left',
    'translation': [-3, 0],
}

# initialise viewer with coins image
viewer = napari.view_image(image, name='coins', rgb=False)

# add the labels
label_layer = viewer.add_labels(label_image, name='segmentation')

shapes_layer = viewer.add_shapes(
    bbox_rects,
    face_color='transparent',
    edge_color='green',
    features=features,
    text=text_parameters,
    name='bounding box',
)

if __name__ == '__main__':
    napari.run()
