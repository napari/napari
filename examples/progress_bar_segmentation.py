"""
Use napari's tqdm wrapper to display the progress of long-running operations
in the viewer.  
"""
import numpy as np
import napari

from time import sleep
from napari.utils.progress import progress
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

from skimage.filters import (
    threshold_isodata,
    threshold_li,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)
from skimage.data import cells3d
from skimage.measure import label

# we will try each of these thresholds on our image
all_thresholds = [
    threshold_isodata,
    threshold_li,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
]

viewer = napari.Viewer()

# load cells data and take just nuclei
cells_im = cells3d()
cell_nuclei = cells_im[:, 1, :, :]
viewer.add_image(
    cell_nuclei, name="Nuclei", blending='additive', colormap="green"
)


def try_thresholds():
    """Tries each threshold, and adds result to viewer."""
    thresholded_nuclei = []

    # we wrap our iterable with `progress`
    # this will automatically add a progress bar to our activity dock
    for threshold_func in progress(all_thresholds):
        current_threshold = threshold_func(cell_nuclei)
        binarised_im = cell_nuclei > current_threshold
        thresholded_nuclei.append(binarised_im)

        # uncomment if processing is too fast
        sleep(0.5)

    # working with a wrapped iterable, the progress bar will be closed
    # as soon as the iteration is complete

    binarised_nuclei = np.stack(thresholded_nuclei)
    viewer.add_labels(
        binarised_nuclei,
        color={1: 'lightgreen'},
        opacity=0.7,
        name="Binarised",
        blending='translucent',
    )


# In the previous example, we were able to see the progress bar, but were not
# able to control it. By using `progress` within a context manager, we can
# manipulate the `progress` object and still get the benefit of automatic
# clean up
def segment_binarised_ims():
    """Segments each of the binarised ims.

    Uses `progress` within a context manager allowing us to manipulate
    the progress bar within the loop
    """
    segmented_nuclei = []
    binarised_data = viewer.layers['Binarised'].data

    # using the `with` keyword we can use `progress` inside a context manager
    # `progress` inherits from tqdm and therefore provides the same API
    # e.g. we can provide the miniters argument if we want to see the
    # progress bar update with each iteration
    with progress(binarised_data, miniters=0) as pbar:
        for i, binarised_cells in enumerate(pbar):
            # this allows us to manipulate the pbar object within the loop
            # e.g. setting the description. 
            pbar.set_description(all_thresholds[i].__name__.split("_")[1])
            labelled_im = label(binarised_cells)
            segmented_nuclei.append(labelled_im)

            # uncomment if processing is too fast
            sleep(0.5)

    # progress bar is still automatically closed

    segmented_nuclei = np.stack(segmented_nuclei)
    viewer.add_labels(
        segmented_nuclei,
        name="Segmented",
        blending='translucent',
    )
    viewer.layers['Binarised'].visible = False


# we can also manually control `progress` objects using their
# `update` method (inherited from tqdm)
def process_ims():
    """
    First performs thresholding, then segmentation on our image.

    Manually updates a `progress` object.
    """
    # we instantiate a manually controlled `progress` object
    # by just passing a total with no iterable
    pbar = progress(total=2)
    pbar.set_description("Thresholding")
    try_thresholds()
    # once one processing step is complete, we increment
    # the value of our progress bar
    pbar.update(1)

    pbar.set_description("Segmenting")
    segment_binarised_ims()
    pbar.update(1)

    # uncomment this line to see the 100% progress bar
    sleep(0.5)

    # if manually updating the progress bar, we must also
    # manually close it
    pbar.close()


button_layout = QVBoxLayout()
process_btn = QPushButton("Full Process")
process_btn.clicked.connect(process_ims)
button_layout.addWidget(process_btn)

thresh_btn = QPushButton("1.Threshold")
thresh_btn.clicked.connect(try_thresholds)
button_layout.addWidget(thresh_btn)

segment_btn = QPushButton("2.Segment")
segment_btn.clicked.connect(segment_binarised_ims)
button_layout.addWidget(segment_btn)

action_widget = QWidget()
action_widget.setLayout(button_layout)
viewer.window.add_dock_widget(action_widget)

napari.run()
