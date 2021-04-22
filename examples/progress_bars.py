"""
Use napari's tqdm wrapper to display the progress of long-running operations
in the viewer.  
"""

from time import sleep
import numpy as np
from skimage.morphology import closing, square
import napari
from napari.utils.progress import progress
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

from skimage.filters import *
from skimage.data import cells3d
from skimage.measure import label

# we will try each of these thresholds on our image
all_thresholds = [
    threshold_isodata,
    threshold_li,
    threshold_niblack,
    threshold_otsu,
    threshold_sauvola,
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
    """Tries each threshold for both nuclei and membranes, and adds result to viewer.
    """
    thresholded_nuclei = []

    # we wrap our iterable with `progress`
    # this will automatically add a progress bar to our activity dock
    for threshold_func in progress(all_thresholds):
        current_threshold = threshold_func(cell_nuclei)
        binarised_im = cell_nuclei > current_threshold
        thresholded_nuclei.append(binarised_im)

    # working with a wrapped interval, the progress bar will be closed
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
    with progress(binarised_data) as pbar:
        for i, binarised_cells in enumerate(pbar):
            # this allows us to manipulate the pbar object within the loop
            # e.g. setting the description. `progress` inherits from tqdm
            # and therefore provides the same API
            pbar.set_description(all_thresholds[i].__name__.split("_")[1])
            labelled_im = label(binarised_cells)
            segmented_nuclei.append(labelled_im)

            # uncomment if processing is too fast
            # sleep(0.5)

    # progress bar is still automatically closed

    segmented_nuclei = np.stack(segmented_nuclei)
    viewer.add_labels(
        segmented_nuclei,
        name="Segmented",
        blending='translucent',
    )
    viewer.layers['Binarised'].visible = False

# TODO: manual updates

button_layout = QVBoxLayout()
thresh_btn = QPushButton("Try Thresholds")
thresh_btn.clicked.connect(try_thresholds)
button_layout.addWidget(thresh_btn)

thresh_w_desc_btn = QPushButton("Segment - Context Manager")
thresh_w_desc_btn.clicked.connect(segment_binarised_ims)
button_layout.addWidget(thresh_w_desc_btn)

action_widget = QWidget()
action_widget.setLayout(button_layout)

viewer.window.add_dock_widget(action_widget)

napari.run()
