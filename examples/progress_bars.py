"""
Use napari's tqdm wrapper to display the progress of long-running operations
in the viewer.  
"""

import numpy as np
from skimage.filters.thresholding import (
    threshold_isodata,
    threshold_li,
    threshold_local,
)
import napari
from napari.utils.progress import progress
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

from skimage.filters import *
from skimage.data import cells3d

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

# load cells data and split channels out
cells_im = cells3d()
cell_membranes = cells_im[:, 0, :, :]
cell_nuclei = cells_im[:, 1, :, :]
viewer.add_image(
    cell_membranes, name="Membranes", blending='additive', colormap="magenta"
)
viewer.add_image(
    cell_nuclei, name="Nuclei", blending='additive', colormap="green"
)

def try_thresholds():
    """Tries each threshold for both nuclei and membranes, and adds result to viewer.
    """
    nuclei_im = viewer.layers['Nuclei'].data
    membranes_im = viewer.layers['Membranes'].data

    thresholded_nuclei = []
    thresholded_membranes = []

    # we wrap our iterable with `progress`
    # this will automatically add a progress bar to our activity dock
    for threshold_func in progress(all_thresholds):
        current_threshold = threshold_func(nuclei_im)
        binarised_im = nuclei_im > current_threshold
        thresholded_nuclei.append(binarised_im)

        current_threshold = threshold_func(membranes_im)
        binarised_im = membranes_im > current_threshold
        thresholded_membranes.append(binarised_im)

    # working with a wrapped interval, the progress bar will be closed
    # as soon as the iteration is complete

    binarised_nuclei = np.stack(thresholded_nuclei)
    binarised_membranes = np.stack(thresholded_membranes)
    viewer.add_labels(
        binarised_nuclei,
        color={1: 'lightgreen'},
        opacity=0.7,
        name="Binary Nuclei",
        blending='translucent',
    )
    viewer.add_labels(
        binarised_membranes,
        color={1: 'violet'},
        opacity=0.7,
        name="Binary Membranes",
        blending='translucent',
    )

# In the previous example, we were able to see the progress bar, but were not
# able to control it. By using `progress` within a context manager, we can 
# manipulate the `progress` object and still get the benefit of automatic
# clean up
def try_thresholds_context_manager():
    """Tries each threshold for both nuclei and membranes, and adds result to viewer.

    Sets description of progress bar to current thresholding function being applied
    """
    nuclei_im = viewer.layers['Nuclei'].data
    membranes_im = viewer.layers['Membranes'].data

    thresholded_nuclei = []
    thresholded_membranes = []

    # using the `with` keyword we can use `progress` inside a context manager
    with progress(all_thresholds) as pbar:
        for threshold_func in pbar:
            pbar.set_description(threshold_func.__name__.split("_")[1])
            current_threshold = threshold_func(nuclei_im)
            binarised_im = nuclei_im > current_threshold
            thresholded_nuclei.append(binarised_im)

            current_threshold = threshold_func(membranes_im)
            binarised_im = membranes_im > current_threshold
            thresholded_membranes.append(binarised_im)

    # progress bar is still automatically closed

    binarised_nuclei = np.stack(thresholded_nuclei)
    binarised_membranes = np.stack(thresholded_membranes)
    viewer.add_labels(
        binarised_nuclei,
        color={1: 'lightgreen'},
        opacity=0.7,
        name="Binary Nuclei",
        blending='translucent',
    )
    viewer.add_labels(
        binarised_membranes,
        color={1: 'violet'},
        opacity=0.7,
        name="Binary Membranes",
        blending='translucent',
    )

button_layout = QVBoxLayout()
thresh_btn = QPushButton("Try Thresholds")
thresh_btn.clicked.connect(try_thresholds)
button_layout.addWidget(thresh_btn)

thresh_w_desc_btn = QPushButton("Try Thresholds - Context Manager")
thresh_w_desc_btn.clicked.connect(try_thresholds_context_manager)
button_layout.addWidget(thresh_w_desc_btn)

action_widget = QWidget()
action_widget.setLayout(button_layout)

viewer.window.add_dock_widget(action_widget)

napari.run()
