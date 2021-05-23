"""Copy canvas or whole viewer to clipboard."""
from skimage import data

import napari
from napari._qt.widgets.qt_viewer_buttons import QtViewerPushButton


# create the viewer with an image
viewer = napari.view_image(data.astronaut(), rgb=True)

layer_buttons = viewer.window.qt_viewer.layerButtons

# Add button to copy image of the canvas to clipboard.
copy_canvas_button = QtViewerPushButton(None, 'warning')
copy_canvas_button.setToolTip("Copy screenshot of the canvas to clipboard.")
copy_canvas_button.clicked.connect(viewer.window.qt_viewer.clipboard)
layer_buttons.layout().insertWidget(3, copy_canvas_button)

copy_canvas_button = QtViewerPushButton(None, 'warning')
copy_canvas_button.setToolTip("Copy screenshot of the entire viewer to clipboard.")
copy_canvas_button.clicked.connect(viewer.window.clipboard)
layer_buttons.layout().insertWidget(3, copy_canvas_button)

napari.run()
