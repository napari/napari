"""
Comparison of Screenshot and Figure Export
==========================================

Display multiple layer types, add scale bar, and take a screenshot or export a
figure from a 'light' canvas. Then switch to a 'dark' canvas and display the
screenshot and figure. Compare the limits of each export method. The screenshot
will include the entire canvas, and results in some layers being clipped
if it extends outside the canvas. This also means that screenshots will
reflect the current zoom. In comparison, the `export_figure` will always
include the extent of the layers and any other elements overlayed
on the canvas, such as the scale bar. Exported figures also move the scale bar
to within the margins of the canvas.

In the final grid state shown below, the first row represents exported images. The first two show that zoom is not reflected in the exported figure. The final one shows how the exported figure adapts to change in the layer extent. In the second row are the screenshots, showing the fact that the entire canvas is captured and that zoom is preserved.

.. tags:: visualization-advanced
"""

import numpy as np
from skimage import data

import napari

# Create a napari viewer with multiple layer types and add a scale bar.
# One of the polygon shapes exists outside the image extent, which is
# useful in displaying how figure export handles the extent of all layers.

viewer = napari.Viewer()

# add a 2D image layer
img_layer = viewer.add_image(data.camera(), name='photographer')
img_layer.colormap = 'gray'

# polygon within image extent
layer_within = viewer.add_shapes(
    np.array([[11, 13], [111, 113], [22, 246]]),
    shape_type='polygon',
    face_color='coral',
    name='shapes_within',
)

# add a polygon shape layer
layer_outside = viewer.add_shapes(
    np.array([[572, 222], [305, 292], [577, 440]]),
    shape_type='polygon',
    face_color='royalblue',
    name='shapes_outside',
)

# add scale_bar with background box
viewer.scale_bar.visible = True
viewer.scale_bar.box = True
# viewer.scale_bar.length = 150  # prevent dynamic adjustment of scale bar length


# Take screenshots and export figures in 'light' theme, to show the canvas
# margins and the extent of the exported figure.
viewer.theme = 'light'
screenshot = viewer.screenshot()
figure = viewer.export_figure()
# optionally, save the exported figure: viewer.export_figure(path='export_figure.png')
# or screenshot: viewer.screenshot(path='screenshot.png')


# Zoom in and take another screenshot and export figure to show the different
# extents of the exported figure and screenshot.
viewer.camera.zoom = 3
screenshot_zoomed = viewer.screenshot()
figure_zoomed = viewer.export_figure()


# Remove the layer that exists outside the image extent and take another
# figure export to show the extent of the exported figure without the
# layer that exists outside the camera image extent.
viewer.layers.remove(layer_outside)
figure_no_outside_shape = viewer.export_figure()


# Display the screenshots and figures in 'dark' theme, and switch to grid mode
# for comparison. In the final grid state shown, the first row represents exported
# images. The first two show that zoom is not reflected in the exported figure.
# The final one shows how the exported figure adapts to change in the layer extent.
# In the second row are the screenshots, showing the fact that the entire canvas
# is captured and that zoom is preserved.
viewer.theme = 'dark'
viewer.layers.select_all()
viewer.layers.remove_selected()

viewer.add_image(screenshot_zoomed, rgb=True, name='screenshot_zoomed')
viewer.add_image(screenshot, rgb=True, name='screenshot')
viewer.add_image(figure_no_outside_shape, rgb=True, name='figure_no_outside_shape')
viewer.add_image(figure_zoomed, rgb=True, name='figure_zoomed')
viewer.add_image(figure, rgb=True, name='figure')

viewer.grid.enabled = True
viewer.grid.shape = (2, 3)

if __name__ == '__main__':
    napari.run()
