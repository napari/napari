"""
Comparison of Screenshot and Figure Export
==========================================

Display multiple layer types, add scale bar, and take a screenshot or export a
figure from a 'light' canvas. Then switch to a 'dark' canvas and display the
screenshot and figure. Compare the limits of each export method. The screenshot
will include the entire canvas, and results in some layers (e.g. the 'pos'
layer) if it extends outside the canvas. In comparison, the `export_figure`
will only include the extent of the layers and any other elements overlayed
on the canvas, such as the scale bar. Exported figures also move the scale bar
to within the margins of the canvas.

Currently, 'export_figure` does not support the 3D view, but screenshot does.

.. tags:: visualization-advanced
"""

import numpy as np
from skimage import data
from vispy.color import Colormap

import napari

# create the viewer and window
viewer = napari.Viewer()

# add the image
img_layer = viewer.add_image(data.camera(), name='photographer')
img_layer.colormap = 'gray'

# create a list of polygons
polygons = [
    np.array([[11, 13], [111, 113], [22, 246]]),
    np.array(
        [
            [505, 60],
            [402, 71],
            [383, 42],
            [251, 95],
            [212, 59],
            [131, 137],
            [126, 187],
            [191, 204],
            [171, 248],
            [211, 260],
            [273, 243],
            [264, 225],
            [430, 173],
            [512, 160],
        ]
    ),
    np.array(
        [
            [310, 382],
            [229, 381],
            [209, 401],
            [221, 411],
            [258, 411],
            [300, 412],
            [306, 435],
            [268, 434],
            [265, 454],
            [298, 461],
            [307, 461],
            [307, 507],
            [349, 510],
            [352, 369],
            [330, 366],
            [330, 366],
        ]
    ),
]

# add polygons
layer = viewer.add_shapes(
    polygons,
    shape_type='polygon',
    edge_width=1,
    edge_color='coral',
    face_color='royalblue',
    name='shapes',
)

# change some attributes of the layer
layer.selected_data = set(range(layer.nshapes))
layer.current_edge_width = 5
layer.opacity = 0.75
layer.selected_data = set()

# add an ellipse to the layer
ellipse = np.array([[59, 222], [110, 289], [170, 243], [119, 176]])
layer.add(
    ellipse,
    shape_type='ellipse',
    edge_width=5,
    edge_color='coral',
    face_color='purple',
)

masks = layer.to_masks([512, 512])
masks_layer = viewer.add_image(masks.astype(float), name='masks')
masks_layer.opacity = 0.7
masks_layer.colormap = Colormap([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]])

labels = layer.to_labels([512, 512])
labels_layer = viewer.add_labels(labels, name='labels')

points = np.array([[100, 100], [200, 200], [333, 111]])
size = np.array([10, 20, 20])
viewer.add_points(points, size=size)

# sample vector coord-like data
n = 100
pos = np.zeros((n, 2, 2), dtype=np.float32)
phi_space = np.linspace(0, 4 * np.pi, n)
radius_space = np.linspace(0, 100, n)

# assign x-y position
pos[:, 0, 0] = radius_space * np.cos(phi_space) + 350
pos[:, 0, 1] = radius_space * np.sin(phi_space) + 256

# assign x-y projection
pos[:, 1, 0] = 2 * radius_space * np.cos(phi_space)
pos[:, 1, 1] = 2 * radius_space * np.sin(phi_space)

# add the vectors, note that they extend beyond image extent
pos_layer = viewer.add_vectors(pos, edge_width=2)

# add scale_bar with background box
viewer.scale_bar.visible = True
viewer.scale_bar.box = True
# viewer.scale_bar.length = 150  # prevent dynamic adjustment of scale bar length

# take screenshots in light theme to show difference in canvas margins
viewer.theme = 'light'
screenshot = viewer.screenshot()
figure = viewer.export_figure()
# optionally, save the exported figure: viewer.export_figure(path='export_figure.png')
# or screenshot: viewer.screenshot(path='screenshot.png')

# show different handling when canvas is zoomed in
viewer.camera.zoom = 2
screenshot_zoomed = viewer.screenshot()
figure_zoomed = viewer.export_figure()

# export a figure without the 'pos' layer
# this is the vectors layer that extends beyond the image extent
viewer.layers.remove(pos_layer)
figure_no_pos = viewer.export_figure()

# switch back to dark theme and remove original layers
viewer.theme = 'dark'
viewer.layers.select_all()
viewer.layers.remove_selected()

# display the screenshot and figure and switch to gride mode for comparison
viewer.add_image(screenshot_zoomed, rgb=True, name='screenshot_zoomed')
viewer.add_image(screenshot, rgb=True, name='screenshot')
viewer.add_image(figure_no_pos, rgb=True, name='figure_no_pos')
viewer.add_image(figure_zoomed, rgb=True, name='figure_zoomed')
viewer.add_image(figure, rgb=True, name='figure')

viewer.grid.enabled = True
viewer.grid.shape = (2, 3)

if __name__ == '__main__':
    napari.run()
