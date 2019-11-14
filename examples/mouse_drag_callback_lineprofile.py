"""
Example updating the status bar with line profile info while dragging
lines around in a shapes layer.
"""

from skimage import data
from skimage import measure
import numpy as np
import napari
from napari._qt.qt_lineprofile import QtLineProfile

with napari.gui_qt():
    np.random.seed(1)
    viewer = napari.Viewer()
    chelsea = data.chelsea().mean(-1)
    viewer.add_image(chelsea)
    shapes_layer = viewer.add_shapes(
        [np.array([[11, 13], [250, 313]])],
        shape_type='line',
        edge_width=5,
        edge_color='coral',
        face_color='royalblue',
    )
    shapes_layer.mode = 'select'
    w = QtLineProfile(viewer=viewer)
    w.show()

    def profile_lines(image, shape_layer):
        # only a single line for this example
        for line in shape_layer.data:
            w.set_data(measure.profile_line(image, line[0], line[1]))

    profile_lines(chelsea, shapes_layer)

    @shapes_layer.mouse_drag_callbacks.append
    def profile_lines_drag(layer, event):
        profile_lines(chelsea, layer)
        yield
        while event.type == 'mouse_move':
            profile_lines(chelsea, layer)
            yield
