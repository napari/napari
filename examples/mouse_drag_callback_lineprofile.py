"""
Example updating the status bar with line profile info while dragging
lines around in a shapes layer.
"""

from skimage import data
from skimage import measure
import numpy as np
import napari
from napari._vispy import Fig


def get_line_data(image, start, end):
    return measure.profile_line(image, start, end)


with napari.gui_qt():
    np.random.seed(1)
    viewer = napari.Viewer()
    chelsea = data.chelsea().mean(-1)
    viewer.add_image(chelsea)
    shapes_layer = viewer.add_shapes(
        [np.array([[11, 13], [250, 313]]), np.array([[100, 10], [10, 345]])],
        shape_type='line',
        edge_width=5,
        edge_color='coral',
        face_color='royalblue',
    )
    shapes_layer.mode = 'select'

    # add the figure as a dock_widget
    fig, dw = viewer.window.add_docked_figure(area='right', initial_width=250)
    # add a new subplot to the figure for each line in the shapes layer
    lines = [
        fig[i, 0].plot(get_line_data(chelsea, *line), marker_size=0)
        for i, line in enumerate(shapes_layer.data)
    ]

    # hook the lines up to events
    def profile_lines(image, shape_layer):
        # only a single line for this example
        for i, line in enumerate(shape_layer.data):
            if i in shape_layer._selected_data:
                lines[i].set_data(get_line_data(image, *line))
                fig[i, 0].autoscale()

    @shapes_layer.mouse_drag_callbacks.append
    def profile_lines_drag(layer, event):
        profile_lines(chelsea, layer)
        yield
        while event.type == 'mouse_move':
            profile_lines(chelsea, layer)
            yield
