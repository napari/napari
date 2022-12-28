"""
Interaction box points
======================

Demonstrate interaction box on points layer

.. tags:: experimental
"""

import napari
import numpy as np
from napari.layers.points._points_utils import points_in_box


def on_selection_box_drag(event):
    # Do selection in world coordinates so box aligns with axes (not sure if this is guaranteed)
    points = viewer.layers.selection.active._data_to_world(viewer.layers.selection.active._view_data)
    sel_i = points_in_box(event.value,points,viewer.layers.selection.active._view_size)
    viewer.layers.selection.active.selected_data = sel_i

def on_selection_box_final(event):
    sel_i = viewer.layers.selection.active.selected_data
    if len(sel_i) > 0:
        viewer.overlays.interaction_box.points = viewer.layers.selection.active._data_to_world(np.array([viewer.layers.selection.active._view_data[i] for i in sel_i]))
        viewer.overlays.interaction_box.show = True
        viewer.overlays.interaction_box.show_vertices = True
        viewer.overlays.interaction_box.show_handle = True
    else:
        viewer.overlays.interaction_box.points = None
        viewer.layers.selection.active.selected_data = []

def on_transform_changed_drag(event):
    sel_i = viewer.layers.selection.active.selected_data
    points = viewer.overlays.interaction_box.points

    for i, index in enumerate(sel_i):
        viewer.layers.selection.active._data[index] = viewer.layers.selection.active.world_to_data(event.value(points[i]))
    viewer.layers.selection.active._clear_extent()
    viewer.layers.selection.active.events.data(value=viewer.layers.selection.active.data)

X, Y = np.mgrid[-500:500:50, -500:500:50]
positions = np.dstack([X.ravel(), Y.ravel()])
viewer = napari.view_points(positions[0,:,:])
viewer.layers.selection.active.interactive = False
viewer.overlays.interaction_box.show = True
viewer.overlays.interaction_box.events.selection_box_drag.connect(on_selection_box_drag)
viewer.overlays.interaction_box.events.selection_box_final.connect(on_selection_box_final)
viewer.overlays.interaction_box.events.transform_drag.connect(on_transform_changed_drag)

if __name__ == '__main__':
    napari.run()
