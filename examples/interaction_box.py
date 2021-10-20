"""
Demonstrate interaction box on points layer
"""

from skimage import data
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
    viewer.interaction_box.points = np.array([viewer.layers.selection.active._view_data[i] for i in sel_i])
    viewer.interaction_box.show = True



# def on_transform_changed_drag(event):
#     sel_i = viewer.layers.selection.active.selected_data
#     points = viewer.layers.selection.active._preselect_points

#     for i, index in enumerate(sel_i):
#         viewer.layers.selection.active._data[index] = event.transform(points[i])
#     viewer.layers.selection.active._update_dims()
#     viewer.layers.selection.active.events.data(value=viewer.layers.selection.active.data)

# def on_transform_changed_final(event):
#     sel_i = viewer.layers.selection.active.selected_data
    
#     viewer.layers.selection.active._preselect_points = [viewer.layers.selection.active._view_data[i] for i in sel_i]


    
 

X, Y = np.mgrid[-500:500:50, -500:500:50]
positions = np.dstack([X.ravel(), Y.ravel()])
viewer = napari.view_points(positions[0,:,:])
viewer.layers.selection.active.interactive = False
viewer.interaction_box.show = True
viewer.interaction_box.events.selection_box_drag.connect(on_selection_box_drag)
viewer.interaction_box.events.selection_box_final.connect(on_selection_box_final)
#viewer.interaction_box.events.transform_changed_drag.connect(on_transform_changed_drag)
#viewer.interaction_box.events.transform_changed_final.connect(on_transform_changed_final)

napari.run()
