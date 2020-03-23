"""
Demonstrate interaction box on image layer
"""

from skimage import data
import napari
import numpy as np

def on_selection_box_drag(event):
    sel_i = napari.layers.points.points.points_in_box(event.box,viewer.active_layer._data_view,viewer.active_layer._size_view)
    

with napari.gui_qt():
    # create the viewer with an image
    X, Y = np.mgrid[-500:500:50, -500:500:50]
    positions = np.dstack([X.ravel(), Y.ravel()])
    viewer = napari.view_points(positions[0,:,:])
    viewer.active_layer.interactive = False
    
    viewer.active_layer._interaction_box.events.selection_box_changed_drag.connect(on_selection_box_drag)
