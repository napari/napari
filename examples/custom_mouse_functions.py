"""
Custom mouse functions
======================

Display one 4-D image layer using the ``add_image`` API

.. tags:: gui
"""

from skimage import data
from skimage.morphology import binary_dilation, binary_erosion
from scipy import ndimage as ndi
import numpy as np
import napari


np.random.seed(1)
viewer = napari.Viewer()
blobs = data.binary_blobs(length=128, volume_fraction=0.1, n_dim=2)
labeled = ndi.label(blobs)[0]
labels_layer = viewer.add_labels(labeled, name='blob ID')

@viewer.mouse_drag_callbacks.append
def get_event(viewer, event):
    print(event)

@viewer.mouse_drag_callbacks.append
def get_ndisplay(viewer, event):
    if 'Alt' in event.modifiers:
        print('viewer display ', viewer.dims.ndisplay)

@labels_layer.mouse_drag_callbacks.append
def get_connected_component_shape(layer, event):
    data_coordinates = layer.world_to_data(event.position)
    cords = np.round(data_coordinates).astype(int)
    val = layer.get_value(data_coordinates)
    if val is None:
        return
    if val != 0:
        data = layer.data
        binary = data == val
        if 'Shift' in event.modifiers:
            binary_new = binary_erosion(binary)
            data[binary] = 0
        else:
            binary_new = binary_dilation(binary)
        data[binary_new] = val
        size = np.sum(binary_new)
        layer.data = data
        msg = (
            f'clicked at {cords} on blob {val} which is now {size} pixels'
        )
    else:
        msg = f'clicked at {cords} on background which is ignored'
    print(msg)

# Handle click or drag events separately
@labels_layer.mouse_drag_callbacks.append
def click_drag(layer, event):
    print('mouse down')
    dragged = False
    yield
    # on move
    while event.type == 'mouse_move':
        print(event.position)
        dragged = True
        yield
    # on release
    if dragged:
        print('drag end')
    else:
        print('clicked!')

# Handle click or drag events separately
@labels_layer.mouse_double_click_callbacks.append
def on_second_click_of_double_click(layer, event):
    print('Second click of double_click', event.position)
    print('note that a click event was also triggered', event.type)


if __name__ == '__main__':
    napari.run()
