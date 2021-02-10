"""
Display one 4-D image layer using the add_image API
"""

from skimage import data
from skimage.morphology import binary_dilation, binary_erosion
from scipy import ndimage as ndi
import numpy as np
import napari


with napari.gui_qt():
    np.random.seed(1)
    viewer = napari.Viewer()
    blobs = data.binary_blobs(length=128, volume_fraction=0.1, n_dim=2)
    labeled = ndi.label(blobs)[0]
    labels_layer = viewer.add_labels(labeled, name='blob ID')

    @viewer.mouse_drag_callbacks.append
    def get_event(viewer, cursor_event):
        print(cursor_event)

    @viewer.mouse_drag_callbacks.append
    def get_ndisplay(viewer, cursor_event):
        if 'Alt' in cursor_event.modifiers:
            print('viewer display ', viewer.dims.ndisplay)

    @labels_layer.mouse_drag_callbacks.append
    def get_connected_component_shape(layer, cursor_event):
        coordinates = cursor_event.data_position
        cords = np.round(coordinates).astype(int)
        val = layer.get_value(coordinates)
        if val is None:
            return
        if val != 0:
            data = layer.data
            binary = data == val
            if 'Shift' in cursor_event.modifiers:
                binary_new = binary_erosion(binary)
                data[binary] = 0
                data[binary_new] = val
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
    def click_drag(layer, cursor_event):
        print('mouse down')
        dragged = False
        yield
        # on move
        while cursor_event.type == 'mouse_move':
            print(cursor_event.position)
            dragged = True
            yield
        # on release
        if dragged:
            print('drag end')
        else:
            print('clicked!')
