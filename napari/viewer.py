from itertools import zip_longest
from ._qt.qt_viewer import QtViewer
from ._qt.qt_main_window import Window
from .components import ViewerModel


class Viewer(ViewerModel):
    """Napari ndarray viewer.

    Parameters
    ----------
    title : string
        The title of the viewer window.
    """

    def __init__(self, title='napari'):
        super().__init__(title=title)
        qt_viewer = QtViewer(self)
        self.window = Window(qt_viewer)
        self.screenshot = self.window.qt_viewer.screenshot
        self.camera = self.window.qt_viewer.view.camera


@Viewer.bind_key('v')
def toggle_last_visible(viewer):
    if len(viewer.layers) > 0:
        layer = viewer.layers[-1]
        layer.visible = not layer.visible


@Viewer.bind_key('[')
def make_next_invisible_layer_visible(viewer):
    if len(viewer.layers) == 1:
        viewer.layers[0].visible = True
    else:
        last_layers = viewer.layers[-1:0:-1]
        layers = viewer.layers[-2::-1]
        for last_layer, layer in zip_longest(last_layers, layers):
            if not last_layer.visible and (layer is None or layer.visible):
                last_layer.visible = True
                break


@Viewer.bind_key(']')
def make_next_visible_layer_invisible(viewer):
    for layer in viewer.layers[::-1]:
        if layer.visible:
            layer.visible = False
            break
