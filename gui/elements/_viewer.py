from .qt import QtViewer


class Viewer:
    """Viewer containing the rendered scene, layers, and controlling elements
    including dimension sliders.

    Parameters
    ----------
    parent : Window
        Parent window.

    Attributes
    ----------
    layers : LayerList
        List of contained layers.
    window : Window
        Parent window
    """
    def __init__(self):
        from ._layer_list import LayerList
        from ._controls import Controls
        from ._center import Center

        # self.layers = LayerList(self)
        self.center = Center(self)
        self.controls = Controls()
        self._qt = QtViewer(self)
