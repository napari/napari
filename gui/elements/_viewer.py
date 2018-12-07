from .qt import QtViewer


class Viewer:
    def __init__(self):
        from ._layer_list import LayerList
        from ._controls import Controls
        from ._center import Center

        # self.layers = LayerList(self)
        self.center = Center(self)
        self.controls = Controls()
        self._qt = QtViewer(self)