from ._qt.qt_viewer import QtViewer
from ._qt.qt_main_window import Window
from .components import ViewerModel


class Viewer(ViewerModel):
    """Napari ndarray viewer.

    Parameters
    ----------
    title : string
        The title of the viewer window.
    ndisplay : int
        Number of displayed dimensions.
    """

    def __init__(self, title='napari', ndisplay=2):
        super().__init__(title=title, ndisplay=ndisplay)
        qt_viewer = QtViewer(self)
        self.window = Window(qt_viewer)
        self.screenshot = self.window.qt_viewer.screenshot
        self.camera = self.window.qt_viewer.view.camera
        self.update_console = self.window.qt_viewer.console.push
