from qtpy.QtWidgets import QApplication
from ._qt.qt_viewer import QtViewer
from ._qt.qt_main_window import Window
from .components import ViewerModel


class Viewer(ViewerModel):
    """Napari ndarray viewer.

    Parameters
    ----------
    title : string
        The title of the viewer window.
    ndisplay : {2, 3}
        Number of displayed dimensions.
    order : tuple of int
        Order in which dimensions are displayed where the last two or last
        three dimensions correspond to row x column or plane x row x column if
        ndisplay is 2 or 3.
    """

    def __init__(self, title='napari', ndisplay=2, order=None):
        # instance() returns the singleton instance if it exists, or None
        app = QApplication.instance()
        # if None, we create our own and we execute after creating our object
        if app is None:
            app = QApplication([])
            self_execute = True
        # otherwise, we don't execute and we delete our reference to app
        else:
            self_execute = False
            # appears to be required to not conflict with IPython Qt loop
            del app
        super().__init__(title=title, ndisplay=ndisplay, order=order)
        qt_viewer = QtViewer(self)
        self.window = Window(qt_viewer)
        self.screenshot = self.window.qt_viewer.screenshot
        self.update_console = self.window.qt_viewer.console.push
        if self_execute:
            app.exec_()
