import platform

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QAction, qApp, QMenuBar

from .image_widget import ImageWidget


class ImageWindow(QMainWindow):
    """Image-based PyQt5 window.

    Parameters
    ----------
    image : NImage
        Image to display.
    window_width : int, optional
        Width of the window.
    window_height : int, optional
        Height of the window.
    parent : PyQt5.QWidget, optional
        Parent widget.
    """
    def __init__(self, image, window_width=800, window_height=800, parent=None):
        super().__init__(parent)

        self.widget = ImageWidget(image, window_width, window_height, containing_window=self)

        self.setCentralWidget(self.widget)

        self.statusBar().showMessage('Ready')

        self.add_menu()
        self.add_toolbar()
        self.show()
        self.raise_()

        self.installEventFilter(self)

    def add_toolbar(self):
        """Adds a toolbar.
        """
        self.denoise_toolbar = self.addToolBar('Denoise')
        gaussianAction = QAction('Gaussian', self)
        gaussianAction.setShortcut('Ctrl+G')
        nlmAction = QAction('NLM', self)
        nlmAction.setShortcut('Ctrl+N')
        bilateralAction = QAction('Bilateral', self)
        bilateralAction.setShortcut('Ctrl+B')

        self.denoise_toolbar.addAction(gaussianAction)
        self.denoise_toolbar.addAction(nlmAction)
        self.denoise_toolbar.addAction(bilateralAction)

        self.toolbar = self.addToolBar('FFTs')

    def add_menu(self):
        """Adds a menu bar.
        """
        menubar = self.menuBar() # parentless menu bar for Mac OS
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('&File')
        editMenu = menubar.addMenu('&Edit')
        editMenu = menubar.addMenu('&Image')
        viewMenu = menubar.addMenu('&Process')
        searchMenu = menubar.addMenu('&Analyse')
        toolsMenu = menubar.addMenu('&Plugins')
        toolsMenu = menubar.addMenu('&Windows')
        helpMenu = menubar.addMenu('&About')


    # def eventFilter(self, object, event):
    #     #print(event)
    #     #print(event.type())
    #     if event.type() == QEvent.ShortcutOverride:
    #         #print(event)
    #         #print(event.key())
    #         if (event.key() == Qt.Key_F or event.key() == Qt.Key_Enter) and not self.isFullScreen():
    #             print("showFullScreen!")
    #             self.showFullScreen()
    #         elif (event.key() == Qt.Key_F or event.key() == Qt.Key_Escape) and self.isFullScreen():
    #             print("showNormal!")
    #             self.showNormal()
    #
    #     return QWidget.eventFilter(self, object, event)


    def update_image(self):
        """Updates the contained image.
        """
        self.widget.update_image()

    @property
    def cmap(self):
        """string: Color map.
        """
        return self.widget.cmap

    @cmap.setter
    def cmap(self, cmap):
        self.widget.cmap = cmap
