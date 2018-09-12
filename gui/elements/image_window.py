import platform

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout,
                             QAction, qApp, QMenuBar)

from .image_widget import ImageWidget


class ImageWindow(QMainWindow):
    """Image-based PyQt5 window.

    Parameters
    ----------
    parent : PyQt5.QWidget, optional
        Parent widget.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.widget = QWidget()
        self.setCentralWidget(self.widget)

        layout = QHBoxLayout()
        self.widget.setLayout(layout)

        self.statusBar().showMessage('Ready')

        self.add_menu()
        self.add_toolbar()

        self.installEventFilter(self)

    def add_image(self, image, meta):
        """Adds an image to the containing layout.

        Parameters
        ----------
        image : np.ndarray
            Image to display.
        meta : dict
            Image metadata.

        Returns
        -------
        widget : ImageWidget
            Widget containing the image.
        """
        widget = ImageWidget(image, meta)
        self.widget.layout().addWidget(widget)
        return widget

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
