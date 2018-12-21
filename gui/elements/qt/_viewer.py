from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtWidgets import QWidget, QSlider, QVBoxLayout, QSplitter
from PyQt5.QtGui import QCursor, QPixmap
from vispy.scene import SceneCanvas, PanZoomCamera

from os.path import join
from ...icons import icons_dir
path_cursor = join(icons_dir, 'cursor_disabled.png')

class QtViewer(QSplitter):

    statusChanged = pyqtSignal(str)
    helpChanged = pyqtSignal(str)

    def __init__(self, viewer):
        super().__init__()

        self.canvas = SceneCanvas(keys=None, vsync=True)
        self.canvas.native.setMinimumSize(QSize(100, 100))

        self.view = self.canvas.central_widget.add_view()
        # Set 2D camera (the camera will scale to the contents in the scene)
        self.view.camera = PanZoomCamera(aspect=1)
        # flip y-axis to have correct aligment
        self.view.camera.flip = (0, 1, 0)
        self.view.camera.set_range()

        center = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.canvas.native)
        layout.addWidget(viewer.dimensions._qt)
        center.setLayout(layout)

        # Add vertical sliders, center, and layerlist
        self.addWidget(viewer.controlBars._qt)
        self.addWidget(center)
        self.addWidget(viewer.layers._qt)

        viewer.dimensions._qt.setFixedHeight(0)

        self._cursors = {
            'diabled' : QCursor(QPixmap(path_cursor).scaled(20,20)),
            'cross' : Qt.CrossCursor,
            'forbidden' : Qt.ForbiddenCursor,
            'pointing' : Qt.PointingHandCursor,
            'standard' : QCursor()
        }
