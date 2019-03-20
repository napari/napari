# This is in interactive test
import numpy
from PyQt5.QtWidgets import QRadioButton, QWidget, QVBoxLayout

from napari import Viewer
from napari.components import Dims
from napari.components._dims.qtdims import QtDims
from napari.components._viewer.qtviewer import QtViewer
from napari.util import app_context

class QtViewerDummy(QWidget):

    def __init__(self, viewer, parent = None):
        super().__init__(parent=parent)

        self.viewer = viewer

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(layout)

        self.dimsview = QtDims(self.viewer.dims, parent=self)
        self.dimsview.setMinimumHeight(100)
        self.layout().addWidget(self.dimsview)
        self.layout().addWidget(QRadioButton())


with app_context():
    viewer  = Viewer()

    # creates a widget to view (and control) the model:
    widget = QtViewer(viewer)

    dataset = numpy.random.rand(10, 10, 512, 512)
    viewer.add_image(dataset)

    widget.show()
