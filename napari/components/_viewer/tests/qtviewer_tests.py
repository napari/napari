# This is in interactive test
from PyQt5.QtWidgets import QSplitter, QRadioButton, QWidget, QGridLayout, QVBoxLayout
from skimage import data
from skimage.color import rgb2gray

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

    astronaut = rgb2gray(data.astronaut())
    coins = rgb2gray(data.coins())

    print(astronaut.shape)
    print(coins.shape)

    viewer.add_image(astronaut)
    viewer.add_image(coins)

    print(viewer.dims.num_dimensions)



    #widget = QtDims(Dims(10))

    # makes the view visible on the desktop:
    #widget.setGeometry(200, 200, 200, 20)
    widget.show()
