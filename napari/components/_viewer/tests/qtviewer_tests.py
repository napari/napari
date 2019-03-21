# This is in interactive test
import numpy as np

from napari import Viewer
from napari.components._viewer.qtviewer import QtViewer
from napari.util import app_context


with app_context():
    viewer  = Viewer()

    # creates a widget to view (and control) the model:
    widget = QtViewer(viewer)

    dataset = np.random.rand(10, 10, 512, 512)
    viewer.add_image(dataset)

    widget.show()
