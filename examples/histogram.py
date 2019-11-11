from napari._qt.qt_histogram import QtHistogramWidget
import napari
import numpy as np


with napari.gui_qt():
    v = napari.Viewer()
    v.add_image(np.random.randn(256, 256))
    w = QtHistogramWidget(viewer=v)
    w.show()


%load_ext autoreload
%autoreload 2
from napari._qt.qt_histogram import QtHistogramWidget
import napari
import numpy as np
v = napari.Viewer()
v.add_image(np.random.randn(256, 256))
w = QtHistogramWidget(viewer=v)
w.show()
