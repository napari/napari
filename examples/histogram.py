from napari._qt.qt_histogram import HistogramWidget
import napari
import numpy as np


with napari.gui_qt():
    v = napari.Viewer()
    v.add_image(np.random.randn(512, 512))
    w = HistogramWidget(v.layers[0])
    w.show()
