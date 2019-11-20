from napari._qt.qt_histogram import QtHistogramWidget
import napari
import numpy as np


with napari.gui_qt():
    v = napari.Viewer()
    v.add_image(np.random.randn(10, 256, 256))
    w = QtHistogramWidget(viewer=v)
    v.window.add_dock_widget('bottom', w)
    # w2 = QtHistogramWidget(viewer=v, vertical=True)
    # v.window.add_dock_widget('right', w2)


# %load_ext autoreload
# %autoreload 2
# from napari._qt.qt_histogram import QtHistogramWidget
# import skimage.data as skid
# import napari
# import numpy as np
# v = napari.Viewer()
# v.add_image(skid.camera())
# w = QtHistogramWidget(viewer=v, vertical=True)
# w.show()
# w2 = QtHistogramWidget(viewer=v, link='view')
