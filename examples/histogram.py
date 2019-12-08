from napari._qt.qt_histogram import QtHistogramWidget
import napari
from scipy import stats


with napari.gui_qt():
    v = napari.Viewer()
    v.add_image(
        stats.distributions.gamma.rvs(3, 20, 1, 256 ** 2).reshape(256, 256)
    )
    w = QtHistogramWidget(viewer=v, vertical=True)
    dw = v.window.add_dock_widget(w, area='right')
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
