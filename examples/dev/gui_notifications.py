import warnings
from napari._qt.widgets.qt_viewer_buttons import QtViewerPushButton
import napari


def raise_():
    raise ValueError("error!")


def warn_():
    warnings.warn("warning!")


viewer = napari.Viewer()
layer_buttons = viewer.window.qt_viewer.layerButtons
err_btn = QtViewerPushButton(None, 'warning', 'new Error', raise_)
warn_btn = QtViewerPushButton(None, 'warning', 'new Warn', warn_)
layer_buttons.layout().insertWidget(3, warn_btn)
layer_buttons.layout().insertWidget(3, err_btn)

napari.run()
