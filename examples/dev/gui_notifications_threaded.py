import time
import warnings

import napari
from napari._qt.widgets.qt_viewer_buttons import QtViewerPushButton
from napari.qt import thread_worker


@thread_worker(start_thread=True)
def make_warning(*_):
    time.sleep(0.05)
    warnings.warn('Warning in another thread')


@thread_worker(start_thread=True)
def make_error(*_):
    time.sleep(0.05)
    raise ValueError("Error in another thread")


viewer = napari.Viewer()
layer_buttons = viewer.window.qt_viewer.layerButtons
err_btn = QtViewerPushButton(None, 'warning', 'new Error', make_error)
warn_btn = QtViewerPushButton(None, 'warning', 'new Warn', make_warning)
layer_buttons.layout().insertWidget(3, warn_btn)
layer_buttons.layout().insertWidget(3, err_btn)


napari.run()
