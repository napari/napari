import warnings
from napari._qt.widgets.qt_viewer_buttons import QtViewerPushButton
import napari

from napari.utils.key_bindings import action_manager


def replay():
    action_manager.play(['reset_view','transpose_axes', 'transpose_axes', 'transpose_axes'])

layer_buttons = viewer.window.qt_viewer.layerButtons
play_button = QtViewerPushButton(None, 'play', 'Play', replay)
layer_buttons.layout().insertWidget(3, play_button)

napari.run()
