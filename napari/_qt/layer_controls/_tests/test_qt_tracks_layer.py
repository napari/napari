import numpy as np
from qtpy.QtCore import Qt

from napari._qt.layer_controls.qt_tracks_controls import QtTracksControls
from napari.layers import Tracks

_TRACKS = np.zeros((2, 4))
_PROPERTIES = {'speed': [50, 30], 'time': [0, 1]}


def test_tracks_controls_color_by(qtbot):
    """Check updating of the color_by combobox."""
    inital_color_by = 'time'
    layer = Tracks(_TRACKS, properties=_PROPERTIES, color_by=inital_color_by)
    qtctrl = QtTracksControls(layer)
    qtbot.addWidget(qtctrl)

    # verify the color_by argument is initialized correctly
    assert layer.color_by == inital_color_by
    assert qtctrl.color_by_combobox.currentText() == inital_color_by

    # update color_by from the layer model
    layer_update_color_by = 'speed'
    layer.color_by = layer_update_color_by
    assert layer.color_by == layer_update_color_by
    assert qtctrl.color_by_combobox.currentText() == layer_update_color_by

    # update color_by from the qt controls
    qt_update_color_by = 'track_id'
    speed_index = qtctrl.color_by_combobox.findText(
        qt_update_color_by, Qt.MatchFixedString
    )
    qtctrl.color_by_combobox.setCurrentIndex(speed_index)
    assert layer.color_by == qt_update_color_by
    assert qtctrl.color_by_combobox.currentText() == qt_update_color_by
