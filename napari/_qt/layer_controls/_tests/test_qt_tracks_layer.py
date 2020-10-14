import numpy as np
from qtpy.QtCore import Qt

from napari._qt.layer_controls.qt_tracks_controls import QtTracksControls
from napari.layers import Tracks

_TRACKS = np.zeros((2, 4))
_PROPERTIES = {'speed': [50, 30], 'time': [0, 1]}


def test_tracks_controls_color_by(qtbot):
    """Check updating of the color_by combobox."""
    layer = Tracks(_TRACKS, properties=_PROPERTIES, color_by='speed')
    qtctrl = QtTracksControls(layer)
    qtbot.addWidget(qtctrl)

    # verify the color_by argument is initialized correctly
    assert layer.color_by == 'speed'
    assert qtctrl.color_by_combobox.currentText() == 'speed'

    # update color_by from the layer model
    layer.color_by = 'time'
    assert layer.color_by == 'time'
    assert qtctrl.color_by_combobox.currentText() == 'time'

    # update color_by from the qt controls
    speed_index = qtctrl.color_by_combobox.findText(
        'speed', Qt.MatchFixedString
    )
    qtctrl.color_by_combobox.setCurrentIndex(speed_index)
    assert layer.color_by == 'speed'
    assert qtctrl.color_by_combobox.currentText() == 'speed'
