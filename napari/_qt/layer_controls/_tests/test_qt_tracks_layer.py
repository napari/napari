import numpy as np
import pytest
from qtpy.QtCore import Qt

from napari._qt.layer_controls.qt_tracks_controls import QtTracksControls
from napari.layers import Tracks

_TRACKS = np.zeros((2, 4))
_PROPERTIES = {'speed': [50, 30], 'time': [0, 1]}


def test_tracks_controls_color_by(qtbot):
    """Check updating of the color_by combobox."""
    inital_color_by = 'time'
    with pytest.warns(UserWarning) as wrn:
        layer = Tracks(
            _TRACKS, properties=_PROPERTIES, color_by=inital_color_by
        )
    assert "Previous color_by key 'time' not present" in str(wrn[0].message)
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


@pytest.mark.parametrize('color_by', ('track_id', 'confidence'))
def test_color_by_same_after_properties_change(color_by, qtbot):
    """See https://github.com/napari/napari/issues/5330"""
    data = np.array(
        [
            [1, 0, 236, 0],
            [1, 1, 236, 100],
            [1, 2, 236, 200],
            [2, 0, 436, 0],
            [2, 1, 436, 100],
            [2, 2, 436, 200],
            [3, 0, 636, 0],
            [3, 1, 636, 100],
            [3, 2, 636, 200],
        ]
    )
    initial_properties = {
        'track_id': data[:, 0],
        'time': data[:, 1],
        'confidence': np.ones(data.shape[0]),
    }
    layer = Tracks(data, properties=initial_properties)
    layer.color_by = color_by
    controls = QtTracksControls(layer)
    qtbot.addWidget(controls)

    # Change the properties value by removing the time column.
    layer.properties = {
        'track_id': initial_properties['track_id'],
        'confidence': initial_properties['confidence'],
    }

    assert layer.color_by == color_by
