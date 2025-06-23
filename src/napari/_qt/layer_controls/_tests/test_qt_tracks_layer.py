import numpy as np
import pytest
from qtpy.QtCore import Qt

from napari._qt.layer_controls.qt_tracks_controls import QtTracksControls
from napari.layers import Tracks


@pytest.fixture
def null_data() -> np.ndarray:
    return np.zeros((2, 4))


@pytest.fixture
def properties() -> dict[str, list]:
    return {
        'track_id': [0, 0],
        'time': [0, 0],
        'speed': [50, 30],
    }


def test_tracks_controls_color_by(null_data, properties, qtbot):
    """Check updating of the color_by combobox."""
    inital_color_by = 'time'
    layer = Tracks(null_data, properties=properties, color_by=inital_color_by)
    controls = QtTracksControls(layer)
    qtbot.addWidget(controls)

    # verify the color_by argument is initialized correctly
    assert layer.color_by == inital_color_by
    assert controls.color_by_combobox.currentText() == inital_color_by

    # update color_by from the layer model
    layer_update_color_by = 'speed'
    layer.color_by = layer_update_color_by
    assert layer.color_by == layer_update_color_by
    assert controls.color_by_combobox.currentText() == layer_update_color_by

    # update color_by from the qt controls
    qt_update_color_by = 'track_id'
    speed_index = controls.color_by_combobox.findText(
        qt_update_color_by, Qt.MatchFixedString
    )
    controls.color_by_combobox.setCurrentIndex(speed_index)
    assert layer.color_by == qt_update_color_by
    assert controls.color_by_combobox.currentText() == qt_update_color_by


@pytest.mark.parametrize('color_by', ['track_id', 'speed'])
def test_color_by_same_after_properties_change(
    null_data, properties, color_by, qtbot
):
    """See https://github.com/napari/napari/issues/5330"""
    layer = Tracks(null_data, properties=properties)
    layer.color_by = color_by
    controls = QtTracksControls(layer)
    qtbot.addWidget(controls)
    assert controls.color_by_combobox.currentText() == color_by

    # Change the properties value by removing the time column.
    layer.properties = {
        'track_id': properties['track_id'],
        'speed': properties['speed'],
    }

    assert layer.color_by == color_by
    assert controls.color_by_combobox.currentText() == color_by


def test_color_by_missing_after_properties_change(
    null_data, properties, qtbot
):
    """See https://github.com/napari/napari/issues/5330"""
    layer = Tracks(null_data, properties=properties)
    layer.color_by = 'time'
    controls = QtTracksControls(layer)
    qtbot.addWidget(controls)
    assert controls.color_by_combobox.currentText() == 'time'

    # Change the properties value by removing the time column.
    with pytest.warns(
        UserWarning,
        match="Previous color_by key 'time' not present in features. Falling back to track_id",
    ):
        layer.properties = {
            'track_id': properties['track_id'],
            'speed': properties['speed'],
        }

    assert layer.color_by == 'track_id'
    assert controls.color_by_combobox.currentText() == 'track_id'


def test_update_max_tail_length(null_data, properties, qtbot):
    """Check updating of the tail length slider beyond current maximum."""
    layer = Tracks(null_data, properties=properties)
    controls = QtTracksControls(layer)
    qtbot.addWidget(controls)

    # verify the max_length argument is initialized correctly
    assert controls.tail_length_slider.maximum() == layer._max_length

    # update max_length beyond the current value
    layer.tail_length = layer._max_length + 200
    assert controls.tail_length_slider.maximum() == layer._max_length


def test_update_max_head_length(null_data, properties, qtbot):
    """Check updating of the head length slider beyond current maximum."""
    layer = Tracks(null_data, properties=properties)
    controls = QtTracksControls(layer)
    qtbot.addWidget(controls)

    # verify the max_length argument is initialized correctly
    assert controls.head_length_slider.maximum() == layer._max_length

    # update max_length beyond the current value
    layer.head_length = layer._max_length + 200
    assert controls.head_length_slider.maximum() == layer._max_length
