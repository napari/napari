import numpy as np
import pytest
from qtpy.QtCore import Qt

from napari._qt.layer_controls.qt_tracks_controls import QtTracksControls
from napari.layers import Tracks


@pytest.fixture
def null_data() -> np.ndarray:
    return np.zeros((2, 4))


@pytest.fixture
def features() -> dict[str, list]:
    return {
        'track_id': [0, 0],
        'time': [0, 0],
        'speed': [50, 30],
    }


def test_tracks_controls_color_by(null_data, features, qtbot):
    """Check updating of the color_by combobox."""
    initial_color_by = 'time'
    layer = Tracks(null_data, features=features, color_by=initial_color_by)
    qtctrl = QtTracksControls(layer)
    qtbot.addWidget(qtctrl)

    # verify the color_by argument is initialized correctly
    assert layer.color_by == initial_color_by
    assert (
        qtctrl._color_features_combobox_control.color_by_combobox.currentText()
        == initial_color_by
    )

    # update color_by from the layer model
    layer_update_color_by = 'speed'
    layer.color_by = layer_update_color_by
    assert layer.color_by == layer_update_color_by
    assert (
        qtctrl._color_features_combobox_control.color_by_combobox.currentText()
        == layer_update_color_by
    )

    # update color_by from the qt controls
    qt_update_color_by = 'track_id'
    speed_index = (
        qtctrl._color_features_combobox_control.color_by_combobox.findText(
            qt_update_color_by, Qt.MatchFixedString
        )
    )
    qtctrl._color_features_combobox_control.color_by_combobox.setCurrentIndex(
        speed_index
    )
    assert layer.color_by == qt_update_color_by
    assert (
        qtctrl._color_features_combobox_control.color_by_combobox.currentText()
        == qt_update_color_by
    )


@pytest.mark.parametrize('color_by', ['track_id', 'speed'])
def test_color_by_same_after_features_change(
    null_data, features, color_by, qtbot
):
    """See https://github.com/napari/napari/issues/5330"""
    layer = Tracks(null_data, features=features)
    layer.color_by = color_by
    controls = QtTracksControls(layer)
    qtbot.addWidget(controls)
    assert (
        controls._color_features_combobox_control.color_by_combobox.currentText()
        == color_by
    )

    # Change the features value by removing the time column.
    layer.features = {
        'track_id': features['track_id'],
        'speed': features['speed'],
    }

    assert layer.color_by == color_by
    assert (
        controls._color_features_combobox_control.color_by_combobox.currentText()
        == color_by
    )


def test_color_by_missing_after_features_change(null_data, features, qtbot):
    """See https://github.com/napari/napari/issues/5330"""
    layer = Tracks(null_data, features=features)
    layer.color_by = 'time'
    controls = QtTracksControls(layer)
    qtbot.addWidget(controls)
    assert (
        controls._color_features_combobox_control.color_by_combobox.currentText()
        == 'time'
    )

    # Change the features value by removing the time column.
    with pytest.warns(
        UserWarning,
        match="Previous color_by key 'time' not present in features. Falling back to track_id",
    ):
        layer.features = {
            'track_id': features['track_id'],
            'speed': features['speed'],
        }

    assert layer.color_by == 'track_id'
    assert (
        controls._color_features_combobox_control.color_by_combobox.currentText()
        == 'track_id'
    )


def test_update_max_tail_length(null_data, features, qtbot):
    """Check updating of the tail length slider beyond current maximum."""
    layer = Tracks(null_data, features=features)
    controls = QtTracksControls(layer)
    qtbot.addWidget(controls)

    # verify the max_length argument is initialized correctly
    assert (
        controls._tail_length_slider_control.tail_length_slider.maximum()
        == layer._max_length
    )

    # update max_length beyond the current value
    layer.tail_length = layer._max_length + 200
    assert (
        controls._tail_length_slider_control.tail_length_slider.maximum()
        == layer._max_length
    )


def test_update_max_head_length(null_data, features, qtbot):
    """Check updating of the head length slider beyond current maximum."""
    layer = Tracks(null_data, features=features)
    controls = QtTracksControls(layer)
    qtbot.addWidget(controls)

    # verify the max_length argument is initialized correctly
    assert (
        controls._head_length_slider_control.head_length_slider.maximum()
        == layer._max_length
    )

    # update max_length beyond the current value
    layer.head_length = layer._max_length + 200
    assert (
        controls._head_length_slider_control.head_length_slider.maximum()
        == layer._max_length
    )
