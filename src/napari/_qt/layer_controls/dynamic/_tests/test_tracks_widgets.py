from __future__ import annotations

from typing import TYPE_CHECKING

from napari._qt.layer_controls.dynamic.widgets._tracks import (
    QtColormapComboBoxControl,
    QtColorPropertiesComboBoxControl,
    QtGraphCheckBoxControl,
    QtHeadLengthSliderControl,
    QtHideCompletedTracksCheckBoxControl,
    QtIdCheckBoxControl,
    QtTailDisplayCheckBoxControl,
    QtTailLengthSliderControl,
    QtTailWidthSliderControl,
)
from napari.layers import Tracks

if TYPE_CHECKING:
    from napari._qt.layer_controls.dynamic._tests.conftest import QtWrap


class TestQtColorPropertiesComboBoxControl:
    def test_init(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtColorPropertiesComboBoxControl([tracks])
        qt_wrap.add_control(control)

    def test_color_by(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtColorPropertiesComboBoxControl([tracks])
        qt_wrap.add_control(control)

        assert tracks.color_by == 'track_id'
        assert control.color_by_combobox.currentText() == 'track_id'

        tracks.color_by = 'time'
        assert control.color_by_combobox.currentText() == 'time'

        control.color_by_combobox.setCurrentText('track_id')
        assert tracks.color_by == 'track_id'

    def test_properties(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtColorPropertiesComboBoxControl([tracks])
        qt_wrap.add_control(control)

        assert control.color_by_combobox.count() == 3

        tracks.properties = {
            'track_id': [0, 0, 1, 1],
            'time': [0, 1, 0, 1],
            'speed': [50, 30, 20, 10],
            'velocity': [50, 30, 20, 10],
        }
        assert control.color_by_combobox.count() == 4


class TestQtColormapComboBoxControl:
    def test_init(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtColormapComboBoxControl([tracks])
        qt_wrap.add_control(control)

    def test_colormap(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtColormapComboBoxControl([tracks])
        qt_wrap.add_control(control)

        assert tracks.colormap == 'turbo'
        assert control.colormap_combobox.currentText() == 'turbo'

        tracks.colormap = 'magma'
        assert control.colormap_combobox.currentText() == 'magma'

        control.colormap_combobox.setCurrentText('viridis')
        assert tracks.colormap == 'viridis'


class TestQtGraphCheckBoxControl:
    def test_init(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtGraphCheckBoxControl([tracks])
        qt_wrap.add_control(control)

    def test_display_graph(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtGraphCheckBoxControl([tracks])
        qt_wrap.add_control(control)

        assert tracks.display_graph is True
        assert control.graph_checkbox.isChecked() is True

        control.graph_checkbox.setChecked(False)
        assert tracks.display_graph is False


class TestQtHeadLengthSliderControl:
    def test_init(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtHeadLengthSliderControl([tracks])
        qt_wrap.add_control(control)

    def test_head_length(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtHeadLengthSliderControl([tracks])
        qt_wrap.add_control(control)

        assert tracks.head_length == 0
        assert control.head_length_slider.value() == 0

        control.head_length_slider.setValue(5)
        assert tracks.head_length == 5

        tracks.head_length = 10
        assert control.head_length_slider.value() == 10


class TestQtHideCompletedTracksCheckBoxControl:
    def test_init(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtHideCompletedTracksCheckBoxControl([tracks])
        qt_wrap.add_control(control)

    def test_hide_completed_tracks(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtHideCompletedTracksCheckBoxControl([tracks])
        qt_wrap.add_control(control)

        assert tracks.hide_completed_tracks is False
        assert control.hide_completed_tracks_checkbox.isChecked() is False

        control.hide_completed_tracks_checkbox.setChecked(True)
        assert tracks.hide_completed_tracks is True

        tracks.hide_completed_tracks = False
        assert control.hide_completed_tracks_checkbox.isChecked() is False


class TestQtIdCheckBoxControl:
    def test_init(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtIdCheckBoxControl([tracks])
        qt_wrap.add_control(control)

    def test_display_id(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtIdCheckBoxControl([tracks])
        qt_wrap.add_control(control)

        assert tracks.display_id is False
        assert control.display_id_checkbox.isChecked() is False

        control.display_id_checkbox.setChecked(True)
        assert tracks.display_id is True

        tracks.display_id = False
        assert control.display_id_checkbox.isChecked() is False


class TestQtTailLengthSliderControl:
    def test_init(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtTailLengthSliderControl([tracks])
        qt_wrap.add_control(control)

    def test_tail_length(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtTailLengthSliderControl([tracks])
        qt_wrap.add_control(control)

        assert tracks.tail_length == 30
        assert control.tail_length_slider.value() == 30

        control.tail_length_slider.setValue(5)
        assert tracks.tail_length == 5

        tracks.tail_length = 15
        assert control.tail_length_slider.value() == 15


class TestQtTailWidthSliderControl:
    def test_init(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtTailWidthSliderControl([tracks])
        qt_wrap.add_control(control)

    def test_tail_width(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtTailWidthSliderControl([tracks])
        qt_wrap.add_control(control)

        assert tracks.tail_width == 2.0
        assert control.tail_width_slider.value() == 2.0

        control.tail_width_slider.setValue(5)
        assert tracks.tail_width == 5

        tracks.tail_width = 10
        assert control.tail_width_slider.value() == 10


class TestQtTailDisplayCheckBoxControl:
    def test_init(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtTailDisplayCheckBoxControl([tracks])
        qt_wrap.add_control(control)

    def test_display_tail(self, qt_wrap: QtWrap, tracks_data) -> None:
        tracks = Tracks(**tracks_data)
        control = QtTailDisplayCheckBoxControl([tracks])
        qt_wrap.add_control(control)

        assert tracks.display_tail is True
        assert control.tail_checkbox.isChecked() is True

        control.tail_checkbox.setChecked(False)
        assert tracks.display_tail is False

        tracks.display_tail = True
        assert control.tail_checkbox.isChecked() is True
