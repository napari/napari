from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np

from napari._qt.layer_controls.dynamic.widgets._image import (
    QtDepictionControl,
    QtImageRenderControl,
    QtInterpolationComboBoxControl,
)
from napari._qt.layer_controls.dynamic.widgets._image.qt_depiction_control import (
    PlaneNormalButtons,
)
from napari.layers import Image
from napari.layers.image import _image_key_bindings

if TYPE_CHECKING:
    import pytest
    from pytestqt.qtbot import QtBot

    from napari._qt.layer_controls.dynamic.widgets.qt_widget_controls_base import (
        QtWidgetControlsBase,
    )


def add_controls_to_qtbot(
    qtbot: QtBot, container: QtWidgetControlsBase
) -> None:
    for label, widget in container.get_widget_controls():
        qtbot.add_widget(widget)
        qtbot.add_widget(label)


class TestPlaneNormalButtons:
    def test_init(self, qtbot: QtBot) -> None:
        img = Image(data=np.zeros((10, 10), dtype=np.uint8))
        widget = PlaneNormalButtons([img])
        qtbot.addWidget(widget)

    def test_orient_single_layer(
        self, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock = Mock()
        mock_oblique = Mock()
        monkeypatch.setattr(
            _image_key_bindings, 'orient_plane_normal_around_cursor', mock
        )
        monkeypatch.setattr(
            'napari._qt.layer_controls.dynamic.widgets._image.qt_depiction_control.orient_plane_normal_along_view_direction_no_gen',
            mock_oblique,
        )

        img = Image(
            data=np.zeros((10, 10, 10), dtype=np.uint8), depiction='plane'
        )
        widget = PlaneNormalButtons([img])
        qtbot.addWidget(widget)

        # Test orienting along x
        widget.x_button.click()
        mock.assert_called_once_with(img, plane_normal=(0, 0, 1))

        widget.y_button.click()
        mock.assert_called_with(img, plane_normal=(0, 1, 0))

        widget.z_button.click()
        mock.assert_called_with(img, plane_normal=(1, 0, 0))

        widget.oblique_button.click()
        mock_oblique.assert_called_once_with(img)

    def test_orient_multiple_layers(
        self, qtbot, monkeypatch: pytest.MonkeyPatch
    ):
        mock = Mock()
        monkeypatch.setattr(
            _image_key_bindings, 'orient_plane_normal_around_cursor', mock
        )
        img1 = Image(
            data=np.zeros((10, 10, 10), dtype=np.uint8), depiction='plane'
        )
        img2 = Image(
            data=np.zeros((10, 10, 10), dtype=np.uint8), depiction='plane'
        )
        widget = PlaneNormalButtons([img1, img2])
        qtbot.addWidget(widget)

        widget.x_button.click()
        assert mock.call_count == 2

        mock.assert_any_call(img1, plane_normal=(0, 0, 1))
        mock.assert_any_call(img2, plane_normal=(0, 0, 1))


class TestQtDepictionControl:
    def test_init(self, qtbot: QtBot) -> None:
        img = Image(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtDepictionControl(layers=[img])
        add_controls_to_qtbot(qtbot, control)

    def test_update_depiction(self, qtbot: QtBot) -> None:
        img = Image(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtDepictionControl(layers=[img])
        add_controls_to_qtbot(qtbot, control)

        assert img.depiction == 'volume'
        assert control.depiction_combobox.currentText() == 'volume'

        img.depiction = 'plane'
        assert control.depiction_combobox.currentText() == 'plane'

        control.depiction_combobox.setCurrentText('volume')
        assert img.depiction == 'volume'

    def test_update_plane_thickness(self, qtbot: QtBot) -> None:
        img = Image(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtDepictionControl(layers=[img])
        add_controls_to_qtbot(qtbot, control)

        assert img.plane.thickness == 1
        assert control.plane_thickness_slider.value() == 1

        img.plane.thickness = 2
        assert control.plane_thickness_slider.value() == 2

        control.plane_thickness_slider.setValue(3)
        assert img.plane.thickness == 3

    def test_change_ndisplay(self, qtbot: QtBot) -> None:
        img = Image(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtDepictionControl(layers=[img])
        add_controls_to_qtbot(qtbot, control)

        assert control._ndisplay == 2
        assert not control.depiction_combobox.isVisible()
        control._change_ndisplay(3)
        assert control._ndisplay == 3
        assert control.depiction_combobox.isVisible()

        control._change_ndisplay(2)
        assert not control.depiction_combobox.isVisible()


class TestQtInterpolationComboBoxControl:
    def test_init(self, qtbot: QtBot) -> None:
        img = Image(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtInterpolationComboBoxControl([img])
        add_controls_to_qtbot(qtbot, control)

    def test_update_interpolation_2d(self, qtbot: QtBot) -> None:
        img = Image(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtInterpolationComboBoxControl([img])
        add_controls_to_qtbot(qtbot, control)

        control._update_interpolation_combo(2)

        assert img.interpolation2d == 'nearest'
        assert control.interpolation_combobox.currentText() == 'nearest'

        img.interpolation2d = 'linear'
        assert control.interpolation_combobox.currentText() == 'linear'

        control.interpolation_combobox.setCurrentText('nearest')
        assert img.interpolation2d == 'nearest'

    def test_update_interpolation_3d(self, qtbot: QtBot) -> None:
        img = Image(data=np.zeros((5, 10, 10), dtype=np.uint8))
        control = QtInterpolationComboBoxControl([img])
        add_controls_to_qtbot(qtbot, control)

        control._update_interpolation_combo(3)

        assert img.interpolation3d == 'linear'
        assert control.interpolation_combobox.currentText() == 'linear'

        img.interpolation3d = 'cubic'
        assert control.interpolation_combobox.currentText() == 'cubic'

        control.interpolation_combobox.setCurrentText('linear')
        assert img.interpolation3d == 'linear'


class TestQtImageRenderControl:
    def test_init(self, qtbot: QtBot) -> None:
        layers = [Image(data=np.zeros((10, 10), dtype=np.uint8))]
        control = QtImageRenderControl(layers)
        add_controls_to_qtbot(qtbot, control)

    def test_change_rendering(self, qtbot: QtBot) -> None:
        img = Image(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtImageRenderControl([img])
        add_controls_to_qtbot(qtbot, control)

        assert img.rendering == 'mip'
        assert control.render_combobox.currentText() == 'mip'

        img.rendering = 'translucent'
        assert control.render_combobox.currentText() == 'translucent'

        control.render_combobox.setCurrentText('additive')
        assert img.rendering == 'additive'

    def test_change_contrast_limits(self, qtbot: QtBot) -> None:
        img = Image(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtImageRenderControl([img])
        add_controls_to_qtbot(qtbot, control)

        assert img.contrast_limits_range == [0, 255]
        assert control.iso_threshold_slider.minimum() == 0
        assert control.iso_threshold_slider.maximum() == 255

        img.contrast_limits_range = [10, 20]
        assert control.iso_threshold_slider.minimum() == 10
        assert control.iso_threshold_slider.maximum() == 20

    def test_change_visibility(self, qtbot: QtBot) -> None:
        # Test behavior when ndisplay changed
        img = Image(data=np.zeros((10, 10), dtype=np.uint8))
        control = QtImageRenderControl([img])
        add_controls_to_qtbot(qtbot, control)

        control._on_display_change_show()
        assert control.render_combobox.isVisible()
        control._on_display_change_hide()
        assert not control.render_combobox.isVisible()
        control._on_display_change_show()
        assert control.render_combobox.isVisible()
