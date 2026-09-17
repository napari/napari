from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from qtpy.QtCore import QThreadPool
from qtpy.QtWidgets import QLabel, QWidget

from napari._qt.layer_controls.dynamic.widgets import (
    QtColormapControl,
    QtContrastLimitsControl,
    QtFaceColorControl,
    QtGammaSliderControl,
    QtMultiscaleLevelControl,
    QtOpacityBlendingControls,
    QtProjectionModeControl,
    QtTextVisibilityControl,
)
from napari._qt.layer_controls.dynamic.widgets.qt_contrast_limits import (
    QContrastLimitsPopup,
)
from napari.layers import Image, Points, Surface
from napari.utils.histogram import _get_computed

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

    from napari._qt.layer_controls.dynamic._tests.conftest import QtWrap


class TestColormapControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtColormapControl([image])
        qt_wrap.add_control(control)

    def test_change_colormap(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtColormapControl([image])
        qt_wrap.add_control(control)

        assert image.colormap.name == 'gray'
        assert control.colormap_combobox.currentText() == 'gray'

        control.colormap_combobox.setCurrentText('magma')
        assert image.colormap.name == 'magma'

    def test_custom_colormap(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtColormapControl([image])
        qt_wrap.add_control(control)

        # Set a custom colormap
        custom_colormap = '#aa00ff'  # Example custom colormap
        assert custom_colormap not in control.colormap_combobox._allitems
        image.colormap = custom_colormap
        assert control.colormap_combobox.currentText() == custom_colormap

    def test_init_with_rgb_image(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10, 3), dtype=np.uint8), rgb=True)
        control = QtColormapControl([image])
        qt_wrap.add_control(control)

        # For RGB images, the colormap should be disabled
        assert isinstance(
            control.colormapWidget.layout().itemAt(0).widget(), QLabel
        )
        assert (
            control.colormapWidget.layout().itemAt(0).widget().text() == 'RGB'
        )

    def test_make_colormap_button(
        self, qt_wrap: QtWrap, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            'napari._qt.utils.get_color', lambda **kwargs: '#005500'
        )
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtColormapControl([image])
        qt_wrap.add_control(control)

        control.colorbar_label.click()
        assert image.colormap.name == '#005500'


class TestQtContrastLimitsControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtContrastLimitsControl([image])
        qt_wrap.add_control(control)

    def test_change_contrast_limits(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtContrastLimitsControl([image])
        qt_wrap.add_control(control)

        np.testing.assert_array_equal(image.contrast_limits, [0, 255])
        np.testing.assert_array_equal(
            control.contrast_limits_slider.value(), (0, 255)
        )

        control.contrast_limits_slider.setValue((50, 200))
        np.testing.assert_array_equal(image.contrast_limits, [50, 200])

        image.contrast_limits = (10, 20)
        np.testing.assert_array_equal(image.contrast_limits, (10, 20))

        image.contrast_limits_range = (20, 40)
        np.testing.assert_array_equal(image.contrast_limits_range, (20, 40))
        np.testing.assert_array_equal(image.contrast_limits, (20, 40))

    def test_change_autocontrast(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtContrastLimitsControl([image])
        qt_wrap.add_control(control)

        np.testing.assert_array_equal(image.contrast_limits, [0, 255])
        np.testing.assert_array_equal(
            control.contrast_limits_slider.value(), (0, 255)
        )

        control.contrast_limits_slider.setValue((50, 200))
        np.testing.assert_array_equal(image.contrast_limits, [50, 200])

    def test_spawn_popup(self, qt_wrap: QtWrap) -> None:
        surface = Surface(
            (
                np.array([[0, 0], [1, 1]]),
                np.array([0, 1, 0]),
                np.array([0, 1, 2]),
            )
        )
        widget = QWidget()
        control = QtContrastLimitsControl([surface], parent=widget)
        qt_wrap.add_widget(widget)

        control.show_clim_popup()
        control.clim_popup.hide()  # so it doesn't show when testing
        qt_wrap.add_widget(control.clim_popup)

    def test_histogram_button(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtContrastLimitsControl([image])
        qt_wrap.add_control(control)

        control.histogram_button.click()

    def test_histogram_button_toggle_creates_inline_content(
        self, qt_wrap: QtWrap
    ) -> None:
        image = Image(np.random.rand(10, 10))
        control = QtContrastLimitsControl([image])
        qt_wrap.add_control(control)

        # Content starts hidden.
        assert control.histogram_content_widget.isHidden()
        assert control.histogram_content is None

        control._on_histogram_button_toggled(True)
        assert not control.histogram_content_widget.isHidden()
        assert control.histogram_content is not None
        assert control.histogram_content.histogram_widget is not None
        assert control.histogram_content.settings_widget is not None

        control._on_histogram_button_toggled(False)
        assert control.histogram_content_widget.isHidden()
        # Content is kept around (just hidden) until disconnect.
        assert control.histogram_content is not None

        control.disconnect_widget_controls()

    def test_content_widget_starts_hidden(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtContrastLimitsControl([image])
        qt_wrap.add_control(control)
        assert control.histogram_content_widget.isHidden()

    def test_get_widget_controls_includes_histogram(
        self, qt_wrap: QtWrap
    ) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtContrastLimitsControl([image])
        qt_wrap.add_control(control)

        controls = control.get_widget_controls()
        assert any(control.histogram_content_widget in c for c in controls)

    def test_histogram_computes_on_enable(
        self, qt_wrap: QtWrap, qtbot: QtBot
    ) -> None:
        layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
        control = QtContrastLimitsControl([layer])
        qt_wrap.add_control(control)

        control._ensure_histogram_content()
        control._schedule_histogram_compute()

        qtbot.waitUntil(
            lambda: _get_computed(layer)['counts'].sum() > 0, timeout=10000
        )
        assert _get_computed(layer)['bin_edges'].size == 257
        assert _get_computed(layer)['counts'].size == 256

        control.disconnect_widget_controls()

    def test_disconnect_widget_controls_calls_cleanup(
        self, qt_wrap: QtWrap
    ) -> None:
        image = Image(np.random.rand(10, 10))
        control = QtContrastLimitsControl([image])
        qt_wrap.add_control(control)

        control._ensure_histogram_content()
        assert control.histogram_content is not None

        control.disconnect_widget_controls()
        control.disconnect_widget_controls()

    def test_disconnect_without__ensure_histogram_content_is_safe(
        self, qt_wrap: QtWrap
    ) -> None:
        image = Image(np.random.rand(10, 10))
        control = QtContrastLimitsControl([image])
        qt_wrap.add_control(control)

        control.disconnect_widget_controls()

    def test_teardown_during_async_compute(self, qt_wrap: QtWrap) -> None:
        dask = pytest.importorskip('dask')
        layer = Image(dask.array.random.random((100, 100), chunks=(10, 10)))
        control = QtContrastLimitsControl([layer])
        qt_wrap.add_control(control)

        control._ensure_histogram_content()
        control._schedule_histogram_compute()
        QThreadPool.globalInstance().waitForDone(2000)
        control.disconnect_widget_controls()
        # should not hang or error after the worker is aborted
        QThreadPool.globalInstance().waitForDone(2000)


class TestQContrastLimitsPopup:
    def test_init(self, qtbot: QtBot) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        widget = QContrastLimitsPopup([image])
        qtbot.add_widget(widget)
        assert widget.slider.decimals() == 0
        assert not widget._needs_content_on_show

    def test_init_float_image(self, qtbot: QtBot) -> None:
        image = Image(np.zeros((10, 10), dtype=np.float32))
        widget = QContrastLimitsPopup([image])
        qtbot.add_widget(widget)
        assert widget.slider.decimals() > 0

    def test_reset_contrast_limits(self, qtbot: QtBot) -> None:
        image = Image(
            np.zeros((10, 10), dtype=np.uint8), contrast_limits=(0, 25)
        )
        widget = QContrastLimitsPopup([image])
        qtbot.add_widget(widget)

        assert widget.slider.maximum() == 25

        widget._reset()
        assert image.contrast_limits == [0, 255]
        assert widget.slider.maximum() == 255

    def test_need_content_on_show(self, qtbot: QtBot) -> None:
        """Check that the histogram content is created lazily and cleaned up."""
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        widget = QContrastLimitsPopup([image])
        qtbot.add_widget(widget)

        # The popup never auto-creates content on show anymore.
        assert not widget._needs_content_on_show
        assert widget.histogram_content is None

        widget._ensure_histogram_content()
        hc = widget.histogram_content
        assert hc is not None
        widget._ensure_histogram_content()
        assert widget.histogram_content is hc, (
            'Histogram content should not be recreated on second call'
        )

        widget._cleanup()

        assert widget.histogram_content is None

    def test_popup_opens_with_histogram(self, qtbot: QtBot) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        widget = QContrastLimitsPopup([image])
        qtbot.add_widget(widget)

        assert widget.histogram_content is None
        widget._ensure_histogram_content()

        hc = widget.histogram_content
        assert hc is not None
        assert hc.histogram_widget is not None
        assert hc.settings_widget is not None

        widget._cleanup()
        assert widget.histogram_content is None


class TestQtFaceColorControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        points = Points(np.random.rand(10, 2))
        control = QtFaceColorControl([points])
        qt_wrap.add_control(control)

    def test_change_face_color(self, qt_wrap: QtWrap) -> None:
        points = Points(np.random.rand(10, 2))
        control = QtFaceColorControl([points])
        qt_wrap.add_control(control)

        assert points.current_face_color == 'white'  # Default white
        assert np.all(control.face_color_edit.color == (1, 1, 1, 1))

        control.face_color_edit.setColor((1, 0, 0))  # Set to red
        assert points.current_face_color == 'red'

        points.current_face_color = 'blue'
        assert np.all(control.face_color_edit.color == (0, 0, 1, 1))


class TestQtGammaSliderControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtGammaSliderControl([image])
        qt_wrap.add_control(control)

    def test_update_gamma(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10), dtype=np.uint8))
        control = QtGammaSliderControl([image])
        qt_wrap.add_control(control)

        assert image.gamma == 1.0
        assert control.gamma_slider.value() == 1.0

        control.gamma_slider.setValue(2.0)
        assert image.gamma == 2.0

        image.gamma = 0.5
        assert control.gamma_slider.value() == 0.5


class TestQtHistogramControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        image = Image(np.random.rand(10, 10))
        control = QtContrastLimitsControl([image])
        qt_wrap.add_control(control)
        qt_wrap.add_widget(control.histogram_content_widget)

    def test_histogram_update(self, qt_wrap: QtWrap) -> None:
        image = Image(np.random.rand(10, 10))
        control = QtContrastLimitsControl([image])
        qt_wrap.add_control(control)
        qt_wrap.add_widget(control.histogram_content_widget)
        control._ensure_histogram_content()
        hit_content = control.histogram_content
        assert hit_content is not None
        control._ensure_histogram_content()
        assert control.histogram_content is hit_content

    @pytest.mark.parametrize(
        'layer_kwargs',
        [{}, {'rgb': True}, {'multiscale': False}],
    )
    def test_histogram_control_works_with_various_image_configs(
        self, qt_wrap: QtWrap, layer_kwargs: dict
    ) -> None:
        data = (
            np.random.rand(8, 8, 3)
            if layer_kwargs.get('rgb')
            else np.random.rand(8, 8)
        )
        layer = Image(data, **layer_kwargs)
        control = QtContrastLimitsControl([layer])
        qt_wrap.add_control(control)

        control._ensure_histogram_content()
        assert control.histogram_content is not None
        assert control.histogram_content.histogram_widget is not None
        assert control.histogram_content.settings_widget is not None

        control.disconnect_widget_controls()


class TestQtTextVisibilityControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        points = Points(np.random.rand(10, 2))
        control = QtTextVisibilityControl([points])
        qt_wrap.add_control(control)

    def test_text_visibility_toggle(self, qt_wrap: QtWrap) -> None:
        points = Points(np.random.rand(10, 2))
        control = QtTextVisibilityControl([points])
        qt_wrap.add_control(control)

        assert points.text.visible is True
        assert control.text_disp_checkbox.isChecked() is True

        control.text_disp_checkbox.setChecked(False)
        assert points.text.visible is False

        points.text.visible = True
        assert control.text_disp_checkbox.isChecked() is True


class TestQtProjectionModeControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10, 10)))
        control = QtProjectionModeControl([image])
        qt_wrap.add_control(control)

    def test_projection_mode_change(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10, 10)))
        control = QtProjectionModeControl([image])
        qt_wrap.add_control(control)

        assert image.projection_mode == 'mean'
        assert control.projection_combobox.currentText() == 'mean'

        control.projection_combobox.setCurrentText('max')
        assert image.projection_mode == 'max'

        image.projection_mode = 'sum'
        assert control.projection_combobox.currentText() == 'sum'


class TestQtOpacityBlendingControls:
    def test_init(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10)))
        control = QtOpacityBlendingControls([image])
        qt_wrap.add_control(control)

    def test_opacity_change(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10)))
        control = QtOpacityBlendingControls([image])
        qt_wrap.add_control(control)

        assert image.opacity == 1.0
        assert control.opacity_slider.value() == 1.0

        control.opacity_slider.setValue(0.5)
        assert image.opacity == 0.5

        image.opacity = 0.8
        assert control.opacity_slider.value() == 0.8

    def test_blending_change(self, qt_wrap: QtWrap) -> None:
        image = Image(np.zeros((10, 10)))
        control = QtOpacityBlendingControls([image])
        qt_wrap.add_control(control)

        assert image.blending == 'translucent'
        assert control.blend_combobox.currentText() == 'translucent'

        control.blend_combobox.setCurrentText('additive')
        assert image.blending == 'additive'

        image.blending = 'minimum'
        assert control.blend_combobox.currentText() == 'minimum'


class TestQtMultiscaleLevelControl:
    def test_init(self, qt_wrap: QtWrap) -> None:
        image = Image([np.zeros((10, 10)), np.zeros((5, 5))])
        assert image.multiscale
        control = QtMultiscaleLevelControl([image])
        qt_wrap.add_control(control)

    def test_multiscale_level_change(self, qt_wrap: QtWrap) -> None:
        image = Image([np.zeros((10, 10)), np.zeros((5, 5))])
        control = QtMultiscaleLevelControl([image])
        qt_wrap.add_control(control)
        assert control.level_combobox.count() == 3

        assert image.locked_data_level is None
        assert control.level_combobox.currentIndex() == 0

        image.locked_data_level = 0
        assert control.level_combobox.currentIndex() == 1

        control.level_combobox.setCurrentIndex(2)
        assert image.locked_data_level == 1

        image.locked_data_level = None
        assert control.level_combobox.currentIndex() == 0

    def test_mismatched_multiscale_shapes(self, qt_wrap: QtWrap) -> None:
        image1 = Image([np.zeros((10, 10)), np.zeros((5, 5))])
        image2 = Image([np.zeros((8, 8)), np.zeros((4, 4))])
        control = QtMultiscaleLevelControl([image1, image2])
        qt_wrap.add_control(control)

        assert control.level_combobox.count() == 0

    def test_locked_data_level_change(self, qt_wrap: QtWrap) -> None:
        image = Image(
            [np.zeros((10, 10)), np.zeros((5, 5))], locked_data_level=0
        )
        control = QtMultiscaleLevelControl([image])
        qt_wrap.add_control(control)

        assert image.locked_data_level == 0
        assert control.level_combobox.currentIndex() == 1
