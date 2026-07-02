import numpy as np

from napari._qt.widgets.qt_histogram import QtHistogramWidget
from napari._qt.widgets.qt_histogram_content import QtHistogramContentWidget
from napari._qt.widgets.qt_histogram_settings import QtHistogramSettingsWidget
from napari.layers import Image
from napari.settings import get_settings
from napari.utils.theme import get_theme


def test_qt_histogram_settings_mode_sync(qtbot):
    """Settings widget mode combobox should sync bidirectionally with model."""
    layer = Image(np.random.rand(10, 10))
    model = layer.histogram
    widget = QtHistogramSettingsWidget(model)
    qtbot.addWidget(widget)

    # Default state
    assert widget.mode_combobox.currentText() == 'canvas'
    assert model.mode == 'canvas'

    # UI → model: changing combobox updates model
    widget.mode_combobox.setCurrentText('full')
    assert model.mode == 'full'

    # Model → UI: changing model updates combobox
    model.mode = 'canvas'
    assert widget.mode_combobox.currentText() == 'canvas'

    widget.cleanup()


def test_qt_histogram_settings_log_scale_sync(qtbot):
    """Settings widget log scale checkbox should sync bidirectionally with model."""
    layer = Image(np.random.rand(10, 10))
    model = layer.histogram
    widget = QtHistogramSettingsWidget(model)
    qtbot.addWidget(widget)

    # Default state
    assert not widget.log_scale_checkbox.isChecked()
    assert not model.log_scale

    # UI → model: toggling checkbox updates model
    widget.log_scale_checkbox.setChecked(True)
    assert model.log_scale

    # Model → UI: changing model updates checkbox
    model.log_scale = False
    assert not widget.log_scale_checkbox.isChecked()

    widget.cleanup()


def test_qt_histogram_content_composition_and_cleanup(qtbot):
    """Content widget should create histogram + settings children and clean up."""
    layer = Image(np.random.rand(10, 10))
    content = QtHistogramContentWidget(layer)
    qtbot.addWidget(content)

    # Both child widgets exist
    assert content.histogram_widget is not None
    assert content.settings_widget is not None
    assert content.settings_widget.mode_combobox is not None
    assert content.settings_widget.log_scale_checkbox is not None

    # Settings controls are functional
    content.settings_widget.mode_combobox.setCurrentText('full')
    assert layer.histogram.mode == 'full'
    content.settings_widget.log_scale_checkbox.setChecked(True)
    assert layer.histogram.log_scale

    # Cleanup does not crash
    content.cleanup()


def test_qt_histogram_widget_updates_theme(qtbot):
    settings = get_settings()
    old_theme = settings.appearance.theme
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    layer.histogram.enabled = True
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    layer.histogram.compute()

    try:
        settings.appearance.theme = 'light'
        light_theme = get_theme('light')

        qtbot.waitUntil(
            lambda: np.allclose(
                widget.canvas.bgcolor.rgba[:3],
                np.array(light_theme.canvas.as_rgb_tuple()) / 255,
            )
        )

        assert widget.histogram_visual._lut_color == (
            *(
                np.array(light_theme.highlight.as_rgb_tuple(), dtype=float)
                / 255
            ),
            0.95,
        )
        assert widget.histogram_visual._axes_color == (
            *(np.array(light_theme.text.as_rgb_tuple(), dtype=float) / 255),
            0.7,
        )
    finally:
        settings.appearance.theme = old_theme
        widget.cleanup()


def test_qt_histogram_widget_updates_from_viewer_theme(
    make_napari_viewer, qtbot
):
    viewer = make_napari_viewer()
    layer = viewer.add_image(
        np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    )
    controls = viewer.window._qt_viewer.controls.widgets[layer]
    controls._histogram_control.ensure_content()
    widget = controls._histogram_control.histogram_widget
    assert widget is not None

    qtbot.addWidget(widget)
    layer.histogram.enabled = True
    layer.histogram.compute()

    viewer.theme = 'light'
    light_theme = get_theme('light')

    qtbot.waitUntil(
        lambda: np.allclose(
            widget.canvas.bgcolor.rgba[:3],
            np.array(light_theme.canvas.as_rgb_tuple()) / 255,
        )
    )

    assert widget.histogram_visual._lut_color == (
        *(np.array(light_theme.highlight.as_rgb_tuple(), dtype=float) / 255),
        0.95,
    )
