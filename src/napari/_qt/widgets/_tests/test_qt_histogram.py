import numpy as np

from napari._qt.widgets.qt_histogram import QtHistogramWidget
from napari.layers import Image
from napari.settings import get_settings
from napari.utils.theme import get_theme


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
