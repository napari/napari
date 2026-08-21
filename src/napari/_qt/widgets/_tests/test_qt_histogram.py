import numpy as np

from napari._qt.widgets.qt_histogram import QtHistogramWidget
from napari._qt.widgets.qt_histogram_content import QtHistogramContentWidget
from napari._qt.widgets.qt_histogram_settings import QtHistogramSettingsWidget
from napari.layers import Image
from napari.settings import get_settings
from napari.utils.histogram import _get_computed
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


def test_qt_histogram_widget_shows_computed(qtbot):
    """The widget renders the computed histogram via updated/completed events."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    layer.histogram.compute(layer)

    computed = _get_computed(layer)
    assert len(computed['counts']) == 256
    # The visual was fed data (no crash) and the canvas is alive.
    assert widget.histogram_visual is not None
    widget.cleanup()


def test_qt_histogram_widget_updates_theme(qtbot):
    settings = get_settings()
    old_theme = settings.appearance.theme
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    layer.histogram.compute(layer)

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


def test_qt_histogram_widget_updates_from_settings_theme(
    make_napari_viewer, qtbot
):
    """Histogram widget responds to theme changes via settings (canonical source)."""
    settings = get_settings()
    old_theme = settings.appearance.theme
    viewer = make_napari_viewer()
    layer = viewer.add_image(
        np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    )
    controls = viewer.window._qt_viewer.controls.widgets[layer]
    controls._contrast_limits_control.ensure_content()
    widget = (
        controls._contrast_limits_control._histogram_content.histogram_widget
    )
    assert widget is not None

    qtbot.addWidget(widget)
    layer.histogram.compute(layer)

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
    finally:
        settings.appearance.theme = old_theme


def test_histogram_visual_set_data_clear_path(qtbot):
    """Calling set_data with no bins/counts should clear the visual."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    visual = widget.histogram_visual

    # First set some data to get a non-empty state
    layer.histogram.compute(layer)
    computed = _get_computed(layer)
    visual.set_data(
        bin_edges=computed['bin_edges'],
        counts=computed['counts'],
        gamma=1.0,
        clims=(0.25, 0.75),
        data_range=(0, 1),
    )

    # Now call set_data with None to trigger _clear path
    visual.set_data()
    # After clear, gamma should be reset to 1.0
    assert visual._gamma == 1.0
    assert visual._clims is None
    assert visual._data_range is None

    widget.cleanup()


def test_histogram_visual_update_lut_line_clims_equal(qtbot):
    """LUT line should handle equal clim values gracefully."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    visual = widget.histogram_visual
    layer.histogram.compute(layer)
    computed = _get_computed(layer)

    # Call with clims where min == max
    visual.set_data(
        bin_edges=computed['bin_edges'],
        counts=computed['counts'],
        gamma=1.0,
        clims=(0.5, 0.5),  # equal clims
        data_range=(0, 1),
    )
    # Should not crash; uses the else branch in _update_lut_line
    assert visual._clims == (0.5, 0.5)

    widget.cleanup()


def test_histogram_visual_destroy(qtbot):
    """Calling destroy on the histogram visual should clean up sub-visuals."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    visual = widget.histogram_visual

    # destroy should not crash
    visual.destroy()

    widget.cleanup()


def test_histogram_visual_update_bars_empty(qtbot):
    """_update_bars with fewer than 2 bins should call _set_empty_data."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    visual = widget.histogram_visual

    # Call _update_bars directly with a single bin (len(bins) < 2)
    visual._update_bars(np.array([0.0]), np.array([5.0]))
    # Should not crash; calls _set_empty_data internally
    # After _set_empty_data, the bars mesh should have 3 dummy vertices
    assert visual._bars.mesh_data.get_vertices() is not None

    widget.cleanup()


def test_histogram_visual_update_bars_zero_range(qtbot):
    """_update_bars should handle zero bin range (all bins identical)."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    visual = widget.histogram_visual

    # All bins have the same value → bin_range == 0 → should use bin_range = 1
    bins = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    counts = np.array([10.0, 5.0], dtype=np.float32)
    visual._update_bars(bins, counts)
    # Should not crash; with 2 bins, should produce 8 vertices (4 per bar)
    vertices = visual._bars.mesh_data.get_vertices()
    assert vertices is not None
    assert len(vertices) == 8, (
        'zero-range bars should produce 8 vertices for 2 bins'
    )

    widget.cleanup()


def test_qt_histogram_layer_bar_color(qtbot):
    """_layer_bar_color should return a 4-tuple based on the layer's colormap."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    # Default colormap (gray) → bar color should be a 4-element tuple
    color = widget._layer_bar_color()
    assert len(color) == 4
    assert all(0 <= c <= 1 for c in color)

    # With a reversed colormap, the bar color should still be non-zero
    # (the method uses map([0.8]) to avoid black-on-black for gray_r)
    layer.colormap = 'gray_r'
    color_r = widget._layer_bar_color()
    assert len(color_r) == 4
    # Even on a reversed colormap, the 0.8 position is near-white, so at
    # least one channel should be > 0.5.
    assert any(c > 0.5 for c in color_r), (
        f'gray_r bar color should be light, got {color_r}'
    )

    widget.cleanup()
