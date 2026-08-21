from napari._qt.widgets.qt_dims import QtDims, QtDimSliderWidget
from napari.components import Dims
from napari.settings import get_settings
from napari.settings._constants import LoopMode


def test_same_margin_popup(qtbot):
    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    slider: QtDimSliderWidget = view.slider_widgets[0]
    # Lazily create the widget
    assert slider.margins_popup is None
    slider.show_margins_popup()
    old_margins_popup = slider.margins_popup
    assert old_margins_popup is not None
    # Reuse old margins popup
    slider.show_margins_popup()
    assert old_margins_popup is slider.margins_popup


def test_move_margin_popup(qtbot):
    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    slider: QtDimSliderWidget = view.slider_widgets[0]
    slider.show_margins_popup()
    # Check that values of the left slider matches the
    # values of the dims margin_right after the
    # margin_right has been moved within the dims
    dims.margin_right = (2, 0, 0)
    assert slider.margins_popup.right_slider.value() == dims.margin_right[0]
    slider.margins_popup.left_slider.setValue(1)
    assert slider.margins_popup.left_slider.value() == dims.margin_left[0]


def test_playback_settings_seed_from_preferences(qtbot):
    """A new slider starts at the playback settings from the preferences."""
    settings = get_settings()
    settings.application.playback_fps = 27
    settings.application.playback_mode = LoopMode.ONCE

    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    slider: QtDimSliderWidget = view.slider_widgets[0]

    assert slider.fps == 27
    assert slider.loop_mode == LoopMode.ONCE


def test_preference_change_keeps_per_axis_override(qtbot):
    """Changing the preference must not stomp an axis's own playback speed.

    Each slider owns its fps, set per axis from the play button's popup, so a
    later preference change is a new default rather than an edict: it applies
    to sliders created afterwards and leaves existing overrides alone.
    """
    settings = get_settings()
    settings.application.playback_fps = 10

    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    slider: QtDimSliderWidget = view.slider_widgets[0]

    slider.fps = 30  # as the popup does
    settings.application.playback_fps = 15

    assert slider.fps == 30


def test_popup_restore_defaults_resets_playback_settings(qtbot):
    """The popup's "restore defaults" re-reads the preference for that axis."""
    settings = get_settings()
    settings.application.playback_fps = 10
    settings.application.playback_mode = LoopMode.LOOP

    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    slider: QtDimSliderWidget = view.slider_widgets[0]

    slider.fps = 30
    slider.loop_mode = LoopMode.ONCE

    slider.play_button.reset_button.click()

    assert slider.fps == 10
    assert slider.loop_mode == LoopMode.LOOP


def test_popup_restore_defaults_resets_only_its_own_axis(qtbot):
    """Restoring one axis's playback settings leaves the other axes alone."""
    settings = get_settings()
    settings.application.playback_fps = 10

    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    first: QtDimSliderWidget = view.slider_widgets[0]
    second: QtDimSliderWidget = view.slider_widgets[1]

    first.fps = 30
    second.fps = 45

    first.play_button.reset_button.click()

    assert first.fps == 10
    assert second.fps == 45


def test_popup_restore_defaults_restores_reverse_playback(qtbot):
    """A negative preference fps restores as reverse playback."""
    settings = get_settings()
    settings.application.playback_fps = -12

    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    slider: QtDimSliderWidget = view.slider_widgets[0]

    slider.fps = 30

    slider.play_button.reset_button.click()

    assert slider.fps == -12
    assert slider.play_button.reverse_check.isChecked()


def test_popup_restore_defaults_announces_fps_once(qtbot):
    """Reset publishes the default speed only, never an intermediate one.

    Reversing direction and changing magnitude are one edit. Announcing the
    new direction against the old magnitude would hand a running animation
    (which follows ``fps_changed``) a speed the user never chose.
    """
    settings = get_settings()
    settings.application.playback_fps = -12

    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    slider: QtDimSliderWidget = view.slider_widgets[0]

    slider.fps = 30

    announced = []
    slider.fps_changed.connect(announced.append)
    slider.play_button.reset_button.click()

    assert announced == [-12]


def test_preferences_reset_then_popup_restore_defaults(qtbot):
    """The full workflow this change is about.

    "Restore Defaults" in preferences restores the preference and leaves the
    per-axis overrides alone; the popup's own button is what re-reads it.
    """
    settings = get_settings()
    settings.application.playback_fps = 24
    settings.application.playback_mode = LoopMode.ONCE
    default_fps = settings.application._defaults['playback_fps']
    default_mode = settings.application._defaults['playback_mode']

    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    first: QtDimSliderWidget = view.slider_widgets[0]
    second: QtDimSliderWidget = view.slider_widgets[1]
    assert first.fps == 24  # seeded from the preference

    first.fps = 30  # as the popup does
    settings.reset()

    # the preference went back to its default; the axes did not move
    assert settings.application.playback_fps == default_fps
    assert first.fps == 30
    assert second.fps == 24

    first.play_button.reset_button.click()

    assert first.fps == default_fps
    assert first.loop_mode == default_mode
    assert second.fps == 24  # still untouched
