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


def test_playback_settings_seed_sliders_but_do_not_bind_them(qtbot):
    settings = get_settings()
    settings.application.playback_fps = 24
    settings.application.playback_mode = LoopMode.ONCE

    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    first, second = view.slider_widgets[0], view.slider_widgets[1]
    assert first.fps == 24
    assert first.loop_mode == LoopMode.ONCE

    first.fps = 30  # as the popup does
    settings.application.playback_fps = 15
    assert first.fps == 30
    assert second.fps == 24

    settings.reset()
    assert first.fps == 30
    assert second.fps == 24


def test_popup_restore_defaults_re_reads_the_preference(qtbot):
    settings = get_settings()
    settings.application.playback_fps = -12
    settings.application.playback_mode = LoopMode.LOOP

    dims = Dims(ndim=3)
    view = QtDims(dims)
    qtbot.addWidget(view)
    first, second = view.slider_widgets[0], view.slider_widgets[1]
    first.fps = 30
    first.loop_mode = LoopMode.ONCE
    second.fps = 45

    announced = []
    first.fps_changed.connect(announced.append)
    first.play_button.reset_button.click()

    assert first.fps == -12
    assert first.play_button.reverse_check.isChecked()
    assert first.loop_mode == LoopMode.LOOP
    assert announced == [-12]  # never the intermediate -30 or 12
    assert second.fps == 45

    settings.application.playback_fps = 20
    first.play_button.reset_button.click()
    assert first.fps == 20
    assert not first.play_button.reverse_check.isChecked()
