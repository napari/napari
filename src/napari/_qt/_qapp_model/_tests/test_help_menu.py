"""For testing the Help menu"""

import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import requests
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QWidget

from napari._app_model import get_app_model
from napari._qt._qapp_model.qactions._help import HELP_URLS, _start_viewer_tour
from napari._qt.widgets.qt_viewer_tour import (
    _BUILTIN_TOUR_TARGETS,
    _TOOLTIP_WIDTH,
    GuidedTour,
    TourStep,
    _TourTooltip,
    build_viewer_tour,
    resolve_tour_target,
)


@pytest.mark.parametrize('url', HELP_URLS.keys())
def test_help_urls(url):
    if url == 'release_notes':
        pytest.skip('No release notes for dev version')

    r = requests.head(HELP_URLS[url])
    r.raise_for_status()


@pytest.mark.parametrize(
    'action_id',
    [
        'napari.window.help.info',
        'napari.window.help.about_macos',
    ]
    if sys.platform == 'darwin'
    else ['napari.window.help.info'],
)
def test_about_action(make_napari_viewer, action_id):
    app = get_app_model()
    viewer = make_napari_viewer()

    with mock.patch(
        'napari._qt.dialogs.qt_about.QtAbout.showAbout'
    ) as mock_about:
        app.commands.execute_command(action_id)
    mock_about.assert_called_once_with(viewer.window._qt_window)


def test_tour_tooltip_widens_for_long_nav_labels(qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)
    tooltip = _TourTooltip(parent)
    qtbot.addWidget(tooltip)

    tooltip.set_content(
        'Title', 'Body.', 2, 5
    )  # middle step: all 3 nav buttons visible
    assert tooltip.width() == _TOOLTIP_WIDTH

    tooltip._back.setText('Previous step in the guided tour')
    tooltip._next.setText('Next step in the guided tour')
    tooltip._skip.setText('Skip this entire guided tour now')
    tooltip._update_size()

    assert tooltip.width() > _TOOLTIP_WIDTH
    for button in (tooltip._back, tooltip._next, tooltip._skip):
        assert (
            button.geometry().width() >= button.minimumSizeHint().width() - 2
        )

    # width should shrink back down once labels return to normal
    tooltip.set_content('Title', 'Body.', 3, 5)
    assert tooltip.width() == _TOOLTIP_WIDTH


def test_tour_tooltip_keyboard_shortcuts(qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)
    tooltip = _TourTooltip(parent)
    qtbot.addWidget(tooltip)

    with qtbot.waitSignal(tooltip.next_clicked, timeout=1000):
        qtbot.keyPress(tooltip, Qt.Key.Key_N)
    with qtbot.waitSignal(tooltip.back_clicked, timeout=1000):
        qtbot.keyPress(tooltip, Qt.Key.Key_P)
    with qtbot.waitSignal(tooltip.skip_clicked, timeout=1000):
        qtbot.keyPress(tooltip, Qt.Key.Key_Escape)


@pytest.mark.parametrize(
    ('shape', 'expect_dims_step'),
    [((4, 4), False), ((4, 4, 4), True)],
)
def test_tour_skips_dims_step_without_extra_dims(
    make_napari_viewer, shape, expect_dims_step
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros(shape))
    tour = build_viewer_tour(viewer.window._qt_window)

    titles = []
    index = tour._seek(0, 1)
    while index is not None:
        titles.append(tour._steps[index].title)
        index = tour._seek(index + 1, 1)

    assert ('Dimension sliders' in titles) is expect_dims_step


def test_resolve_tour_target_builtins(make_napari_viewer):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window

    for name in _BUILTIN_TOUR_TARGETS:
        assert resolve_tour_target(qt_window, name) is not None

    assert resolve_tour_target(qt_window, 'nonexistent') is None


def test_resolve_tour_target_plugin_dock_widget(make_napari_viewer):
    viewer = make_napari_viewer(show=True)
    qt_window = viewer.window._qt_window

    label = QLabel('hello')
    viewer.window.add_dock_widget(label, name='My Plugin Widget')

    assert resolve_tour_target(qt_window, 'My Plugin Widget') is label


def test_tour_skips_missing_target(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    window.resize(640, 480)
    window.show()

    first = QWidget(window)
    first.setGeometry(0, 0, 50, 50)
    first.show()
    last = QWidget(window)
    last.setGeometry(60, 0, 50, 50)
    last.show()

    tour = GuidedTour(
        [
            TourStep(target=lambda: first, title='First', body=''),
            TourStep(target=lambda: None, title='Missing', body=''),
            TourStep(target=lambda: last, title='Last', body=''),
        ],
        window,
    )
    qtbot.addWidget(tour._tooltip)
    qtbot.addWidget(tour._overlay)
    tour.start()
    qtbot.waitUntil(lambda: tour._steps[tour._current].title == 'First')

    tour._on_next()
    assert tour._steps[tour._current].title == 'Last'
    tour.close_tour()


def test_tour_current_step_unchanged_when_target_becomes_hidden(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    window.resize(640, 480)
    window.show()

    first = QWidget(window)
    first.setGeometry(0, 0, 50, 50)
    first.show()
    second = QWidget(window)
    second.setGeometry(60, 0, 50, 50)
    second.show()

    tour = GuidedTour(
        [
            TourStep(target=lambda: first, title='First', body=''),
            TourStep(target=lambda: second, title='Second', body=''),
        ],
        window,
    )
    qtbot.addWidget(tour._tooltip)
    qtbot.addWidget(tour._overlay)
    tour.start()
    qtbot.waitUntil(lambda: tour._steps[tour._current].title == 'First')

    second.hide()
    tour._on_next()
    assert tour._steps[tour._current].title == 'First'
    tour.close_tour()


def test_start_viewer_tour():
    tour = mock.Mock()
    qt_window = SimpleNamespace(_viewer_tour=None)
    viewer = SimpleNamespace(layers=[], open_sample=mock.Mock())
    window = SimpleNamespace(
        _qt_window=qt_window,
        _qt_viewer=SimpleNamespace(viewer=viewer),
    )

    with mock.patch(
        'napari._qt._qapp_model.qactions._help.build_viewer_tour',
        return_value=tour,
    ) as mock_build_tour:
        _start_viewer_tour(window)
        _start_viewer_tour(window)

    viewer.open_sample.assert_called_once_with('napari', 'balls_3d')
    mock_build_tour.assert_called_once_with(qt_window)
    tour.finished.connect.assert_called_once()
    tour.start.assert_called_once()
    assert qt_window._viewer_tour is tour

    tour.finished.connect.call_args.args[0]()
    assert qt_window._viewer_tour is None
