"""For testing the Help menu"""

import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import requests
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget

from napari._app_model import get_app_model
from napari._qt._qapp_model.qactions._help import HELP_URLS, _start_viewer_tour
from napari._qt.widgets.qt_viewer_tour import _TourTooltip, build_viewer_tour


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
