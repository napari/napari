"""For testing the Help menu"""

import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import requests
from qt_tour import TourAnchor
from qtpy.QtWidgets import QLabel

from napari._app_model import get_app_model
from napari._qt._qapp_model.qactions._help import HELP_URLS, _take_viewer_tour
from napari._qt.widgets.qt_viewer_tour import (
    _BUILTIN_TOUR_TARGETS,
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


def test_build_viewer_tour_first_step_is_centered(make_napari_viewer):
    viewer = make_napari_viewer()
    qt_window = viewer.window._qt_window
    tour = build_viewer_tour(viewer.window, sample=None)

    first_step = tour._steps[0]
    assert first_step.target() is qt_window._qt_viewer.canvas.native
    assert first_step.anchor == TourAnchor.CENTER


def test_build_viewer_tour_reveals_and_restores_hidden_docks(
    make_napari_viewer,
):
    viewer = make_napari_viewer(show=True)
    qt_viewer = viewer.window._qt_viewer
    qt_viewer.dockLayerList.hide()
    qt_viewer.dockLayerControls.hide()

    tour = build_viewer_tour(viewer.window, sample=None)
    steps = {step.title: step for step in tour._steps}

    assert steps['Layer list'].ensure_visible()
    assert qt_viewer.dockLayerList.isVisible()
    assert steps['Layer controls'].ensure_visible()
    assert qt_viewer.dockLayerControls.isVisible()

    tour.finished.emit()
    assert not qt_viewer.dockLayerList.isVisible()
    assert not qt_viewer.dockLayerControls.isVisible()


def test_build_viewer_tour_leaves_visible_docks_alone(make_napari_viewer):
    viewer = make_napari_viewer(show=True)
    qt_viewer = viewer.window._qt_viewer
    assert qt_viewer.dockLayerList.isVisible()
    assert qt_viewer.dockLayerControls.isVisible()

    tour = build_viewer_tour(viewer.window, sample=None)
    tour.finished.emit()
    assert qt_viewer.dockLayerList.isVisible()
    assert qt_viewer.dockLayerControls.isVisible()


@pytest.mark.parametrize(
    ('shape', 'expect_dims_step'),
    [((4, 4), False), ((4, 4, 4), True)],
)
def test_tour_skips_dims_step_without_extra_dims(
    make_napari_viewer, shape, expect_dims_step
):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros(shape))
    tour = build_viewer_tour(viewer.window)

    titles = []
    index = tour._seek(0, 1)
    while index is not None:
        titles.append(tour._steps[index].title)
        index = tour._seek(index + 1, 1)

    assert ('Dimension sliders' in titles) is expect_dims_step


def test_resolve_tour_target_builtins(make_napari_viewer):
    viewer = make_napari_viewer(show=True)

    for name in _BUILTIN_TOUR_TARGETS:
        assert resolve_tour_target(viewer.window, name) is not None

    assert resolve_tour_target(viewer.window, 'nonexistent') is None


def test_resolve_tour_target_plugin_dock_widget(make_napari_viewer):
    viewer = make_napari_viewer(show=True)

    label = QLabel('hello')
    viewer.window.add_dock_widget(label, name='My Plugin Widget')

    assert resolve_tour_target(viewer.window, 'My Plugin Widget') is label


def test_take_viewer_tour():
    tour = mock.Mock()
    qt_window = SimpleNamespace(_viewer_tour=None)
    window = SimpleNamespace(_qt_window=qt_window)

    with mock.patch(
        'napari._qt._qapp_model.qactions._help.build_viewer_tour',
        return_value=tour,
    ) as mock_build_tour:
        _take_viewer_tour(window)
        _take_viewer_tour(window)

    mock_build_tour.assert_called_once_with(qt_window)
    tour.finished.connect.assert_called_once()
    tour.start.assert_called_once()
    assert qt_window._viewer_tour is tour

    tour.finished.connect.call_args.args[0]()
    assert qt_window._viewer_tour is None


@pytest.mark.parametrize(
    ('has_layers', 'pass_sample', 'expect_loaded'),
    [
        (False, True, True),  # no layers, sample given -> loads it
        (True, True, False),  # layers already present -> skips loading
        (False, False, False),  # sample=None -> never loads
    ],
)
def test_build_viewer_tour_sample(
    make_napari_viewer, tmp_plugin, has_layers, pass_sample, expect_loaded
):
    @tmp_plugin.contribute.sample_data(key='fake')
    def _generate_fake_data():
        return [(np.zeros((4, 4, 4)), {'name': 'fake'})]

    viewer = make_napari_viewer()
    if has_layers:
        viewer.add_image(np.zeros((4, 4)))
    n_layers_before = len(viewer.layers)

    sample = (tmp_plugin.name, 'fake') if pass_sample else None
    build_viewer_tour(viewer.window, sample=sample)

    n_new_layers = len(viewer.layers) - n_layers_before
    assert (n_new_layers > 0) is expect_loaded
