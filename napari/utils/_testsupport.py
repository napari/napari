import sys
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING, List
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pytest import FixtureRequest


def pytest_addoption(parser):
    parser.addoption(
        "--show-napari-viewer",
        action="store_true",
        default=False,
        help="don't show viewer during tests",
    )


@pytest.fixture
def test_napari_plugin_manager(monkeypatch):
    from unittest.mock import patch

    import napari
    from napari.plugins._plugin_manager import NapariPluginManager

    pm = NapariPluginManager()

    # make it so that internal requests for the plugin_manager
    # get this test version for the duration of the test.
    monkeypatch.setattr(napari.plugins, 'plugin_manager', pm)
    monkeypatch.setattr(napari.plugins.io, 'plugin_manager', pm)
    try:
        monkeypatch.setattr(napari._qt.qt_main_window, 'plugin_manager', pm)
    except AttributeError:  # headless tests
        pass

    # prevent discovery of plugins in the environment
    # you can still use `pm.register` to explicitly register something.
    pm._discover_patcher = patch.object(pm, 'discover')
    pm._discover_patcher.start()
    pm._initialize()  # register our builtins
    yield pm
    pm._discover_patcher.stop()


@pytest.fixture
def make_napari_viewer(
    qtbot, request: 'FixtureRequest', test_napari_plugin_manager
):
    """A fixture function that creates a napari viewer for use in testing.

    This uses a strict qtbot variant that asserts that no widgets are left over
    after the viewer is closed.

    Examples
    --------
    >>> def test_adding_shapes(make_napari_viewer):
    ...     viewer = make_napari_viewer()
    ...     viewer.add_shapes()
    ...     assert len(viewer.layers) == 1
    """
    from qtpy.QtWidgets import QApplication

    from napari import Viewer
    from napari.utils.settings import SETTINGS

    SETTINGS.reset()
    viewers: List[Viewer] = []

    # may be overridden by using `make_napari_viewer(strict=True)`
    _strict = False

    initial = QApplication.topLevelWidgets()
    prior_exception = getattr(sys, 'last_value', None)
    is_internal_test = request.module.__name__.startswith("napari.")

    def actual_factory(
        *model_args,
        ViewerClass=Viewer,
        strict=is_internal_test,
        block_plugin_discovery=True,
        **model_kwargs,
    ):
        nonlocal _strict
        _strict = strict

        if not block_plugin_discovery:
            test_napari_plugin_manager._discover_patcher.stop()

        should_show = request.config.getoption("--show-napari-viewer")
        model_kwargs['show'] = model_kwargs.pop('show', should_show)
        viewer = ViewerClass(*model_args, **model_kwargs)
        viewers.append(viewer)

        return viewer

    yield actual_factory

    # Some tests might have the viewer closed, so this call will not be able
    # to access the window.
    with suppress(AttributeError):
        SETTINGS.reset()

    # close viewers, but don't saving window settings while closing
    for viewer in viewers:
        if hasattr(viewer.window, '_qt_window'):
            with patch.object(
                viewer.window._qt_window, '_save_current_window_settings'
            ):
                viewer.close()
        else:
            viewer.close()

    # only check for leaked widgets if an exception was raised during the test,
    # or "strict" mode was used.
    if _strict and getattr(sys, 'last_value', None) is prior_exception:
        QApplication.processEvents()
        leak = set(QApplication.topLevelWidgets()).difference(initial)
        # still not sure how to clean up some of the remaining vispy
        # vispy.app.backends._qt.CanvasBackendDesktop widgets...
        if any([n.__class__.__name__ != 'CanvasBackendDesktop' for n in leak]):
            # just a warning... but this can be converted to test errors
            # in pytest with `-W error`
            warnings.warn(f'Widgets leaked!: {leak}')
