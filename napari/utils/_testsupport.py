import os
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
def napari_plugin_manager(monkeypatch):
    """A napari plugin manager that blocks discovery by default.

    Note you can still use `napari_plugin_manager.register()` to directly
    register a plugin module, class or dict for testing.

    Or, to re-enable global discovery, use:
    `napari_plugin_manager.discovery_blocker.stop()`
    """
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
    pm.discovery_blocker = patch.object(pm, 'discover')
    pm.discovery_blocker.start()
    pm._initialize()  # register our builtins
    yield pm
    pm.discovery_blocker.stop()


@pytest.fixture
def make_napari_viewer(
    qtbot, request: 'FixtureRequest', napari_plugin_manager
):
    """A fixture function that creates a napari viewer for use in testing.

    Use this fixture as a function in your tests:

        viewer = make_napari_viewer()

    It accepts all the same arguments as napari.Viewer, plus the following
    test-related paramaters:

    ViewerClass : Type[napari.Viewer], optional
        Override the viewer class being used.  By default, will
        use napari.viewer.Viewer
    strict_qt : bool or str, optional
        If True, a check will be performed after test cleanup to make sure that
        no top level widgets were created and *not* cleaned up during the
        test.  If the string "raise" is provided, an AssertionError will be
        raised.  Otherwise a warning is emitted.
        By default, this is False unless the test is being performed within
        the napari package.
        This can be made globally true by setting the 'NAPARI_STRICT_QT'
        environment variable.
    block_plugin_discovery : bool, optional
        Block discovery of non-builtin plugins.  Note: plugins can still be
        manually registered by using the 'napari_plugin_manager' fixture and
        the `napari_plugin_manager.register()` method. By default, True.

    Examples
    --------
    >>> def test_adding_shapes(make_napari_viewer):
    ...     viewer = make_napari_viewer()
    ...     viewer.add_shapes()
    ...     assert len(viewer.layers) == 1

    >>> def test_something_with_plugins(make_napari_viewer):
    ...     viewer = make_napari_viewer(block_plugin_discovery=False)

    >>> def test_something_with_strict_qt_tests(make_napari_viewer):
    ...     viewer = make_napari_viewer(strict_qt=True)
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
        strict_qt=is_internal_test or os.getenv("NAPARI_STRICT_QT"),
        block_plugin_discovery=True,
        **model_kwargs,
    ):
        nonlocal _strict
        _strict = strict_qt

        if not block_plugin_discovery:
            napari_plugin_manager.discovery_blocker.stop()

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
            if _strict == 'raise':
                raise AssertionError(f'Widgets leaked!: {leak}')
            else:
                warnings.warn(f'Widgets leaked!: {leak}')
