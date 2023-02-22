import collections
import gc
import os
import sys
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING
from unittest.mock import patch
from weakref import WeakSet

import pytest

if TYPE_CHECKING:
    from pytest import FixtureRequest

_SAVE_GRAPH_OPNAME = "--save-leaked-object-graph"


def _empty(*_, **__):
    """Empty function for mocking"""


def pytest_addoption(parser):
    parser.addoption(
        "--show-napari-viewer",
        action="store_true",
        default=False,
        help="don't show viewer during tests",
    )

    parser.addoption(
        _SAVE_GRAPH_OPNAME,
        action="store_true",
        default=False,
        help="Try to save a graph of leaked object's reference (need objgraph"
        "and graphviz installed",
    )


COUNTER = 0


def fail_obj_graph(Klass):
    """
    Fail is a given class _instances weakset is non empty and print the object graph.
    """

    try:
        import objgraph
    except ModuleNotFoundError:
        return

    if len(Klass._instances) != 0:
        global COUNTER
        COUNTER += 1
        import gc

        gc.collect()

        objgraph.show_backrefs(
            list(Klass._instances),
            max_depth=20,
            filename=f'{Klass.__name__}-leak-backref-graph-{COUNTER}.pdf',
        )

        # DO not remove len, this can break as C++ obj are gone, but python objects
        # still hang around and _repr_ would crash.
        assert False, len(Klass._instances)


@pytest.fixture
def napari_plugin_manager(monkeypatch):
    """A napari plugin manager that blocks discovery by default.

    Note you can still use `napari_plugin_manager.register()` to directly
    register a plugin module, class or dict for testing.

    Or, to re-enable global discovery, use:
    `napari_plugin_manager.discovery_blocker.stop()`
    """
    import napari
    import napari.plugins.io
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


GCPASS = 0


@pytest.fixture(autouse=True)
def clean_themes():
    from napari.utils import theme

    themes = set(theme.available_themes())
    yield
    for name in theme.available_themes():
        if name not in themes:
            del theme._themes[name]


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # https://docs.pytest.org/en/latest/example/simple.html#making-test-result-information-available-in-fixtures
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    # set a report attribute for each phase of a call, which can
    # be "setup", "call", "teardown"

    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture
def make_napari_viewer(
    qtbot,
    request: 'FixtureRequest',
    napari_plugin_manager,
    monkeypatch,
    clean_themes,
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
    from napari._qt.qt_viewer import QtViewer
    from napari.settings import get_settings

    global GCPASS
    GCPASS += 1

    if GCPASS % 50 == 0:
        gc.collect()
    else:
        gc.collect(1)

    _do_not_inline_below = len(QtViewer._instances)
    # # do not inline to avoid pytest trying to compute repr of expression.
    # # it fails if C++ object gone but not Python object.
    if request.config.getoption(_SAVE_GRAPH_OPNAME):
        fail_obj_graph(QtViewer)
    QtViewer._instances.clear()
    assert _do_not_inline_below == 0, (
        "Some instance of QtViewer is not properly cleaned in one of previous test. For easier debug one may "
        f"use {_SAVE_GRAPH_OPNAME} flag for pytest to get graph of leaked objects. If you use qtbot (from pytest-qt)"
        " to clean Qt objects after test you may need to switch to manual clean using "
        "`deleteLater()` and `qtbot.wait(50)` later."
    )

    settings = get_settings()
    settings.reset()

    viewers: WeakSet[Viewer] = WeakSet()

    # may be overridden by using `make_napari_viewer(strict=True)`
    _strict = False

    initial = QApplication.topLevelWidgets()
    prior_exception = getattr(sys, 'last_value', None)
    is_internal_test = request.module.__name__.startswith("napari.")

    # disable throttling cursor event in tests
    monkeypatch.setattr(
        "napari._qt.qt_main_window._QtMainWindow._throttle_cursor_to_status_connection",
        _empty,
    )

    def actual_factory(
        *model_args,
        ViewerClass=Viewer,
        strict_qt=None,
        block_plugin_discovery=True,
        **model_kwargs,
    ):
        if strict_qt is None:
            strict_qt = is_internal_test or os.getenv("NAPARI_STRICT_QT")
        nonlocal _strict
        _strict = strict_qt

        if not block_plugin_discovery:
            napari_plugin_manager.discovery_blocker.stop()

        should_show = request.config.getoption("--show-napari-viewer")
        model_kwargs['show'] = model_kwargs.pop('show', should_show)
        viewer = ViewerClass(*model_args, **model_kwargs)
        viewers.add(viewer)

        return viewer

    yield actual_factory

    # Some tests might have the viewer closed, so this call will not be able
    # to access the window.
    with suppress(AttributeError):
        get_settings().reset()

    # close viewers, but don't saving window settings while closing
    for viewer in viewers:
        if hasattr(viewer.window, '_qt_window'):
            with patch.object(
                viewer.window._qt_window, '_save_current_window_settings'
            ):
                viewer.close()
        else:
            viewer.close()

    if GCPASS % 50 == 0 or len(QtViewer._instances):
        gc.collect()
    else:
        gc.collect(1)

    if request.config.getoption(_SAVE_GRAPH_OPNAME):
        fail_obj_graph(QtViewer)

    if request.node.rep_call.failed:
        # IF test failed do not check for leaks
        QtViewer._instances.clear()

    _do_not_inline_below = len(QtViewer._instances)

    QtViewer._instances.clear()  # clear to prevent fail of next test

    # do not inline to avoid pytest trying to compute repr of expression.
    # it fails if C++ object gone but not Python object.
    assert _do_not_inline_below == 0

    # only check for leaked widgets if an exception was raised during the test,
    # or "strict" mode was used.
    if _strict and getattr(sys, 'last_value', None) is prior_exception:
        QApplication.processEvents()
        leak = set(QApplication.topLevelWidgets()).difference(initial)
        # still not sure how to clean up some of the remaining vispy
        # vispy.app.backends._qt.CanvasBackendDesktop widgets...
        if any(n.__class__.__name__ != 'CanvasBackendDesktop' for n in leak):
            # just a warning... but this can be converted to test errors
            # in pytest with `-W error`
            msg = f"""The following Widgets leaked!: {leak}.

            Note: If other tests are failing it is likely that widgets will leak
            as they will be (indirectly) attached to the tracebacks of previous failures.
            Please only consider this an error if all other tests are passing.
            """
            # Explanation notes on the above: While we are indeed looking at the
            # difference in sets of widgets between before and after, new object can
            # still not be garbage collected because of it.
            # in particular with VisPyCanvas, it looks like if a traceback keeps
            # contains the type, then instances are still attached to the type.
            # I'm not too sure why this is the case though.
            if _strict:
                raise AssertionError(msg)
            else:
                warnings.warn(msg)


@pytest.fixture
def make_napari_viewer_proxy(make_napari_viewer, monkeypatch):
    """Fixture that returns a function for creating a napari viewer wrapped in proxy.
    Use in the same way like `make_napari_viewer` fixture.

    Parameters
    ----------
    make_napari_viewer : fixture
        The make_napari_viewer fixture.

    Returns
    -------
    function
        A function that creates a napari viewer.
    """
    from napari.utils._proxies import PublicOnlyProxy

    proxies = []

    def actual_factory(*model_args, ensure_main_thread=True, **model_kwargs):
        monkeypatch.setenv(
            "NAPARI_ENSURE_PLUGIN_MAIN_THREAD", str(ensure_main_thread)
        )
        viewer = make_napari_viewer(*model_args, **model_kwargs)
        proxies.append(PublicOnlyProxy(viewer))
        return proxies[-1]

    proxies.clear()

    yield actual_factory


@pytest.fixture
def MouseEvent():
    """Create a subclass for simulating vispy mouse events.

    Returns
    -------
    Event : Type
        A new tuple subclass named Event that can be used to create a
        NamedTuple object with fields "type" and "is_dragging".
    """
    return collections.namedtuple(
        'Event',
        field_names=[
            'type',
            'is_dragging',
            'position',
            'view_direction',
            'dims_displayed',
            'dims_point',
        ],
    )
