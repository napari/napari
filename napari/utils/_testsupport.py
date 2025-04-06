import warnings
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pytest import FixtureRequest  # noqa: PT013

    from napari._qt.qt_main_window import _QtMainWindow

_SAVE_GRAPH_OPNAME = '--save-leaked-object-graph'


def _empty(*_, **__):
    """Empty function for mocking"""


def pytest_addoption(parser):
    parser.addoption(
        '--show-napari-viewer',
        action='store_true',
        default=False,
        help="don't show viewer during tests",
    )

    parser.addoption(
        _SAVE_GRAPH_OPNAME,
        action='store_true',
        default=False,
        help="Try to save a graph of leaked object's reference (need objgraph"
        'and graphviz installed',
    )


COUNTER = 0


def fail_obj_graph(Klass):  # pragma: no cover
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

        leaked_objects_count = len(Klass._instances)

        gc.collect()
        file_path = Path(
            f'{Klass.__name__}-leak-backref-graph-{COUNTER}.pdf'
        ).absolute()
        objgraph.show_backrefs(
            list(Klass._instances),
            max_depth=20,
            filename=str(file_path),
        )

        Klass._instances.clear()

        assert file_path.exists()

        # DO not remove len, this can break as C++ obj are gone, but python objects
        # still hang around and _repr_ would crash.
        pytest.fail(
            f'Test run fail with leaked {leaked_objects_count} instances of {Klass}.'
            f'The object graph is saved in {file_path}.'
            f'{len(Klass._instances)} objects left after cleanup'
        )


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
    with suppress(AttributeError):
        monkeypatch.setattr(napari._qt.qt_main_window, 'plugin_manager', pm)
    # prevent discovery of plugins in the environment
    # you can still use `pm.register` to explicitly register something.
    pm.discovery_blocker = patch.object(pm, 'discover')
    pm.discovery_blocker.start()
    pm._initialize()  # register our builtins
    yield pm
    pm.discovery_blocker.stop()


GCPASS = 0


@pytest.fixture(autouse=True)
def _clean_themes():
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

    setattr(item, f'rep_{rep.when}', rep)


@pytest.fixture
def mock_app_model():
    """Mock clean 'test_app' `NapariApplication` instance.

    This fixture must be used whenever `napari._app_model.get_app_model()` is called to
    return a 'test_app' `NapariApplication` instead of the 'napari'
    `NapariApplication`. The `make_napari_viewer` fixture is already equipped with
    a `mock_app_model`.

    Note that `NapariApplication` registers app-model actions.
    If this is not desired, please create a clean
    `app_model.Application` in the test.

    It does not register Qt related actions, providers or processors, which is done
    via `init_qactions()`. Nor does it register plugins, done via `_initialize_plugins`.
    If these are required, the `make_napari_viewer` fixture can be used, which
    will register ALL actions, providers and processors and register plugins.
    It will also automatically clear the lru cache.

    Alternatively, you can specifically run `init_qactions()` or
    `_initialize_plugins` within the test, ensuring that you `cache_clear()`
    first.
    """
    from app_model import Application

    from napari._app_model._app import NapariApplication, _napari_names

    app = NapariApplication('test_app')
    app.injection_store.namespace = _napari_names
    with patch.object(NapariApplication, 'get_app_model', return_value=app):
        try:
            yield app
        finally:
            Application.destroy('test_app')


def clean_viewer_references(main_window: '_QtMainWindow') -> None:
    """Clean up references to the viewer in the viewer._instances set."""
    from napari.settings import get_settings

    settings = get_settings()
    settings.reset()
    settings.application.save_window_geometry = False
    settings.application.save_window_state = False

    viewer = main_window._qt_viewer.viewer
    with suppress(KeyError):
        viewer._instances.remove(viewer)
        viewer.window._qt_window._instances.remove(viewer.window._qt_window)


@pytest.fixture
def make_napari_viewer(
    qtbot,
    request: 'FixtureRequest',
    mock_app_model,
    napari_plugin_manager,
    monkeypatch,
):
    """A pytest fixture function that creates a napari viewer for use in testing.

    This fixture will take care of creating a viewer and cleaning up at the end of the
    test. When using this function, it is **not** necessary to use a `qtbot` fixture,
    nor should you do any additional cleanup (such as using `qtbot.addWidget` or
    calling `viewer.close()`) at the end of the test. Duplicate cleanup may cause
    an error.

    To use this fixture as a function in your tests:

        def test_something_with_a_viewer(make_napari_viewer):
            # `make_napari_viewer` takes any keyword arguments that napari.Viewer() takes
            viewer = make_napari_viewer()

    It accepts all the same arguments as `napari.Viewer`, notably `show`
    which should be set to `True` for tests that require the `Viewer` to be visible
    (e.g., tests that check aspects of the Qt window or layer rendering).
    It also accepts the following test-related parameters:

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
    from qtpy.QtWidgets import QWidget

    from napari import Viewer
    from napari._qt._qapp_model.qactions import init_qactions
    from napari.plugins import _initialize_plugins
    from napari.settings import get_settings

    settings = get_settings()
    settings.reset()
    settings.application.save_window_geometry = False
    settings.application.save_window_state = False

    _initialize_plugins.cache_clear()
    init_qactions.cache_clear()

    # may be overridden by using the parameter `strict_qt`

    # disable thread for status checker
    monkeypatch.setattr(
        'napari._qt.threads.status_checker.StatusChecker.start',
        _empty,
    )

    if 'enable_console' not in request.keywords:

        def _dummy_widget(*_):
            w = QWidget()
            w._update_theme = _empty
            return w

        monkeypatch.setattr(
            'napari._qt.qt_viewer.QtViewer._get_console', _dummy_widget
        )

    def actual_factory(
        *model_args,
        ViewerClass=Viewer,
        strict_qt=None,
        block_plugin_discovery=True,
        use_qtbot=True,
        **model_kwargs,
    ):
        if strict_qt is not None:
            warnings.warn(
                'strict_qt is deprecated, we use qtbot to manage qt cleanup',
                DeprecationWarning,
                stacklevel=2,
            )
        if not block_plugin_discovery:
            napari_plugin_manager.discovery_blocker.stop()

        should_show = request.config.getoption('--show-napari-viewer')
        model_kwargs['show'] = model_kwargs.pop('show', should_show)
        viewer = ViewerClass(*model_args, **model_kwargs)

        if use_qtbot:
            qtbot.add_widget(
                viewer.window._qt_window,
                before_close_func=clean_viewer_references,
            )

        return viewer

    yield actual_factory

    # Some tests might have the viewer closed, so this call will not be able
    # to access the window.
    with suppress(AttributeError):
        settings = get_settings()
        settings.reset()


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
            'NAPARI_ENSURE_PLUGIN_MAIN_THREAD', str(ensure_main_thread)
        )
        viewer = make_napari_viewer(*model_args, **model_kwargs)
        proxies.append(PublicOnlyProxy(viewer))
        return proxies[-1]

    proxies.clear()

    return actual_factory


@pytest.fixture
def MouseEvent():
    """Create a subclass for simulating vispy mouse events.

    Returns
    -------
    Event : Type
        A new dataclass named Event that can be used to create an
        object with fields "type" and "is_dragging".
    """

    @dataclass
    class Event:
        type: str
        position: tuple[float]
        is_dragging: bool = False
        dims_displayed: tuple[int] = (0, 1)
        dims_point: list[float] = None
        view_direction: list[int] = None
        pos: list[int] = (0, 0)
        button: int = None
        handled: bool = False

    return Event
