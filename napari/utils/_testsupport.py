import sys
import warnings
from typing import List

import pytest
from napari_plugin_engine import PluginManager


def pytest_addoption(parser):
    parser.addoption(
        "--show-napari-viewer",
        action="store_true",
        default=False,
        help="don't show viewer during tests",
    )


@pytest.fixture
def _strict_qtbot(qtbot):
    """A modified qtbot fixture that makes sure no widgets have been leaked."""
    from qtpy.QtWidgets import QApplication

    initial = QApplication.topLevelWidgets()
    prior_exception = getattr(sys, 'last_value', None)

    yield qtbot

    # if an exception was raised during the test, we should just quit now and
    # skip looking for leaked widgets.
    if getattr(sys, 'last_value', None) is not prior_exception:
        return

    QApplication.processEvents()
    leaks = set(QApplication.topLevelWidgets()).difference(initial)
    # still not sure how to clean up some of the remaining vispy
    # vispy.app.backends._qt.CanvasBackendDesktop widgets...
    if any([n.__class__.__name__ != 'CanvasBackendDesktop' for n in leaks]):
        raise AssertionError(f'Widgets leaked!: {leaks}')
    if leaks:
        warnings.warn(f'Widgets leaked!: {leaks}')


@pytest.fixture
def make_napari_viewer(_strict_qtbot, request):
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
    from napari import Viewer

    viewers: List[Viewer] = []

    def actual_factory(*model_args, viewer_class=Viewer, **model_kwargs):
        model_kwargs['show'] = model_kwargs.pop(
            'show', request.config.getoption("--show-napari-viewer")
        )
        viewer = viewer_class(*model_args, **model_kwargs)
        viewers.append(viewer)
        return viewer

    yield actual_factory

    for viewer in viewers:
        viewer.close()


class _TestPluginManager(PluginManager):
    def assert_plugin_name_registered(self, plugin_name):
        assert plugin_name in self.plugins

    def assert_module_registered(self, module):
        if isinstance(module, str):
            assert module in {m.__name__ for m in self.plugins.values()}
        else:
            assert module in self.plugins.values()

    def assert_implementations_registered(self, plugin, hook_names=()):
        from typing import Collection

        plugin = self._ensure_plugin(plugin)
        regnames = {hook.name for hook in self._plugin2hookcallers[plugin]}
        _hook_names = (
            hook_names
            if isinstance(hook_names, Collection)
            and not isinstance(hook_names, str)
            else [hook_names]
        )
        if not _hook_names:
            if not regnames:
                raise AssertionError(
                    f"No implementations were registered for plugin {plugin!r}"
                )
        else:
            for hook in _hook_names:
                if hook not in regnames:
                    raise AssertionError(
                        f"{hook!r} was not registered for plugin {plugin!r}"
                    )


@pytest.fixture
def napari_plugin_tester():
    """A fixture that can be used to test plugin registration.

    See _TestPluginManager above for tests implementations:
    Examples
    --------
    >>> def test_pm(napari_plugin_tester):
    >>> napari_plugin_tester.assert_plugin_name_registered("test-plugin")
    >>> napari_plugin_tester.assert_module_registered(_test)
    >>> napari_plugin_tester.assert_implementations_registered(
    >>>     "test-plugin", "napari_get_reader"
    >>> )
    """
    from napari.plugins import hook_specifications

    pm = _TestPluginManager('napari', discover_entry_point='napari.plugin')
    pm.add_hookspecs(hook_specifications)
    pm.discover()
    return pm
