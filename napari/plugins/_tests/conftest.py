import os
import sys
from contextlib import contextmanager

import pytest
from naplugi import HookImpl, HookimplMarker, PluginManager

from napari.plugins import hook_specifications, _builtins


@pytest.fixture
def plugin_manager():
    """PluginManager fixture that loads some test plugins"""
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures')
    plugin_manager = PluginManager(
        project_name='napari', autodiscover=fixture_path
    )
    plugin_manager.add_hookspecs(hook_specifications)
    plugin_manager.register(_builtins, name='builtins')
    assert fixture_path not in sys.path, 'discover path leaked into sys.path'
    return plugin_manager


@pytest.fixture
def builtin_plugin_manager(plugin_manager):
    for mod in plugin_manager.get_plugins():
        if mod != _builtins:
            plugin_manager.unregister(mod)
    assert plugin_manager.get_plugins() == set([_builtins])
    return plugin_manager


@pytest.fixture
def temporary_hookimpl(plugin_manager):
    """A fixture that can be used to insert a HookImpl in the hook call loop.

    Example
    -------

    .. code-block: python

        def bad_write_points(path, data, meta):
            raise ValueError("shoot!")

        with temporary_hookimpl(bad_write_points, 'napari_write_points'):
            with pytest.raises(PluginCallError):
                writer(tmpdir, layer_data, plugin_manager)
    """

    @contextmanager
    def inner(
        func, specname, tryfirst=True, trylast=None, plugin_name="<temp>"
    ):
        caller = getattr(plugin_manager.hook, specname)
        HookimplMarker('napari')(tryfirst=tryfirst, trylast=trylast)(func)
        impl = HookImpl(None, plugin_name, func, func.napari_impl)
        caller._add_hookimpl(impl)
        try:
            yield
        finally:
            if impl in caller._nonwrappers:
                caller._nonwrappers.remove(impl)
            if impl in caller._wrappers:
                caller._wrappers.remove(impl)
            assert impl not in caller.get_hookimpls()

    return inner
