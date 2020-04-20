import os
import sys
from contextlib import contextmanager

import pytest
from pluggy.hooks import HookImpl, HookimplMarker

import napari.plugins._builtins
from napari.plugins import PluginManager


@pytest.fixture
def plugin_manager():
    """PluginManager fixture that loads some test plugins"""
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures')
    plugin_manager = PluginManager(
        project_name='napari', autodiscover=fixture_path
    )
    assert fixture_path not in sys.path, 'discover path leaked into sys.path'
    return plugin_manager


@pytest.fixture
def builtin_plugin_manager(plugin_manager):
    for mod in plugin_manager.get_plugins():
        if mod != napari.plugins._builtins:
            plugin_manager.unregister(mod)
    assert plugin_manager.get_plugins() == set([napari.plugins._builtins])
    return plugin_manager


@pytest.fixture
def temporary_hookimpl(plugin_manager):
    @contextmanager
    def inner(func, specname):
        caller = getattr(plugin_manager.hook, specname)
        HookimplMarker('napari')(tryfirst=True)(func)
        impl = HookImpl(None, "<temp>", func, func.napari_impl)
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
