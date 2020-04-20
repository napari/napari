import os
import sys
from contextlib import contextmanager

import numpy as np
import pytest
from pluggy.hooks import HookImpl, HookimplMarker

import napari.plugins._builtins
from napari.layers import Image, Points
from napari.plugins import PluginManager
from napari.plugins._builtins import napari_write_image, napari_write_points
from napari.utils import io


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


@pytest.fixture(params=['image', 'points', 'points-with-properties'])
def layer_writer_and_data(request):
    if request.param == 'image':
        data = np.random.rand(20, 20)
        Layer = Image
        layer = Image(data)
        writer = napari_write_image
        extension = '.tif'

        def reader(path):
            return (
                io.imread(path),
                {},
            )

    elif request.param == 'points':
        data = np.random.rand(20, 2)
        Layer = Points
        layer = Points(data)
        writer = napari_write_points
        extension = '.csv'

        def reader(path):
            return (
                io.read_csv(path)[0][:, 1:3],
                {},
            )

    elif request.param == 'points-with-properties':
        data = np.random.rand(20, 2)
        Layer = Points
        layer = Points(data, properties={'values': np.random.rand(20)})
        writer = napari_write_points
        extension = '.csv'

        def reader(path):
            return (
                io.read_csv(path)[0][:, 1:3],
                {
                    'properties': {
                        io.read_csv(path)[1][3]: io.read_csv(path)[0][:, 3]
                    }
                },
            )

    else:
        return None, None, None, None, None

    layer_data = layer.as_layer_data_tuple()
    return writer, layer_data, extension, reader, Layer


@pytest.fixture
def layer_data_and_types():
    layers = [
        Image(np.random.rand(20, 20), name='ex_img'),
        Image(np.random.rand(20, 20)),
        Points(np.random.rand(20, 2), name='ex_pts'),
        Points(
            np.random.rand(20, 2), properties={'values': np.random.rand(20)}
        ),
    ]
    extensions = ['.tif', '.tif', '.csv', '.csv']
    layer_data = [l.as_layer_data_tuple() for l in layers]
    layer_types = [ld[2] for ld in layer_data]
    filenames = [l.name + e for l, e in zip(layers, extensions)]
    return layers, layer_data, layer_types, filenames
