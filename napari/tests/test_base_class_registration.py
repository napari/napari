import inspect

import pytest

from napari import layers as module
from napari.util.layer_view import _baseclass_layer_to_subclass

from napari._qt.layers import QtLayerControls, QtLayerProperties
from napari._vispy import VispyBaseLayer


layers = []

for name in dir(module):
    obj = getattr(module, name)

    if obj is module.Layer or not inspect.isclass(obj):
        continue

    if issubclass(obj, module.Layer):
        layers.append(obj)


def name(obj):
    return obj.__name__


def qt_subclass_name_pattern(base, layer):
    return base.replace('Layer', layer)


def vispy_subclass_name_pattern(base, layer):
    return base.replace('Base', layer)


@pytest.mark.parametrize(
    'base_class,pattern',
    [
        (VispyBaseLayer, vispy_subclass_name_pattern),
        (QtLayerControls, qt_subclass_name_pattern),
        (QtLayerProperties, qt_subclass_name_pattern),
    ],
    ids=name,
)
@pytest.mark.parametrize('layer', layers, ids=name)
def test_properly_registered(base_class, pattern, layer):
    subclass = _baseclass_layer_to_subclass[base_class][layer]
    subclass_name = pattern(name(base_class), name(layer))

    assert name(subclass) == subclass_name
