"""
Ensure that layers and their convenience methods on the viewer
have the same signatures and docstrings.
"""

import inspect

import pytest

from napari import layers as module, Viewer
from napari.util.misc import camel_to_snake


layers = []

for name in dir(module):
    obj = getattr(module, name)

    if obj is module.Layer or not inspect.isclass(obj):
        continue

    if issubclass(obj, module.Layer):
        layers.append(obj)


@pytest.mark.parametrize('layer', layers, ids=lambda layer: layer.__name__)
def test_docstring(layer):
    method = getattr(Viewer, f'add_{camel_to_snake(layer.__name__)}')
    layer_doc = inspect.getdoc(layer)

    i = [
        idx
        for idx, section in enumerate(layer_doc.split('\n\n'))
        if section.startswith('Parameters')
    ][0]

    # 'inspect.getdoc' automatically removes common indentation
    method_doc = inspect.getdoc(method)

    # We only check the parameters section, which is the first
    # section after separating by empty line. We also strip empty
    # line and white space that occurs at the end of a class docstring
    layer_param = layer_doc.split('\n\n')[i].rstrip('\n ')
    method_param = method_doc.split('\n\n')[i].rstrip('\n ')

    fail_msg = f"Docstrings don't match for class {layer.__name__}"
    assert layer_param == method_param, fail_msg
