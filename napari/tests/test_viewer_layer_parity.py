"""
Ensure that layers and their convenience methods on the viewer
have the same signatures and docstrings.
"""

import inspect

import pytest

from napari import layers as module, Viewer


layers = []

for name in dir(module):
    obj = getattr(module, name)

    if obj is module.Layer or not inspect.isclass(obj):
        continue

    if issubclass(obj, module.Layer):
        layers.append(obj)


@pytest.mark.parametrize('layer', layers, ids=lambda layer: layer.__name__)
def test_docstring(layer):
    method = f'add_{layer.__name__.lower()}'
    layer_doc = layer.__doc__
    i = [
        idx
        for idx, section in enumerate(layer.__doc__.split('\n\n'))
        if section.strip().startswith('Parameters')
    ][0]
    # We only check the parameters section, which is the first
    # section after separating by empty line. We also strip empty
    # line and white space that occurs at the end of a class docstring
    layer_param = layer_doc.split('\n\n')[i].rstrip('\n ')
    method_doc = getattr(Viewer, method).__doc__
    # For the method docstring, we also need to
    # remove the extra indentation of the method docstring compared
    # to the class
    method_param = method_doc.split('\n\n')[i].replace('\n    ', '\n')[4:]
    fail_msg = f"Docstrings don't match for class {layer.__name__}"
    assert layer_param == method_param, fail_msg
