"""
Ensure that layers and their convenience methods on the viewer
have the same signatures and docstrings.
"""

import inspect
import re

import pytest

from napari import layers as module, Viewer
from napari.util.misc import camel_to_snake
from napari.util._register import CallSignature


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


@pytest.mark.parametrize('layer', layers, ids=lambda layer: layer.__name__)
def test_signature(layer):
    method = getattr(Viewer, f'add_{camel_to_snake(layer.__name__)}')

    class_signature = inspect.signature(layer.__init__)
    method_signature = inspect.signature(method)

    fail_msg = f"Signatures don't match for class {layer.__name__}"
    assert class_signature == method_signature, fail_msg

    code = inspect.getsource(method)

    args = re.search(
        rf'layer = layers\.{layer.__name__}\((.+?)\)', code, flags=re.S
    )
    args = ' '.join(args.group(1).split())
    if args.endswith(','):
        args = args[:-1]

    autogen = CallSignature.from_callable(layer.__init__)
    autogen = autogen.replace(
        parameters=[p for k, p in autogen.parameters.items() if k != 'self']
    )
    autogen = str(autogen)[1:-1]

    assert args == autogen
