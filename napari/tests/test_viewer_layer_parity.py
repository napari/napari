"""
Ensure that layers and their convenience methods on the viewer
have the same signatures and docstrings.
"""

import inspect
import re

import pytest
from numpydoc.docscrape import FunctionDoc, ClassDoc

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
    name = layer.__name__

    method_name = f'add_{camel_to_snake(name)}'
    method = getattr(Viewer, method_name)

    method_doc = FunctionDoc(method)
    layer_doc = ClassDoc(layer)

    # check summary section
    method_summary = ' '.join(method_doc['Summary'])  # join multi-line summary

    summary_format = 'Add an? .+? layer to the layers list.'

    assert re.match(
        summary_format, method_summary
    ), f"improper 'Summary' section of '{method_name}'"

    # check parameters section
    method_params = method_doc['Parameters']
    layer_params = layer_doc['Parameters']

    try:
        assert len(method_params) == len(layer_params)
        for method_param, layer_param in zip(method_params, layer_params):
            m_name, m_type, m_description = method_param
            l_name, l_type, l_description = layer_param

            # descriptions are treated as lists where each line is an element
            m_description = ' '.join(m_description)
            l_description = ' '.join(l_description)

            assert m_name == l_name, 'different parameter names or order'
            assert m_type == l_type, f"type mismatch of parameter '{m_name}'"
            assert (
                m_description == l_description
            ), f"description mismatch of parameter '{m_name}'"
    except AssertionError as e:
        raise AssertionError(f"docstrings don't match for class {name}") from e

    # check returns section
    method_returns, = method_doc[
        'Returns'
    ]  # only one thing should be returned
    description = ' '.join(method_returns[-1])  # join multi-line description
    method_returns = *method_returns[:-1], description

    assert method_returns == (
        'layer',
        f':class:`napari.layers.{name}`',
        f'The newly-created {name.lower()} layer.',
    ), f"improper 'Returns' section of '{method_name}'"


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
