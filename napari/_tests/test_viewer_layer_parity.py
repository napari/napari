"""
Ensure that layers and their convenience methods on the viewer
have the same signatures and docstrings.
"""

import inspect
import re

import pytest
from numpydoc.docscrape import FunctionDoc, ClassDoc

from napari import layers as module, Viewer
from napari.utils.misc import camel_to_snake, callsignature


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

    # Remove path parameter from viewer method if it exists
    method_params = [m for m in method_params if m.name != 'path']

    if name == 'Image':
        # For Image just test arguments that are in layer are in method
        named_method_params = [m.name for m in method_params]
        for layer_param in layer_params:
            l_name, l_type, l_description = layer_param
            assert l_name in named_method_params
    else:
        try:
            assert len(method_params) == len(layer_params)
            for method_param, layer_param in zip(method_params, layer_params):
                m_name, m_type, m_description = method_param
                l_name, l_type, l_description = layer_param

                # descriptions are treated as lists where each line is an
                # element
                m_description = ' '.join(m_description)
                l_description = ' '.join(l_description)

                assert m_name == l_name, 'different parameter names or order'
                assert (
                    m_type == l_type
                ), f"type mismatch of parameter '{m_name}'"
                assert (
                    m_description == l_description
                ), f"description mismatch of parameter '{m_name}'"
        except AssertionError as e:
            raise AssertionError(
                f"docstrings don't match for class {name}"
            ) from e

    # check returns section
    (method_returns,) = method_doc[
        'Returns'
    ]  # only one thing should be returned
    description = ' '.join(method_returns[-1])  # join multi-line description
    method_returns = *method_returns[:-1], description

    if name == 'Image':
        assert method_returns == (
            'layer',
            f':class:`napari.layers.{name}` or list',
            f'The newly-created {name.lower()} layer or list of {name.lower()} layers.',  # noqa: E501
        ), f"improper 'Returns' section of '{method_name}'"
    else:
        assert method_returns == (
            'layer',
            f':class:`napari.layers.{name}`',
            f'The newly-created {name.lower()} layer.',
        ), f"improper 'Returns' section of '{method_name}'"


@pytest.mark.parametrize('layer', layers, ids=lambda layer: layer.__name__)
def test_signature(layer):
    name = layer.__name__

    method = getattr(Viewer, f'add_{camel_to_snake(name)}')

    class_parameters = dict(inspect.signature(layer.__init__).parameters)
    method_parameters = dict(inspect.signature(method).parameters)

    # Remove path and data parameters from viewer method if path exists
    if 'path' in method_parameters:
        del method_parameters['path']
        del method_parameters['data']
        del class_parameters['data']

    fail_msg = f"signatures don't match for class {name}"
    if name == 'Image':
        # If Image just test that class params appear in method
        for class_param in class_parameters.keys():
            assert class_param in method_parameters.keys(), fail_msg
    else:
        assert class_parameters == method_parameters, fail_msg

    code = inspect.getsource(method)

    args = re.search(rf'layer = layers\.{name}\((.+?)\)', code, flags=re.S)
    # get the arguments & normalize whitepsace
    args = ' '.join(args.group(1).split())

    if args.endswith(','):  # remove tailing comma if present
        args = args[:-1]

    autogen = callsignature(layer)
    autogen = autogen.replace(
        # remove 'self' parameter
        parameters=[p for k, p in autogen.parameters.items() if k != 'self']
    )
    autogen = str(autogen)[1:-1]  # remove parentheses

    try:
        assert args == autogen
    except AssertionError as e:
        msg = f'arguments improperly passed from convenience method to layer {name}'  # noqa: E501
        raise SyntaxError(msg) from e
