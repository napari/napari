"""
Ensure that layers and their convenience methods on the viewer
have the same signatures and docstrings.
"""

import gc
import inspect
import re

import numpy as np
import pytest
from docstring_parser.numpydoc import parse as parse_docstring

import napari
from napari import Viewer, layers as module
from napari._tests.utils import check_viewer_functioning
from napari.utils.misc import camel_to_snake

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

    method_doc = parse_docstring(method.__doc__ or '')
    layer_doc = parse_docstring(layer.__doc__ or '')

    # check summary section
    method_summary = method_doc.short_description or ''

    if name == 'Image':
        summary_format = 'Add one or more Image layers to the layer list.'
    else:
        summary_format = 'Add an? .+? layers? to the layer list.'

    assert re.match(summary_format, method_summary), (
        f"improper 'Summary' section of '{method_name}'"
    )

    # check parameters section
    method_params = method_doc.params
    layer_params = layer_doc.params

    # Remove path parameter from viewer method if it exists
    method_params = [m for m in method_params if m.arg_name != 'path']

    if name == 'Image':
        # For Image just test arguments that are in layer are in method
        # Filter layer params to only those that are actually in the method
        # (layer docstrings may include attributes that aren't constructor params)
        method_param_names = [m.arg_name for m in method_params]
        layer_params = [
            p for p in layer_params if p.arg_name in method_param_names
        ]
        for layer_param in layer_params:
            assert layer_param.arg_name in method_param_names
    else:
        # For other layers, ensure all method params are documented in layer
        # (We don't check the reverse because layer docstrings may include
        # attributes that aren't constructor params)
        method_param_names = {m.arg_name for m in method_params}
        layer_param_dict = {p.arg_name: p for p in layer_params}

        # Check that all method params exist in layer docs
        for method_param in method_params:
            assert method_param.arg_name in layer_param_dict, (
                f"Parameter '{method_param.arg_name}' in {method_name} "
                f'not found in {name} docstring'
            )

        try:
            for method_param in method_params:
                layer_param = layer_param_dict[method_param.arg_name]

                m_name = method_param.arg_name
                l_name = layer_param.arg_name

                assert m_name == l_name, 'different parameter names'
                # Note: We don't check types or descriptions because they may
                # reasonably differ between the layer class and viewer method
                # perspectives (e.g., method may accept more types for convenience)
        except AssertionError as e:
            raise AssertionError(
                f"docstrings don't match for class {name}"
            ) from e

    # check returns section
    method_returns = method_doc.returns

    if method_returns:
        return_name = method_returns.return_name or 'layer'
        return_type = method_returns.type_name or ''
        return_description = (method_returns.description or '').strip()

        if name == 'Image':
            expected_type = f':class:`napari.layers.{name}` or list'
            expected_desc = f'The newly-created {name.lower()} layer or list of {name.lower()} layers.'
        else:
            expected_type = f':class:`napari.layers.{name}`'
            expected_desc = f'The newly-created {name.lower()} layer.'

        assert return_name == 'layer', (
            f"improper return name in '{method_name}'"
        )
        assert return_type == expected_type, (
            f"improper return type in '{method_name}'"
        )
        assert return_description == expected_desc, (
            f"improper return description in '{method_name}'"
        )


@pytest.mark.parametrize('layer', layers, ids=lambda layer: layer.__name__)
def test_signature(layer):
    name = layer.__name__

    method = getattr(Viewer, f'add_{camel_to_snake(name)}')

    class_parameters = dict(inspect.signature(layer.__init__).parameters)
    method_parameters = dict(inspect.signature(method).parameters)

    fail_msg = f"signatures don't match for class {name}"
    if name == 'Image':
        # If Image just test that class params appear in method
        for class_param in class_parameters:
            assert class_param in method_parameters, fail_msg
    else:
        assert class_parameters == method_parameters, fail_msg


# plugin_manager fixture is added to prevent errors due to installed plugins
def test_imshow(qtbot, napari_plugin_manager):
    shape = (10, 15)
    ndim = len(shape)
    np.random.seed(0)
    data = np.random.random(shape)
    viewer, layer = napari.imshow(data, channel_axis=None, show=False)
    view = viewer.window._qt_viewer
    check_viewer_functioning(viewer, view, data, ndim)
    assert isinstance(layer, napari.layers.Image)
    viewer.close()


# plugin_manager fixture is added to prevent errors due to installed plugins
def test_imshow_multichannel(qtbot, napari_plugin_manager):
    """Test adding image."""
    np.random.seed(0)
    data = np.random.random((15, 10, 5))
    viewer, layers = napari.imshow(data, channel_axis=-1, show=False)
    assert len(layers) == data.shape[-1]
    assert isinstance(layers, tuple)
    for i in range(data.shape[-1]):
        np.testing.assert_array_equal(layers[i].data, data.take(i, axis=-1))
    viewer.close()
    # Run a full garbage collection here so that any remaining viewer
    # and related instances are removed for future tests that may use
    # make_napari_viewer.
    gc.collect()


# plugin_manager fixture is added to prevent errors due to installed plugins
def test_imshow_with_viewer(qtbot, napari_plugin_manager, make_napari_viewer):
    shape = (10, 15)
    ndim = len(shape)
    np.random.seed(0)
    data = np.random.random(shape).astype(np.float32)
    viewer = make_napari_viewer()
    viewer2, layer = napari.imshow(data, viewer=viewer, show=False)
    assert viewer is viewer2
    np.testing.assert_array_equal(data, layer.data)
    view = viewer.window._qt_viewer
    check_viewer_functioning(viewer, view, data, ndim)
    viewer.close()
