import ast

import numpy as np
import pytest

from napari.components.layerlist import LayerList
from napari.layers import Image, Labels
from napari.layers._layer_actions import (
    _LAYER_ACTIONS,
    ContextAction,
    SubMenu,
    _convert_dtype,
    _project,
)
from napari.utils.context._expressions import Expr
from napari.utils.context._layerlist_context import LayerListContextKeys


class assert_expression_variables(ast.NodeVisitor):
    def __init__(self, expression, names) -> None:
        self._variables: list[str] = []
        self.visit(ast.parse(expression))
        for i in self._variables:
            assert i in names

    def visit_Name(self, node):
        self._variables.append(node.id)


def test_layer_actions():
    """Test that all variables used in layer actions expressions are
    keys in CONTEXT_KEYS.
    """
    names = set(LayerListContextKeys.__members__)
    valid_keys = set.union(
        set(ContextAction.__annotations__), set(SubMenu.__annotations__)
    )
    for action_dict in _LAYER_ACTIONS:
        for action in action_dict.values():
            assert set(action).issubset(valid_keys)
            expr = action.get('enable_when')
            if not expr:
                continue
            assert isinstance(expr, (bool, Expr))
            if isinstance(expr, Expr):
                assert_expression_variables(expr, names)
            expr = action.get('show_when', None)
            if isinstance(expr, Expr):
                assert_expression_variables(expr, names)


@pytest.mark.parametrize(
    'mode', ['max', 'min', 'std', 'sum', 'mean', 'median']
)
def test_projections(mode):
    ll = LayerList()
    ll.append(Image(np.random.rand(8, 8, 8)))
    assert len(ll) == 1
    assert ll[-1].data.ndim == 3
    _project(ll, mode=mode)
    assert len(ll) == 2
    # because we use keepdims = True
    assert ll[-1].data.shape == (1, 8, 8)


@pytest.mark.parametrize(
    'mode',
    ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'],
)
def test_convert_dtype(mode):
    ll = LayerList()
    data = np.zeros((10, 10), dtype=np.int16)
    ll.append(Labels(data))
    assert ll[-1].data.dtype == np.int16

    data[5, 5] = 1000
    assert data[5, 5] == 1000
    if mode == 'int8' or mode == 'uint8':
        # label value 1000 is outside of the target data type range.
        with pytest.raises(AssertionError):
            _convert_dtype(ll, mode=mode)
        assert ll[-1].data.dtype == np.int16
    else:
        _convert_dtype(ll, mode=mode)
        assert ll[-1].data.dtype == np.dtype(mode)

    assert ll[-1].data[5, 5] == 1000
    assert ll[-1].data.flatten().sum() == 1000
