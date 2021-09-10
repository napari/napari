import ast

import numpy as np
import pytest

from napari.components.layerlist import _CONTEXT_KEYS, LayerList
from napari.layers import Image
from napari.layers._layer_actions import (
    _LAYER_ACTIONS,
    ContextAction,
    SubMenu,
    _project,
)


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
    names = set(_CONTEXT_KEYS.keys())
    valid_keys = set.union(
        set(ContextAction.__annotations__), set(SubMenu.__annotations__)
    )
    for action_dict in _LAYER_ACTIONS:
        for action in action_dict.values():
            assert set(action).issubset(valid_keys)
            expr = action.get('enable_when', None)
            if expr:
                assert_expression_variables(expr, names)
            expr = action.get('show_when', None)
            if expr:
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
