import ast

from napari.components.layerlist import _CONTEXT_KEYS
from napari.layers._layer_actions import _LAYER_ACTIONS, ContextAction


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
    valid_keys = set(ContextAction.__annotations__)
    for action in _LAYER_ACTIONS.values():
        if action == {}:  # empty separator
            continue
        assert set(action) == set(valid_keys)
        expr = action.get('enable_when', None)
        if expr:
            assert_expression_variables(expr, names)
        expr = action.get('show_when', None)
        if expr:
            assert_expression_variables(expr, names)
