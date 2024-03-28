"""
File with things that are useful for testing, but not to be fixtures
"""

import inspect
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from docstring_parser import parse

from napari.utils._proxies import ReadOnlyWrapper


@dataclass
class MouseEvent:
    """Create a subclass for simulating vispy mouse events."""

    type: str
    is_dragging: bool = False
    modifiers: list[str] = field(default_factory=list)
    position: Union[tuple[int, int], tuple[int, int, int]] = (
        0,
        0,
    )  # world coords
    pos: np.ndarray = field(
        default_factory=lambda: np.zeros(2)
    )  # canvas coords
    view_direction: Optional[list[float]] = None
    up_direction: Optional[list[float]] = None
    dims_displayed: list[int] = field(default_factory=lambda: [0, 1])
    delta: Optional[tuple[float, float]] = None
    native: Optional[bool] = None


def read_only_mouse_event(*args, **kwargs):
    return ReadOnlyWrapper(
        MouseEvent(*args, **kwargs), exceptions=('handled',)
    )


def validate_docstring_all_params_in(func):
    assert func.__doc__ is not None, f'Function {func} has no docstring'

    parsed = parse(func.__doc__)
    params = [x for x in parsed.params if x.args[0] == 'param']
    # get only parameters from docstring

    signature = inspect.signature(func)
    assert set(signature.parameters.keys()) == {
        x.arg_name for x in params
    }, 'Parameters in signature and docstring do not match'
    for sig, doc in zip(signature.parameters.values(), params):
        assert (
            sig.name == doc.arg_name
        ), 'Parameters in signature and docstring are not in the same order.'
        # assert sig.annotation == doc.type_name, f"Type of parameter {sig.name} in signature and docstring do not match"


def validate_kwargs_sorted(func):
    signature = inspect.signature(func)
    kwargs_list = [
        x.name
        for x in signature.parameters.values()
        if x.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    assert kwargs_list == sorted(
        kwargs_list
    ), 'Keyword arguments are not sorted in function signature'
