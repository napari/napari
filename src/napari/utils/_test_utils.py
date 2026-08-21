"""
File with things that are useful for testing, but not to be fixtures
"""

import inspect
from dataclasses import dataclass, field

import numpy as np
from docstring_parser import parse

from napari.utils._proxies import ReadOnlyWrapper


@dataclass
class MouseEvent:
    """Create a subclass for simulating vispy mouse events."""

    type: str
    is_dragging: bool = False
    button: int | None = None
    modifiers: list[str] = field(default_factory=list)
    position: tuple[int, int] | tuple[int, int, int] = (
        0,
        0,
    )  # world coords
    pos: np.ndarray = field(
        default_factory=lambda: np.zeros(2)
    )  # canvas coords
    view_direction: list[float] | None = None
    up_direction: list[float] | None = None
    dims_displayed: list[int] = field(default_factory=lambda: [0, 1])
    delta: tuple[float, float] | None = None
    native: bool | None = None


def read_only_mouse_event(*args, **kwargs):
    return ReadOnlyWrapper(
        MouseEvent(*args, **kwargs), exceptions=('handled',)
    )


def validate_all_params_in_docstring(
    func: type, section_name: str = 'param'
) -> None:
    """
    Validate if all the parameters in the function signature are present in the docstring.
    """
    assert func.__doc__ is not None, (
        f'Function {func.__name__} has no docstring'
    )

    parsed = parse(func.__doc__)
    params = [x for x in parsed.params if x.args[0] == section_name]
    # get only parameters from docstring

    signature = inspect.signature(func)
    sig_params = set(signature.parameters)
    docs_params = {x.arg_name for x in params}
    assert sig_params == docs_params, (
        f'Parameters in signature and docstring do not match {sig_params.symmetric_difference(docs_params)}'
    )
    for sig, doc in zip(signature.parameters.values(), params, strict=False):
        assert sig.name == doc.arg_name, (
            'Parameters in signature and docstring are not in the same order.'
        )
        # assert sig.annotation == doc.type_name, f"Type of parameter {sig.name} in signature and docstring do not match"


def validate_kwargs_sorted(func):
    """
    Validate if the keyword arguments in the function signature are sorted alphabetically.
    """
    signature = inspect.signature(func)
    kwargs_list = [
        x.name
        for x in signature.parameters.values()
        if x.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    assert kwargs_list == sorted(kwargs_list), (
        'Keyword arguments are not sorted in function signature'
    )


def validate_docstring_parent_class_consistency(
    klass: type, skip=('data', 'ndim'), section_name: str = 'param'
):
    """
    Validate if the docstrings of the class parameters and type information
    are consistent with the parent class.

    Parameters
    ----------
    klass : type
        The class to validate.
    skip : tuple or set, optional
        Name of parameters that we know are different from the parent class.
        By default, ('data', 'ndim').

    Raises
    ------
    AssertionError
        If the docstrings of the parameters are not consistent with the parent class.
    """
    parsed = parse(klass.__doc__)
    params = {
        x.arg_name: x
        for x in parsed.params
        if x.args[0] == section_name and x.arg_name not in skip
    }
    for base_klass in klass.__bases__:
        base_parsed = {
            x.arg_name: x
            for x in parse(base_klass.__doc__).params
            if x.args[0] == section_name
        }
        for name, doc in params.items():
            if name not in base_parsed:
                continue
            assert doc.description == base_parsed[name].description, (
                f'Description of parameter "{name}" in {klass.__name__} and {base_klass.__name__} do not match'
            )
            assert doc.type_name == base_parsed[name].type_name, (
                f'Type annotation of parameter "{name}" in {klass.__name__} ({doc.type_name}) and {base_klass.__name__} ({base_parsed[name].type_name}) do not match'
            )
