from types import NoneType, UnionType
from typing import Annotated, Dict, List, Union, get_args, get_origin  # noqa


def iter_inner_types(type_):
    origin = get_origin(type_)
    args = get_args(type_)
    if origin in (list, List):  # noqa
        yield from iter_inner_types(args[0])
    elif origin in (dict, Dict):  # noqa
        yield from iter_inner_types(args[1])
    elif origin is Annotated:
        yield from iter_inner_types(args[0])
    elif origin in (UnionType, Union):
        for arg in args:
            yield from iter_inner_types(arg)
    elif type_ is not NoneType:
        yield type_


def get_inner_type(type_):
    """Roughly replacing pydantic.v1 Field.type_"""
    return Union[tuple(iter_inner_types(type_))]


def get_outer_type(type_):
    """Roughly replacing pydantic.v1 Field.outer_type_"""
    origin = get_origin(type_)
    args = get_args(type_)
    if origin in (UnionType, Union):
        # filter args to remove optional None
        args = tuple(filter(lambda t: t is not NoneType, get_args(type_)))
        if len(args) == 1:
            # it was just optional, pretend there was no None
            return get_outer_type(args[0])
        # It's an actual union of types, so there's no "outer type"
        return None
    if origin is not None:
        return origin
    return type_


def is_list_type(type_):
    """Roughly replacing pydantic.v1 comparison to SHAPE_LIST"""
    return get_outer_type(type_) is list


__all__ = (
    'get_inner_type',
    'get_outer_type',
    'is_list_type',
    'iter_inner_types',
)
