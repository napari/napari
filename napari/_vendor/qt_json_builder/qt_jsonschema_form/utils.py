from functools import wraps
from typing import Iterator

from qtpy import QtWidgets


class StateProperty(property):
    def setter(self, fset):
        @wraps(fset)
        def _setter(*args):
            *head, value = args
            if value is not None:
                fset(*head, value)

        return super().setter(_setter)


state_property = StateProperty


def reject_none(func):
    """Only invoke function if state argument is not None"""

    @wraps(func)
    def wrapper(self, state):
        if state is None:
            return
        func(self, state)

    return wrapper


def is_concrete_schema(schema: dict) -> bool:
    return "type" in schema


def iter_layout_items(layout) -> Iterator[QtWidgets.QLayoutItem]:
    return (layout.itemAt(i) for i in range(layout.count()))


def iter_layout_widgets(
    layout: QtWidgets.QLayout,
) -> Iterator[QtWidgets.QWidget]:
    return (i.widget() for i in iter_layout_items(layout))
