from typing import TYPE_CHECKING, Generic, Iterable, Optional, TypeVar

from ._set import EventedSet

if TYPE_CHECKING:
    from pydantic.fields import ModelField

_T = TypeVar("_T")
_S = TypeVar("_S")


class Selection(EventedSet[_T]):
    """An unordered collection of selected elements, with a `current` item.

    There can only be one 'current' item. There can be multiple selected items.

    An item can be the current item and selected at the same time. Qt views
    will ensure that there is always a current item as keyboard navigation,
    for example, requires a current item.

    This pattern mimics current/selected items from Qt:
    https://doc.qt.io/qt-5/model-view-programming.html#current-item-and-selected-items

    Parameters
    ----------
    data : iterable, optional
        Elements to initialize the set with.
    current : Any, optional
        The current item.

    Events
    ------
    added (value: Set[_T])
        emitted after an item or items are added to the set.
        Will not be emitted if item was already in the set when added.
    removed (value: Set[_T])
        emitted after an item or items are removed from the set.
        Will not be emitted if the item was not in the set when discarded.
    current (value: _T, previous: _T)
        emitted when the current item has changed.
    """

    def __init__(self, data: Iterable[_T] = (), current: Optional[_T] = None):
        super().__init__(data=data)
        self.events.add(current=None)
        self._current = current

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return f"{clsname}({repr(self._set)}, current={self.current})"

    @property
    def current(self) -> Optional[_T]:
        """Get current item."""
        return self._current

    @current.setter
    def current(self, index: Optional[_T]):
        """Set current item."""
        if index == self._current:
            return
        previous, self._current = self._current, index
        self.events.current(value=index, previous=previous)

    def toggle(self, obj: _T):
        """Toggle selection state of obj."""
        self.symmetric_difference_update({obj})

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, field: 'ModelField'):
        """Pydantic validator."""
        from pydantic.utils import sequence_like

        if isinstance(v, dict):
            data = v.get("data", [])
            current = v.get("current", None)
        elif isinstance(v, Selection):
            data = v._set
            current = v.current
        else:
            data = v
            current = None

        if not sequence_like(data):
            raise TypeError(f'Value is not a valid sequence: {data}')

        # no type parameter was provided, just return
        if not field.sub_fields:
            return cls(data=data, current=current)

        # Selection[type] parameter was provided.  Validate contents
        type_field = field.sub_fields[0]
        errors = []
        for i, v_ in enumerate(data):
            _, error = type_field.validate(v_, {}, loc=f'[{i}]')
            if error:
                errors.append(error)
        if current is not None:
            _, error = type_field.validate(current, {}, loc='current')
            if error:
                errors.append(error)

        if errors:
            from pydantic import ValidationError

            raise ValidationError(errors, cls)  # type: ignore
        return cls(data=data, current=current)

    def _json_encode(self):
        """Return an object that can be used by json.dumps."""
        return {'data': super()._json_encode(), 'current': self.current}


class Selectable(Generic[_S]):
    """Mixin that adds a selection model to an object."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self._selection: Selection[_S] = Selection()

    @property
    def selection(self) -> Selection[_S]:
        """Get current selection."""
        return self._selection

    @selection.setter
    def selection(self, new_selection) -> None:
        """Set selection, without deleting selection model object."""
        self._selection.intersection_update(new_selection)
        self._selection.update(new_selection)
