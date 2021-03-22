from typing import TYPE_CHECKING, Generic, Iterable, Optional, TypeVar

from ._set import EventedSet

if TYPE_CHECKING:
    from pydantic.fields import ModelField

_T = TypeVar("_T")
_S = TypeVar("_S")


class Selection(EventedSet[_T]):
    """An model of selected items, with a `active` and `current` item.

    There can only be one `active` and one `current` item, but there can be
    multiple selected items.  An "active" item is defined as a single selected
    item (if multiple items are selected, there is no active item).  The
    "current" item is mostly useful for (e.g.) keyboard actions: even with
    multiple items selected, you may only have one current item, and keyboard
    events (like up and down) can modify that current item.  It's possible to
    have a current item without an active item, but an active item will almost
    always be the current item.

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
        The current item, if any.

    Attributes
    ----------
    active : Any, optional
        The active item, if any.

    Events
    ------
    changed (added: Set[_T], removed: Set[_T])
        Emitted when the set changes, includes item(s) that have been added
        and/or removed from the set.
    current (value: _T)
        emitted when the current item has changed.
    active (value: _T)
        emitted when the current item has changed.
    """

    def __init__(self, data: Iterable[_T] = (), current: Optional[_T] = None):
        self._active: Optional[_T] = None
        self._current = current
        super().__init__(data=data)
        self.events.add(current=None, active=None)
        self.events.changed.connect(self._update_active)
        self._update_active()

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
        self._current = index
        self.events.current(value=index)

    @property
    def active(self) -> Optional[_T]:
        """Return the currently active item or None."""
        return self._active

    @active.setter
    def active(self, value: Optional[_T]):
        """Set the active item.

        This make `value` the only selected item, and make it current.
        """
        if value == self._active:
            return
        self._active = value
        self.clear() if value is None else self.select_only(value)
        self.current = value
        self.events.active(value=value)

    def _update_active(self, event=None):
        """On a selection event, update the active item based on selection.

        (An active item is a single selected item).
        """
        if len(self) == 1:
            self.active = list(self)[0]
        else:
            if self._active is not None:
                self._active = None
                self.events.active(value=None)

    def clear(self, keep_current: bool = False) -> None:
        """Clear the selection."""
        if not keep_current:
            self.current = None
        super().clear()

    def toggle(self, obj: _T):
        """Toggle selection state of obj."""
        self.symmetric_difference_update({obj})

    def select_only(self, obj: _T):
        """Unselect everything but `obj`. Add to selection if not present."""
        self.intersection_update({obj})
        self.add(obj)

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, field: 'ModelField'):
        """Pydantic validator."""
        from pydantic.utils import sequence_like

        if isinstance(v, dict):
            data = v.get("selection", [])
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
        # we don't serialize active, as it's gleaned from the selection.
        return {'selection': super()._json_encode(), 'current': self.current}


class Selectable(Generic[_S]):
    """Mixin that adds a selection model to an object."""

    def __init__(self, *args, **kwargs) -> None:
        self._selection: Selection[_S] = Selection()
        super().__init__(*args, **kwargs)  # type: ignore

    @property
    def selection(self) -> Selection[_S]:
        """Get current selection."""
        return self._selection

    @selection.setter
    def selection(self, new_selection: Iterable[_S]) -> None:
        """Set selection, without deleting selection model object."""
        self._selection.intersection_update(new_selection)
        self._selection.update(new_selection)
