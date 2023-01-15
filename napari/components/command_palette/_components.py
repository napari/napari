from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

_R = TypeVar("_R")


@dataclass
class Command(Generic[_R]):
    """A command representation."""

    function: Callable[..., _R]
    title: str
    desc: str
    tooltip: str = ""
    when: Callable[..., bool] = field(default=lambda *_: True)

    def __call__(self, *args, **kwargs) -> _R:
        return self.function(*args, **kwargs)

    def fmt(self) -> str:
        """Format command for display in the palette."""
        if self.title:
            return f"{self.title}: {self.desc}"
        return self.desc

    def matches(self, input_text: str) -> bool:
        """Return True if the command matches the input text."""
        fmt = self.fmt().lower()
        words = input_text.lower().split(" ")
        return all(word in fmt for word in words)

    def enabled(self, *args) -> bool:
        """Return True if the command is enabled."""
        return self.when(*args)


class Storage:
    """The variable storage."""

    _INSTANCES: dict[str, Storage] = {}

    def __init__(self):
        self._varmap: dict[str, Callable[[], Any]] = {}

    def mark_getter(self, name: str, func: Callable[[], Any] | None = None):
        """Mark a function as a getter for variable named 'name'."""

        def wrapper(f: Callable[[], Any]):
            self._varmap[name] = f
            return f

        return wrapper if func is None else wrapper(func)

    def mark_constant(self, name: str, value: Any):
        """Mark a constant value for variable named 'name'."""
        self._varmap[name] = lambda *_: value

    @classmethod
    def instance(cls, name: str = "") -> Storage:
        if name not in cls._INSTANCES:
            cls._INSTANCES[name] = Storage()
        return cls._INSTANCES[name]

    def call(self, func: Callable[..., _R], viewer) -> _R:
        """Call a function with variables from the storage."""
        args = []
        for v in inspect.signature(func).parameters.keys():
            if getter := self._varmap.get(v, None):
                args.append(getter(viewer))
            else:
                raise ValueError(f"Variable {v} not found in storage")
        return func(*args)


def get_storage(name: str = "") -> Storage:
    """Get the name specific storage instance."""
    return Storage.instance(name)
