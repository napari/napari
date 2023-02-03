from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

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
        out = self.function(*args, **kwargs)
        return out

    def fmt(self) -> str:
        """Format command for display in the palette."""
        if self.title:
            return f"{self.title} > {self.desc}"
        return self.desc

    def matches(self, input_text: str) -> bool:
        """Return True if the command matches the input text."""
        fmt = self.fmt().lower()
        words = input_text.lower().split(" ")
        return all(word in fmt for word in words)

    def enabled(self, *args) -> bool:
        """Return True if the command is enabled."""
        return self.when(*args)
