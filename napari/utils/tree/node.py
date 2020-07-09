from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .group import Group


class Node:
    def __init__(self, name: str = 'Node'):
        self._parent: Optional[Group] = None
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def parent(self) -> Optional[Group]:
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent

    def is_group(self) -> bool:
        return False

    def index_in_parent(self) -> int:
        if self.parent is not None:
            return self.parent.index(self)
        return 0

    def index_from_root(self) -> Tuple[int, ...]:
        indices: List[int] = []
        item = self
        while item.parent is not None:
            indices.insert(0, item.index_in_parent())
            item = item.parent
        return tuple(indices)

    def traverse(self):
        yield self

    def __len__(self) -> int:
        return 0

    def __str__(self):
        """Render ascii tree string representation of this node"""
        return "\n".join(self._render())

    def _render(self):
        """Recursively return list of strings that can render ascii tree."""
        return [self.name]

    def emancipate(self):
        if self.parent is not None:
            self.parent.remove(self)
            return self
        raise IndexError('Cannot emancipate orphaned Node: {self!r}')
