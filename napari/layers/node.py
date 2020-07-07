from __future__ import annotations

from abc import ABC
from typing import Generator, Iterable, List, Optional, Tuple, Union

from ..utils.list._evented_list import NestableEventedList


class Node(ABC):
    def __init__(self, name: str = 'Node'):
        print("node init")
        self._parent: Optional['Group'] = None
        self._name = name

    def __len__(self) -> int:
        return 0

    @property
    def parent(self) -> Optional['Group']:
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

    def __str__(self):
        """Render ascii tree string representation of this node"""
        return "\n".join(self._render())

    def _render(self):
        """Recursively return list of strings that can render ascii tree."""
        return [self.name]


class Group(Node, NestableEventedList):
    def __init__(self, children: Iterable[Node] = None, name='Group') -> None:
        Node.__init__(self, name=name)
        NestableEventedList.__init__(self)
        self.extend(children or [])

    def __len__(self) -> int:
        return NestableEventedList.__len__(self)

    def __delitem__(self, key: Union[int, slice]):
        if isinstance(key, int):
            self[key].parent = None
        else:
            for item in self[key]:
                item.parent = None
        super().__delitem__(key)

    def insert(self, index: int, value):
        value.parent = self
        super().insert(index, value)

    def extend(self, values: Iterable):
        for v in values:
            v.parent = self
        super().extend(values)

    def is_group(self) -> bool:
        return True

    def traverse(self) -> Generator[Node, None, None]:
        "Recursively traverse all nodes and leaves of the Group tree."
        yield self
        for child in self:
            yield from child.traverse()

    def get_nested_index(self, indices: Tuple[int, ...]) -> Node:
        item: Group = self
        for idx in indices:
            item = item[idx]
        return item

    def _render(self):
        """Recursively return list of strings that can render ascii tree."""
        lines = []
        lines.append(self.name)

        for n, child in enumerate(self):
            child_tree = child._render()
            lines.append('  +--' + child_tree.pop(0))
            spacer = '   ' if n == len(self) - 1 else '  |'
            lines.extend([spacer + l for l in child_tree])

        return lines
