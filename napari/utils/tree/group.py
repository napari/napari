from __future__ import annotations

from typing import Generator, Iterable

from ..events import NestableEventedList
from .node import Node


class Group(NestableEventedList[Node], Node):
    def __init__(self, children: Iterable[Node] = (), name: str = "Group"):
        Node.__init__(self, name=name)
        NestableEventedList.__init__(self, children, basetype=Node)

    def __delitem__(self, key: int | slice):
        if isinstance(key, int):
            self[key].parent = None
        else:
            for item in self[key]:
                item.parent = None
        super().__delitem__(key)

    def insert(self, index: int, value):
        value.parent = self
        super().insert(index, value)

    def is_group(self) -> bool:
        return True

    def __contains__(self, other):
        for item in self.traverse():
            if item is other:
                return True
        return False

    def traverse(self, leaves_only=False) -> Generator[Node, None, None]:
        "Recursively traverse all nodes and leaves of the Group tree."
        if not leaves_only:
            yield self
        for child in self:
            yield from child.traverse(leaves_only)

    def _render(self) -> list[str]:
        """Recursively return list of strings that can render ascii tree."""
        lines = [self.name]

        for n, child in enumerate(self):
            space, bul = (
                ("   ", "└──") if n == len(self) - 1 else ("  │", "├──")
            )
            child_tree = child._render()
            lines.append(f"  {bul}" + child_tree.pop(0))
            lines.extend([space + lay for lay in child_tree])

        return lines
