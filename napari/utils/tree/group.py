from .node import Node
from ..events import NestableEventedList

from typing import Generator, Iterable, Tuple, Union


class Group(Node, NestableEventedList):
    def __init__(self, children: Iterable[Node] = None, name='Group') -> None:
        Node.__init__(self, name=name)
        NestableEventedList.__init__(self)
        self.extend(children or [])

    def __len__(self) -> int:
        return NestableEventedList.__len__(self)

    def __delitem__(self, key: Union[int, slice]):
        if isinstance(key, int):
            print('removing', self[key])
            self[key].parent = None
        else:
            for item in self[key]:
                print('removing', item)
                item.parent = None
        super().__delitem__(key)

    def insert(self, index: int, value):
        value.parent = self
        print('inserted', value)
        super().insert(index, value)

    def extend(self, values: Iterable):
        for v in values:
            v.parent = self
            print('inserted', v)
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
