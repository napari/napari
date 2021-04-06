from typing import Generator, Iterable, List, TypeVar

from ..events.containers._nested_list import MaybeNestedIndex
from ..events.containers._selectable_list import SelectableNestableEventedList
from .node import Node

NodeType = TypeVar("NodeType", bound=Node)


class Group(Node, SelectableNestableEventedList[NodeType]):
    """An object that can contain other objects in a composite Tree pattern.

    The ``Group`` (aka composite) is an element that has sub-elements:
    which may be ``Nodes`` or other ``Groups``.  By inheriting from
    :class:`NestableEventedList`, ``Groups`` have basic python list-like
    behavior and emit events when modified.  The main addition in this class
    is that when objects are added to a ``Group``, they are assigned a
    ``.parent`` attribute pointing to the group, which is removed upon
    deletion from the group.

    For additional background on the composite design pattern, see:
    https://refactoring.guru/design-patterns/composite

    Parameters
    ----------
    children : Iterable[Node], optional
        Items to initialize the Group, by default ().  All items must be
        instances of ``Node``.
    name : str, optional
        A name/id for this group, by default "Group"
    """

    def __init__(
        self,
        children: Iterable[NodeType] = (),
        name: str = "Group",
        basetype=Node,
    ):
        Node.__init__(self, name=name)
        SelectableNestableEventedList.__init__(
            self, data=children, basetype=basetype
        )

    def __delitem__(self, key: MaybeNestedIndex):
        """Remove item at ``key``, and unparent."""
        if isinstance(key, (int, tuple)):
            self[key].parent = None  # type: ignore
        else:
            for item in self[key]:
                item.parent = None
        super().__delitem__(key)

    def insert(self, index: int, value):
        """Insert ``value`` as child of this group at position ``index``."""
        value.parent = self
        super().insert(index, value)

    def is_group(self) -> bool:
        """Return True, indicating that this ``Node`` is a ``Group``."""
        return True

    def __contains__(self, other):
        """Return true if ``other`` appears anywhere under this group."""
        return any(item is other for item in self.traverse())

    def traverse(self, leaves_only=False) -> Generator[NodeType, None, None]:
        """Recursive all nodes and leaves of the Group tree."""
        if not leaves_only:
            yield self
        for child in self:
            yield from child.traverse(leaves_only)

    def _render(self) -> List[str]:
        """Recursively return list of strings that can render ascii tree."""
        lines = [self._node_name()]

        for n, child in enumerate(self):
            spacer, bul = (
                ("   ", "└──") if n == len(self) - 1 else ("  │", "├──")
            )
            child_tree = child._render()
            lines.append(f"  {bul}" + child_tree.pop(0))
            lines.extend([spacer + lay for lay in child_tree])

        return lines
