from __future__ import annotations

from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from .group import Group


class Node:
    """An object that can be a member of a :class:`Group`.

    ``Node`` forms the base object of a composite Tree pattern. This class
    describes operations that are common to both simple (node) and complex
    (group) elements of the tree.  ``Node`` may not have children, whereas
    :class:`~napari.utils.tree.group.Group` can.

    For additional background on the composite design pattern, see:
    https://refactoring.guru/design-patterns/composite

    Parameters
    ----------
    name : str, optional
        A name/id for this node, by default "Node"

    Attributes
    ----------
    parent : Group, optional
        The parent of this Node.
    """

    def __init__(self, name: str = "Node"):

        self.parent: Group | None = None
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def is_group(self) -> bool:
        return False

    def index_in_parent(self) -> int:
        if self.parent is not None:
            return self.parent.index(self)
        # TODO: check if this can be None?
        return 0

    def index_from_root(self) -> tuple[int, ...]:
        item = self
        indices: list[int] = []
        while item.parent is not None:
            indices.insert(0, item.index_in_parent())
            item = item.parent
        return tuple(indices)

    def root(self):
        """Get the root parent."""
        obj = self
        while obj.parent:
            obj = obj.parent
        return obj

    def traverse(
        self, leaves_only=False, with_ancestors=False
    ) -> Generator[Node, None, None]:
        yield self

    def __str__(self):
        """Render ascii tree string representation of this node"""
        return "\n".join(self._render())

    def _render(self) -> list[str]:
        """Return list of strings that can render ascii tree.

        For ``Node``, we just return the name of this specific node.
        """
        return [self.name]

    def unparent(self):
        if self.parent is not None:
            self.parent.remove(self)
            return self
        raise IndexError("Cannot unparent orphaned Node: {self!r}")
