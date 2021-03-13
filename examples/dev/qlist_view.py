"""Example of using low-level QtNodeTreeView with Node and Group

:class:`napari.utils.tree.Node` is a class that may be used as a mixin that
allows an object to be a member of a "tree".

:class:`napari.utils.tree.Group` is a (nestable) mutable sequence of Nodes, and
is also itself a Node (this is the "composite" patter):
https://refactoring.guru/design-patterns/composite/python/example

These two classes may be used to create tree-like data structures that behave
like pure python lists of lists.

This examples shows that :class:`napari._qt.containers.QtNodeTreeView`
is capable of providing a basic GUI for any tree structure based on
`napari.utils.tree.Group`.
"""
import napari
from napari.qt import get_app
from napari._qt.containers import QtListView
from napari.utils.events.containers import SelectableEventedList

import logging

# create some readable logging.  Drag and drop the items in the tree to
# see what sort of events are happening in the background.
end = "\033[0m"
Bold = "\033[1m"
Dim = "\033[2m"
ResetDim = "\033[22m"
red = "\033[0;31m"
green = "\033[0;32m"
colorlog_format = f'{green}%(levelname)6s:{end} {Dim}%(name)43s.{ResetDim}{red}%(funcName)-18s{end}{"%(message)s"}'
# logging.basicConfig(level=logging.DEBUG, format=colorlog_format)

get_app()


class T:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


root: SelectableEventedList[T] = SelectableEventedList(
    [T('a'), T('b'), T('c')]
)
# pretty repr makes nested tree structure more interpretable
print(root)
root.events.reordered.connect(lambda e: print(e.value))
root.selection.events.connect(lambda e: print("selection", e.type, e.value))
view = QtListView(root)

view.show()

napari.run()
