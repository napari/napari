"""Example of using low-level QtNodeTreeView with Node and Group

:class:`napari.utils.tree.Node` is a class that may be used as a mixin that
allows an object to be a member of a "tree".

:class:`napari.utils.tree.Group` is a (nestable) mutable sequence of Nodes, and
is also itself a Node (this is the "composite" patter):
https://refactoring.guru/design-patterns/composite/python/example

These two classes may be used to create tree-like data structures that behave
like pure python lists of lists.

This examples shows that :class:`napari._qt.tree.qt_tree_view.QtNodeTreeView`
is capable of providing a basic GUI for any tree structure based on
`napari.utils.tree.Group`.
"""
from qtpy.QtWidgets import QApplication

from napari._qt.tree.qt_tree_view import QtNodeTreeView
from napari.utils.tree import Node, Group
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
logging.basicConfig(level=logging.DEBUG, format=colorlog_format)

app = QApplication([])

tip = Node(name='tip')
lg2 = Group([Node(name='2'), Node(name='3')], name="g2")
lg1 = Group(
    [Node(name='1'), lg2, Node(name='4'), Node(name='5'), tip],
    name="g1",
)
root = Group(
    [Node(name='6'), lg1, Node(name='7'), Node(name='8'), Node(name='9')],
    name="root",
)

# pretty repr makes nested tree structure more interpretable
print(root)
root.events.reordered.connect(lambda e: print(e.value))

view = QtNodeTreeView(root)
view.show()

app.exec_()
