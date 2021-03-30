"""Example of using low-level QtNodeTreeView with Node and Group

:class:`napari.utils.tree.Node` is a class that may be used as a mixin that
allows an object to be a member of a "tree".

:class:`napari.utils.tree.Group` is a (nestable) mutable sequence of Nodes, and
is also itself a Node (this is the "composite" pattern):
https://refactoring.guru/design-patterns/composite/python/example

These two classes may be used to create tree-like data structures that behave
like pure python lists of lists.

This examples shows that :class:`napari._qt.containers.QtNodeTreeView`
is capable of providing a basic GUI for any tree structure based on
`napari.utils.tree.Group`.
"""
import napari
from napari.qt import get_app
from napari._qt.containers import QtNodeTreeView
from napari.utils.tree import Node, Group

get_app()

# create a group of nodes. 
root = Group(
    [
        Node(name='6'),
        Group(
            [
                Node(name='1'),
                Group([Node(name='2'), Node(name='3')], name="g2"),
                Node(name='4'),
                Node(name='5'),
                Node(name='tip'),
            ],
            name="g1",
        ),
        Node(name='7'),
        Node(name='8'),
        Node(name='9'),
    ],
    name="root",
)
# create Qt view onto the Group
view = QtNodeTreeView(root)
# show the view
view.show()


# pretty __str__ makes nested tree structure more interpretable
print(root)
# root
#   ├──6
#   ├──g1
#   │  ├──1
#   │  ├──g2
#   │  │  ├──2
#   │  │  └──3
#   │  ├──4
#   │  ├──5
#   │  └──tip
#   ├──7
#   ├──8
#   └──9


# spy on events
root.events.reordered.connect(lambda e: print("reordered to: ", e.value))
root.selection.events.changed.connect(
    lambda e: print(
        f"selection changed.  added: {e.added}, removed: {e.removed}"
    )
)
root.selection.events._current.connect(
    lambda e: print(f"current item changed to: {e.value}")
)

napari.run()
