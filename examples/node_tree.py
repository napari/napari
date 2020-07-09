from napari import gui_qt
from napari._qt.tree._tree_view import QtNodeTreeView
from napari.utils.tree import Node, Group


with gui_qt():
    tip = Node(name='tip')
    lg2 = Group([Node(name='2'), Node(name='3')], name="g2")
    lg1 = Group(
        [Node(name='1'), lg2, Node(name='4'), Node(name='5'), tip], name="g1",
    )
    root = Group(
        [Node(name='6'), lg1, Node(name='7'), Node(name='8'), Node(name='9')],
        name="root",
    )
    tree = QtNodeTreeView(root)
    model = tree.model()
    tree.show()
