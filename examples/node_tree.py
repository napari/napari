from napari import gui_qt
from napari._qt.tree._tree_view import QtNodeTreeView
from napari.utils.tree import Node, Group
import logging

end = "\033[0m"
Bold = "\033[1m"
Dim = "\033[2m"
ResetDim = "\033[22m"
red = "\033[0;31m"
green = "\033[0;32m"
colorlog_format = f'{green}%(levelname)9s:{end} {Dim}%(name)36s.{ResetDim}{red}%(funcName)-18s{end}{"%(message)s"}'
logging.basicConfig(level=logging.DEBUG, format=colorlog_format)

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
