from qtpy.QtWidgets import QApplication
from napari._qt.tree.qt_tree_view import QtNodeTreeView
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

app = QApplication([])

root = Group(
    [
        Node(name="1"),
        Group(
            [
                Node(name="2"),
                Group([Node(name="3"), Node(name="4")], name="g2"),
                Node(name="5"),
                Node(name="6"),
                Node(name="7"),
            ],
            name="g1",
        ),
        Node(name="8"),
        Node(name="9"),
    ],
    name="root",
)
tree = QtNodeTreeView(root)
tree.show()

app.exec_()
