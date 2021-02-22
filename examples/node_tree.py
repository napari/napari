from napari.qt import get_app, run
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

get_app()

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
model = view.model()
view.show()

run()
