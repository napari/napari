from qtpy.QtWidgets import QAbstractItemView, QTreeView, QWidget

from ._tree_model import QtNodeTreeModel


class QtNodeTreeView(QTreeView):
    def __init__(self, root, parent: QWidget = None):
        super().__init__(parent)
        _model = QtNodeTreeModel(root, self)
        root.events.connect(lambda e: _model.layoutChanged.emit())
        root.events.connect(lambda e: print(f"Root event: {e.type} {e.value}"))
        self.setModel(_model)
        self.setHeaderHidden(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.selectionModel().selectionChanged.connect(_model.setSelection)
        self.setStyleSheet(r"QTreeView::item {padding: 4px;}")


if __name__ == '__main__':
    from napari import gui_qt
    from napari.utils.tree import Node, Group

    with gui_qt():
        tip = Node(name='tip')
        lg2 = Group(name="g2", children=[Node(name='2')])
        lg1 = Group(
            name="g1", children=[lg2, Node(name='3'), tip, Node(name='1')]
        )
        root = Group(
            name="root",
            children=[
                lg1,
                Node(name='4'),
                Node(name='5'),
                Node(name='6'),
                Node(name='7'),
                Node(name='8'),
                Node(name='9'),
            ],
        )
        tree = QtNodeTreeView(root)
        model = tree.model()
        tree.show()
