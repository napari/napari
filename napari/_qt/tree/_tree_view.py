from qtpy.QtWidgets import QAbstractItemView, QTreeView, QWidget

from ._tree_model import QtNodeTreeModel


class QtNodeTreeView(QTreeView):
    def __init__(self, root, parent: QWidget = None):
        super().__init__(parent)
        _model = QtNodeTreeModel(root, self)
        # root.events.connect(self._update_view_selection)
        root.events.connect(lambda e: print(f"Root event: {e.type} {e.value}"))
        self.setModel(_model)
        self.setHeaderHidden(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # self.selectionModel().selectionChanged.connect(
        #     self._update_model_selection
        # )
        self.setStyleSheet(r"QTreeView::item {padding: 4px;}")
        # self._update_view_selection()

    # def _update_model_selection(
    #     self, selected: QItemSelection, deselected: QItemSelection
    # ):
    #     model = self.model()
    #     for idx in selected.indexes():
    #         model.getItem(idx).selected = True
    #     for idx in deselected.indexes():
    #         model.getItem(idx).selected = False

    # def _update_view_selection(self, event=None):
    #     model = self.model()
    #     model.layoutChanged.emit()
    #     # TODO: optimize by not recursing the whole tree.
    #     selection = QItemSelection()
    #     for idx in model.iter_indices():
    #         if model.getItem(idx).selected:
    #             selection.select(idx, idx)

    #     sel_model = self.selectionModel()
    #     with qt_signals_blocked(sel_model):
    #         sel_model.select(selection, QItemSelectionModel.ClearAndSelect)


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
