from __future__ import annotations

import logging
import pickle
from typing import Any, List, Tuple, Union, cast

from qtpy.QtCore import QAbstractItemModel, QMimeData, QModelIndex, Qt
from qtpy.QtWidgets import QWidget
from ...utils.tree import Group, Node
from ...layers import Layer, LayerGroup


logger = logging.getLogger(__name__)


# TODO: cleanup stuff related to MIME formats and convenience methods
class NodeMimeData(QMimeData):
    def __init__(self, nodes: List[Node] = None):
        super().__init__()
        self.nodes = nodes or []
        if nodes:
            self.setData(
                'application/x-tree-node', pickle.dumps(self.node_indices())
            )
            self.setText(" ".join([node.name for node in nodes]))

    def formats(self) -> List[str]:
        return ['application/x-tree-node', 'text/plain']

    def node_indices(self) -> List[Tuple[int, ...]]:
        return [node.index_from_root() for node in self.nodes]

    def node_names(self) -> List[str]:
        return [node.name for node in self.nodes]


# https://doc.qt.io/qt-5/model-view-programming.html#model-subclassing-reference
class QtNodeTreeModel(QAbstractItemModel):
    def __init__(self, root: Group, parent: QWidget = None):
        super().__init__(parent)
        self.setRoot(root)

    # ########## Reimplemented Public Qt Functions ##################

    def canDropMimeData(self, *args):
        parent: QModelIndex = args[-1]
        return self.getItem(parent).is_group()

    def columnCount(self, parent: QModelIndex) -> int:
        return 1

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> Any:
        """Return data stored under ``role`` for the item at ``index``."""
        item = self.getItem(index)
        if role == Qt.DisplayRole:
            return str(item.name)
        if role == Qt.UserRole:
            return self.getItem(index)
        return None

    def dropMimeData(
        self,
        data: QMimeData,
        action: Qt.DropAction,
        destRow: int,
        col: int,
        parent: QModelIndex,
    ) -> bool:
        """Handles dropped data that ended with ``action``.

        Returns true if the data and action were handled by the model;
        otherwise returns false.

        """
        if not data or action != Qt.MoveAction:
            return False
        if not data.hasFormat(self.mimeTypes()[0]):
            return False

        dest_idx = self.getItem(parent).index_from_root()
        dest_idx = dest_idx + (destRow,)

        moving_indices = data.node_indices()

        logger.debug(f"dropMimeData: indices {moving_indices} ➡ {dest_idx}")

        if len(moving_indices) == 1:
            self._root.move(moving_indices[0], dest_idx)
        else:
            self._root.move_multiple(moving_indices, dest_idx)
        # If we return true, removeRows is called!?
        return False

    def flags(self, index: QModelIndex) -> Union[Qt.ItemFlag, Qt.ItemFlags]:
        """Returns the item flags for the given index."""
        if not index.isValid():
            # for root
            return Qt.ItemIsDropEnabled

        base_flags = (
            Qt.ItemIsSelectable
            | Qt.ItemIsEditable
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsDragEnabled
            | Qt.ItemIsEnabled
            | Qt.ItemIsDropEnabled
        )
        item = self.getItem(index)
        if item.is_group():
            return base_flags
        return base_flags | Qt.ItemNeverHasChildren

    def index(
        self, row: int, column: int = 0, parent: QModelIndex = QModelIndex()
    ) -> QModelIndex:
        """Return index of the item specified by row, column and parent index.
        """
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        parentItem = cast(Group, self.getItem(parent))
        if parentItem.is_group():
            return self.createIndex(row, column, parentItem[row])
        return QModelIndex()

    def mimeData(self, indices: List[QModelIndex]) -> QMimeData:
        """Return object containing serialized data corresponding to indexes.
        """
        if not indices:
            return 0
        return NodeMimeData([self.getItem(i) for i in indices])

    def mimeTypes(self):
        return ['application/x-tree-node', 'text/plain']

    def parent(self, index: QModelIndex) -> QModelIndex:
        if not index.isValid():
            return QModelIndex()

        parentItem = self.getItem(index).parent
        if parentItem is None or parentItem == self._root:
            return QModelIndex()

        return self.createIndex(parentItem.index_in_parent(), 0, parentItem)

    def rowCount(self, parent: QModelIndex) -> int:
        return len(self.getItem(parent))

    def supportedDropActions(self) -> Qt.DropActions:
        return Qt.MoveAction

    # ########## New methods added for our model ##################

    def setRoot(self, root: Group):
        self._root = root
        self._root.events.removing.connect(self._on_begin_removing)
        self._root.events.removed.connect(self._on_end_remove)
        self._root.events.inserting.connect(self._on_begin_inserting)
        self._root.events.inserted.connect(self._on_end_insert)
        self._root.events.moving.connect(self._on_begin_moving)
        self._root.events.moved.connect(self._on_end_move)

    def _on_end_remove(self, e):
        self.endRemoveRows()

    def _on_end_insert(self, e):
        self.endInsertRows()

    def _on_end_move(self, e):
        self.endMoveRows()

    def getItem(self, index: QModelIndex) -> Node:
        if index.isValid():
            item = index.internalPointer()
            if item is not None:
                return item
        return self._root

    def findIndex(self, obj: Node) -> QModelIndex:
        hits = self.match(
            self.index(0),
            Qt.UserRole,
            obj,
            1,
            Qt.MatchExactly | Qt.MatchRecursive,
        )
        if hits:
            return hits[0]
        raise IndexError(f"Could not find node {obj!r} in the model")

    def nestedIndex(self, nested_index: Tuple[int, ...]) -> QModelIndex:
        parent = QModelIndex()
        if isinstance(nested_index, tuple):
            if not nested_index:
                return parent
            *parents, child = nested_index
            for i in parents:
                parent = self.index(i, 0, parent)
        elif isinstance(nested_index, int):
            child = nested_index
        else:
            raise TypeError("nested_index must be an int or tuple of int.")
        return self.index(child, 0, parent)

    def _on_begin_removing(self, event):
        par, idx = self._split_nested_index(event.index)
        self.beginRemoveRows(par, idx, idx)

    def _on_begin_inserting(self, event):
        par, idx = self._split_nested_index(event.index)
        self.beginInsertRows(par, idx, idx)

    def _on_begin_moving(self, event):
        src_par, src_idx = self._split_nested_index(event.index)
        dest_par, dest_idx = self._split_nested_index(event.new_index)
        logger.debug(
            f"beginMoveRows({self.getItem(src_par).name}, {src_idx}, "
            f"{self.getItem(dest_par).name}, {dest_idx})"
        )

        self.beginMoveRows(src_par, src_idx, src_idx, dest_par, dest_idx)

    def _split_nested_index(
        self, nested_index: Union[int, Tuple[int, ...]]
    ) -> Tuple[QModelIndex, int]:
        """Given a nested index, return (nested_parent_index, row)."""
        # TODO: split after using nestedIndex?
        if isinstance(nested_index, int):
            return QModelIndex(), nested_index
        par = QModelIndex()
        *_p, idx = nested_index
        for i in _p:
            par = self.index(i, 0, par)
        return par, idx


class QtLayerTreeModel(QtNodeTreeModel):
    def __init__(self, root: LayerGroup, parent: QWidget = None):
        super().__init__(root, parent)
        self.setRoot(root)

    def getItem(self, index: QModelIndex) -> Layer:
        return cast(Layer, super().getItem(index))

    def data(self, index: QModelIndex, role: Qt.ItemDataRole):
        """Return data stored under ``role`` for the item at ``index``."""
        item = self.getItem(index)
        if role == Qt.DisplayRole:
            return item.name
        if role == Qt.CheckStateRole:
            return item.visible
        if role == Qt.UserRole:
            return self.getItem(index)
        return None
