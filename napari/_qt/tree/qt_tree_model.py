from __future__ import annotations

import logging
import pickle
from typing import cast

from qtpy.QtCore import QAbstractItemModel, QMimeData, QModelIndex, Qt
from qtpy.QtWidgets import QWidget

from ...utils.events import disconnect_events
from ...utils.tree import Group, Node

logger = logging.getLogger(__name__)


# https://doc.qt.io/qt-5/model-view-programming.html#model-subclassing-reference
class QtNodeTreeModel(QAbstractItemModel):
    """A concrete QItemModel for a tree of ``Node`` and ``Group`` objects.

    Key concepts and references:
        - Qt `Model/View Programming
          <https://doc.qt.io/qt-5/model-view-programming.html>`_
        - Qt `Model Subclassing Reference
          <https://doc.qt.io/qt-5/model-view-programming.html#model-subclassing-reference>`_
        - `Model Index <https://doc.qt.io/qt-5/model-view-programming.html#model-indexes>`_
        - `Simple Tree Model Example
          <https://doc.qt.io/qt-5/qtwidgets-itemviews-simpletreemodel-example.html>`_
    """

    _root: Group

    def __init__(self, root: Group, parent: QWidget = None):
        super().__init__(parent)
        self.setRoot(root)

    # ########## Reimplemented Public Qt Functions ##################

    def canDropMimeData(
        self,
        data: QMimeData,
        action: Qt.DropAction,
        row: int,
        column: int,
        parent: QModelIndex,
    ) -> bool:
        """Returns true if a model can accept a drop of the data.

        The default implementation only checks if data has at least one format in the list
        of mimeTypes() and if action is among the model's supportedDropActions().

        Reimplementing this function lets us test whether the data can be dropped at
        ``row``, ``column``, ``parent`` with ``action``.  Here, we just check that
        ``parent`` is a :class:`Group`.

        Returns
        -------
        bool
            Whether we can accept a drop of data at ``row``, ``column``, ``parent``
        """
        return self.getItem(parent).is_group()

    def columnCount(self, parent: QModelIndex) -> int:
        """Returns the number of columns for the children of the given ``parent``.

        In a tree, the number of columns is always 1.
        """
        return 1

    def data(self, index: QModelIndex, role: Qt.ItemDataRole):
        """Return data stored under ``role`` for the item at ``index``.

        A given class:`QModelIndex` can store multiple types of data, each with its own
        "ItemDataRole".
        """
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
        """Handles ``data`` from a drag and drop operation that ended with ``action``.

        The specified row, column and parent indicate the location of an item in the model
        where the operation ended. It is the responsibility of the model to complete the
        action at the correct location.

        Returns
        -------
        bool
            ``True`` if the ``data`` and ``action`` were handled by the model;
            otherwise returns ``False``.
        """
        if not data or action != Qt.MoveAction:
            return False
        if not data.hasFormat(self.mimeTypes()[0]):
            return False

        if isinstance(data, NodeMimeData):
            dest_idx = self.getItem(parent).index_from_root()
            dest_idx = dest_idx + (destRow,)
            moving_indices = data.node_indices()

            logger.debug(
                f"dropMimeData: indices {moving_indices} ➡ {dest_idx}"
            )

            if len(moving_indices) == 1:
                self._root.move(moving_indices[0], dest_idx)
            else:
                self._root.move_multiple(moving_indices, dest_idx)

        # XXX: returning False seems to go against the Qt docs instructions, but
        # If we return true, removeRows is called!?
        return False

    def flags(self, index: QModelIndex) -> Qt.ItemFlag | Qt.ItemFlags:
        """Returns the item flags for the given ``index``.

        This describes the properties of a given item in the model.  We set them to be
        editable, checkable, dragable, droppable, etc...
        If not a Group, we additional set ``Qt.ItemNeverHasChildren``

        See Qt::ItemFlags https://doc.qt.io/qt-5/qt.html#ItemFlag-enum
        """
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
        if self.getItem(index).is_group():
            return base_flags
        return base_flags | Qt.ItemNeverHasChildren

    def index(
        self, row: int, column: int = 0, parent: QModelIndex = QModelIndex()
    ) -> QModelIndex:
        """Return index of the item specified by ``row``, ``column`` and ``parent`` index.

        From Qt Docs: "When reimplementing this function in a subclass, call
        ``createIndex()`` to generate model indexes that other components can use to refer
        to items in your model.
        """
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        parentItem = self.getItem(parent)
        if parentItem.is_group():
            parentItem = cast(Group, parentItem)
            return self.createIndex(row, column, parentItem[row])
        return QModelIndex()

    def mimeData(self, indices: list[QModelIndex]) -> NodeMimeData:
        """Return an object containing serialized data corresponding to specified indexes.

        The format used to describe the encoded data is obtained from the mimeTypes()
        function. The implementation uses the default MIME type returned by the
        default implementation of mimeTypes(). If you reimplement mimeTypes() in your custom
        model to return more MIME types, reimplement this function to make use of them.
        """
        # If the list of indexes is empty, or there are no supported MIME types, nullptr is
        # returned rather than a serialized empty list.
        if not indices:
            return 0
        return NodeMimeData([self.getItem(i) for i in indices])

    def mimeTypes(self) -> list[str]:
        """Returns the list of allowed MIME types.

        By default, the built-in models and views use an internal MIME type:
        application/x-qabstractitemmodeldatalist.

        When implementing drag and drop support in a custom model, if you will return data
        in formats other than the default internal MIME type, reimplement this function to
        return your list of MIME types.

        If you reimplement this function in your custom model, you must also reimplement the
        member functions that call it: mimeData() and dropMimeData().

        Returns
        -------
        list of str
            MIME types allowed for drag & drop support
        """
        return ["application/x-tree-node", "text/plain"]

    def parent(self, index: QModelIndex) -> QModelIndex:
        """Returns the parent of the model item with the given ``inde``x.

        If the item has no parent, an invalid QModelIndex is returned.
        """
        if not index.isValid():
            return QModelIndex()  # null index

        parentItem = self.getItem(index).parent
        if parentItem is None or parentItem == self._root:
            return QModelIndex()

        # A common convention used in models that expose tree data structures is that only
        # items in the first column have children. So here,the column of the returned is 0.
        return self.createIndex(parentItem.index_in_parent(), 0, parentItem)

    def rowCount(self, parent: QModelIndex) -> int:
        """Returns the number of rows under the given parent.

        When the parent is valid it means that rowCount is returning the number of
        children of parent.
        """
        try:
            return len(self.getItem(parent))  # type:ignore
        except TypeError:
            return 0

    def supportedDropActions(self) -> Qt.DropActions:
        """Returns the drop actions supported by this model.

        The default implementation returns Qt::CopyAction. We re-implement to support only
        MoveAction. See also dropMimeData(), which must handles each operation.
        """
        return Qt.MoveAction

    # ########## New methods added for our model ##################

    def setRoot(self, root: Group):
        if not isinstance(root, Group):
            raise TypeError(
                "root node must be an instance of napari.utils.tree.Group"
            )
        current_root: Group | None = getattr(self, "_root", None)
        if root is current_root:
            return
        elif current_root is not None:
            disconnect_events(self._root.events, self)

        self._root = root
        self._root.events.removing.connect(self._on_begin_removing)
        self._root.events.removed.connect(self._on_end_remove)
        self._root.events.inserting.connect(self._on_begin_inserting)
        self._root.events.inserted.connect(self._on_end_insert)
        self._root.events.moving.connect(self._on_begin_moving)
        self._root.events.moved.connect(self._on_end_move)

    def _on_begin_removing(self, event):
        """Begins a row removal operation.

        See Qt documentation: https://doc.qt.io/qt-5/qabstractitemmodel.html#beginRemoveRows
        """
        par, idx = self._split_nested_index(event.index)
        self.beginRemoveRows(par, idx, idx)

    def _on_end_remove(self, e):
        self.endRemoveRows()

    def _on_begin_inserting(self, event):
        """Begins a row insertion operation.

        See Qt documentation: https://doc.qt.io/qt-5/qabstractitemmodel.html#beginInsertRows
        """
        par, idx = self._split_nested_index(event.index)
        self.beginInsertRows(par, idx, idx)

    def _on_end_insert(self, e):
        self.endInsertRows()

    def _on_begin_moving(self, event):
        """Begins a row move operation.

        See Qt documentation: https://doc.qt.io/qt-5/qabstractitemmodel.html#beginMoveRows
        """
        src_par, src_idx = self._split_nested_index(event.index)
        dest_par, dest_idx = self._split_nested_index(event.new_index)
        logger.debug(
            f"beginMoveRows({self.getItem(src_par).name}, {src_idx}, "
            f"{self.getItem(dest_par).name}, {dest_idx})"
        )

        self.beginMoveRows(src_par, src_idx, src_idx, dest_par, dest_idx)

    def _on_end_move(self, e):
        self.endMoveRows()

    def getItem(self, index: QModelIndex) -> Node:
        """Return ``Node`` object for a given `QModelIndex`.

        A null or invalid ``QModelIndex`` will return the root Node.
        """
        if index.isValid():
            item = index.internalPointer()
            if item is not None:
                return item
        return self._root

    def findIndex(self, obj: Node) -> QModelIndex:
        """Find the QModelIndex for a given ``Node`` object in the model."""
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

    def nestedIndex(self, nested_index: tuple[int, ...]) -> QModelIndex:
        """Return a QModelIndex for a given ``nested_index``."""
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

    def _split_nested_index(
        self, nested_index: int | tuple[int, ...]
    ) -> tuple[QModelIndex, int]:
        """Given a nested index, return (nested_parent_index, row)."""
        # TODO: split after using nestedIndex?
        if isinstance(nested_index, int):
            return QModelIndex(), nested_index
        par = QModelIndex()
        *_p, idx = nested_index
        for i in _p:
            par = self.index(i, 0, par)
        return par, idx


# TODO: cleanup stuff related to MIME formats and convenience methods
class NodeMimeData(QMimeData):
    def __init__(self, nodes: list[Node] = None):
        super().__init__()
        self.nodes = nodes or []
        if nodes:
            self.setData(
                "application/x-tree-node", pickle.dumps(self.node_indices())
            )
            self.setText(" ".join([node.name for node in nodes]))

    def formats(self) -> list[str]:
        return ["application/x-tree-node", "text/plain"]

    def node_indices(self) -> list[tuple[int, ...]]:
        return [node.index_from_root() for node in self.nodes]

    def node_names(self) -> list[str]:
        return [node.name for node in self.nodes]
