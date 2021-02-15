"""

General rendering flow:

1. ``QtLayerTreeView`` needs to display or edit an index in the model...
2. It gets the ``itemDelegate``
    a. A custom delegate can be set with ``setItemDelegate``
    b. ``QStyledItemDelegate`` is the default delegate for all Qt item views,
       and is installed upon them when they are created.
3. ``itemDelegate.paint`` is called on the index being displayed
4. Each index in the model has various data elements (i.e. name, image, etc..),
   each of which has a "data role".  A model should return the appropriate data
   for each role by reimplementing ``QAbstractItemModel.data``.
    a. `QStyledItemDelegate` implements display and editing for the most common
       datatypes expected by users, including booleans, integers, and strings.
    b. If the delegate does not support painting of the data types you need or
       you want to customize the drawing of items, you need to subclass
       ``QStyledItemDelegate``, and reimplement ``paint()`` and possibly
       ``sizeHint()``.
    c. When reimplementing ``paint()``, one typically handles the datatypes
       they would like to draw and uses the superclass implementation for other
       types.
5. The default implementation of ``QStyledItemDelegate.paint`` paints the item
   using the view's ``QStyle`` (which is, by default, an OS specific style...
   but see ``QCommonStyle`` for a generic implementation)
    a. It is also possible to override the view's style, using either a
    subclass of ``QCommonStyle``, for a platform-independent look and feel, or
    ``QProxyStyle``, which let's you override only certain stylistic elements
    on any platform, falling back to the system default otherwise.
    b. ``QStyle`` paints various elements using methods like ``drawPrimitive``
       and ``drawControl``.  These can be overridden for very fine control.
6. It has hard to use stylesheets with custom ``QStyles``... but it's possible
   to style sub-controls in ``QAbstractItemView`` (such as ``QTreeView``):
   https://doc.qt.io/qt-5/stylesheet-reference.html#list-of-sub-controls

"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtGui import QBrush, QIcon
from qtpy.QtCore import (
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QRect,
    QSize,
    Qt,
    QVariant,
)
from qtpy.QtGui import QImage, QPainter, QPalette, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QProxyStyle,
    QStyle,
    QStyledItemDelegate,
    QStyleOption,
    QStyleOptionViewItem,
    QWidget,
)

from .qt_tree_model import QtNodeTreeModel
from .qt_tree_view import QtNodeTreeView

if TYPE_CHECKING:
    from ...layers import Layer
    from ...layers.layergroup import LayerGroup
    from ...utils.events.containers._nested_list import MaybeNestedIndex


class QtLayerTreeModel(QtNodeTreeModel):
    LayerRole = Qt.UserRole
    ThumbnailRole = Qt.UserRole + 1

    def __init__(self, root: LayerGroup, parent: QWidget = None):
        super().__init__(root, parent)
        self.data
        self.setRoot(root)

    def getItem(self, index: QModelIndex) -> Layer:
        # TODO: this ignore should be fixed by making QtNodeTreeModel Generic.
        return super().getItem(index)  # type: ignore

    def data(self, index: QModelIndex, role: Qt.ItemDataRole):
        """Return data stored under ``role`` for the item at ``index``."""
        layer = self.getItem(index)
        if role == Qt.DisplayRole:
            return layer.name
        if role == Qt.EditRole:
            return layer.name
        if role == Qt.ToolTipRole:
            return layer.name
        if role == Qt.CheckStateRole:
            return layer.visible
        if role == Qt.SizeHintRole:
            h = 36 if layer.is_group() else 42
            return QSize(228, h)
        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
        if role == QtLayerTreeModel.LayerRole:
            return self.getItem(index)
        if role == QtLayerTreeModel.ThumbnailRole:
            thumbnail = layer.thumbnail
            return QImage(
                thumbnail,
                thumbnail.shape[1],
                thumbnail.shape[0],
                QImage.Format_RGBA8888,
            )
        if role == Qt.DecorationRole:
            pass
        return QVariant()

    def setData(self, index: QModelIndex, value, role: int) -> bool:
        if role == Qt.CheckStateRole:
            self.getItem(index).visible = value
        elif role == Qt.EditRole:
            self.getItem(index).name = value
        else:
            return super().setData(index, value, role=role)

        self.dataChanged.emit(index, index, [role])
        return True


class QtLayerTreeView(QtNodeTreeView):
    def __init__(self, root: LayerGroup = None, parent: QWidget = None):
        super().__init__(root, parent)
        self.setItemDelegate(LayerDelegate())
        self.setAnimated(True)
        self.setAutoExpandDelay(300)
        # self.setStyle(QtLayerTreeStyle())
        # self.setStyleSheet(
        #     """
        # QtLayerTreeView::branch { background-color: 'blue'; }
        # """
        # )

    def setRoot(self, root: LayerGroup):
        self.setModel(QtLayerTreeModel(root, self))
        self.model().rowsRemoved.connect(self._redecorate_root)
        self.model().rowsInserted.connect(self._redecorate_root)
        self._redecorate_root()
        root.events.selection.connect(lambda e: self._select(e.index, e.value))
        # initialize selection model
        for child in root.traverse():
            selected = getattr(child, 'selected', False)
            self._select(child.index_from_root(), selected)

    def _redecorate_root(self, parent=None, *_):
        """Add a branch/arrow column only if there are Groups in the root."""
        if not parent or not parent.isValid():
            self.setRootIsDecorated(self.model().hasGroups())

    def selectionChanged(
        self, selected: QItemSelection, deselected: QItemSelection
    ):
        model = self.model()
        for q_index in selected.indexes():
            model.getItem(q_index).selected = True
        for q_index in deselected.indexes():
            model.getItem(q_index).selected = False
        return super().selectionChanged(selected, deselected)

    def _select(self, nested_index: MaybeNestedIndex, selected=True):
        idx = self.model().nestedIndex(nested_index)
        if nested_index == () or not idx.isValid():
            return
        s = getattr(QItemSelectionModel, 'Select' if selected else 'Deselect')
        # TODO: figure out bug on pop(0) in examples/layer_tree
        self.selectionModel().select(idx, s)
        print("select", nested_index, selected)

    # def reset(self):
    #     print("reset called")
    #     self.setRootIsDecorated(False)
    #     super().reset()

    # def rowsInserted(
    #     self, parent: QtCore.QModelIndex, start: int, end: int
    # ) -> None:
    #     # if not parent.isValid() and self.model().nestedIndex(start)
    #     #     self.setRootIsDecorated(True)
    #     print("rows inserted", parent.isValid(), parent.parent().isValid())
    #     return super().rowsInserted(parent, start, end)

    # def dataChanged(
    #     self,
    #     topLeft: QtCore.QModelIndex,
    #     bottomRight: QtCore.QModelIndex,
    #     roles,
    # ) -> None:
    #     print("data changed from ", topLeft.row(), "to", bottomRight.row())
    #     return super().dataChanged(topLeft, bottomRight, roles=roles)

    # def drawBranches(
    #     self,
    #     painter: QPainter,
    #     rect: QRect,
    #     index: QModelIndex,
    # ) -> None:
    #     """Responsible for drawing the arrows."""
    #     # rect.setWidth(rect.width() * 3 // 4)
    #     print('b ', index.row(), rect)
    #     return super().drawBranches(painter, rect, index)


class LayerDelegate(QStyledItemDelegate):
    pass

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ):
        thumb_rect = option.rect.translated(4, 0)  # figure out why 4
        # MAGICNUMBER: 4 comes from the margin applied in the stylesheet to
        # QtLayerTreeView::item
        h = index.data(Qt.SizeHintRole).height() - 4
        thumb_rect.setWidth(h)
        thumb_rect.setHeight(h)
        option.rect.translate(h, -2)
        option.rect.setWidth(option.rect.width() - h)
        # paint the standard itemView
        super().paint(painter, option, index)
        # paint the thumbnail
        image = index.data(QtLayerTreeModel.ThumbnailRole)
        pen = painter.pen()
        brush = painter.brush()
        painter.setBrush(QBrush(image))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(thumb_rect, 2, 2)
        painter.setPen(pen)
        painter.setBrush(brush)
        # painter.drawPixmap(thumb_rect)

    # def createEditor(
    #     self,
    #     parent: QWidget,
    #     option: QStyleOptionViewItem,
    #     index: QModelIndex,
    # ) -> QWidget:
    #     print("createEditor")
    #     return super().createEditor(parent, option, index)

    def updateEditorGeometry(
        self,
        editor: QWidget,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        h = index.data(Qt.SizeHintRole).height()
        option.rect.translate(h, 0)
        option.rect.setWidth(option.rect.width() - h)
        return super().updateEditorGeometry(editor, option, index)


# METRICS = {
#     QStyle.PM_IndicatorWidth: 16,
#     QStyle.PM_IndicatorHeight: 16,
#     QStyle.PM_FocusFrameHMargin: 2,
# }


# class QtLayerTreeStyle(QProxyStyle):
#     """A QProxyStyle wraps a QStyle (usually the default system style) for the
#     purpose of dynamically overriding painting or other style behavior.
#     """

#     def pixelMetric(
#         self,
#         metric: QStyle.PixelMetric,
#         option: QStyleOption = None,
#         widget: QWidget = None,
#     ) -> int:
#         if metric in METRICS:
#             return METRICS[metric]
#         return super().pixelMetric(metric, option=option, widget=widget)

#     def drawPrimitive(
#         self,
#         pe: QStyle.PrimitiveElement,
#         opt: QStyleOption,
#         p: QPainter,
#         widget: QWidget,
#     ) -> None:
#         if pe == QStyle.PE_IndicatorItemViewItemCheck:
#             print('pe  ', opt.rect)
#             if opt.state & QStyle.State_On:
#                 p.drawImage(opt.rect, QImage(":/themes/dark/visibility.svg"))
#             elif opt.state & QStyle.State_Off:
#                 p.drawImage(
#                     opt.rect, QImage(":/themes/dark/visibility_off.svg")
#                 )
#             elif opt.state & QStyle.State_NoChange:
#                 p.setPen(opt.palette.color(QPalette.Dark))
#                 p.fillRect(opt.rect, opt.palette.brush(QPalette.Background))
#                 p.drawRect(opt.rect)
#             return
#         if pe == QStyle.PE_IndicatorBranch:
#             pass
#         return super().drawPrimitive(pe, opt, p, widget=widget)
