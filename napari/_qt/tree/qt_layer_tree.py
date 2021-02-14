from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import (
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QSize,
    Qt,
)
from qtpy.QtGui import QImage, QPainter, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QCommonStyle,
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
    def __init__(self, root: LayerGroup, parent: QWidget = None):
        super().__init__(root, parent)
        self.setRoot(root)

    def getItem(self, index: QModelIndex) -> Layer:
        # TODO: this ignore should be fixed by making QtNodeTreeModel Generic.
        return super().getItem(index)  # type: ignore

    def data(self, index: QModelIndex, role: Qt.ItemDataRole):
        """Return data stored under ``role`` for the item at ``index``."""
        layer = self.getItem(index)
        if role == Qt.DisplayRole:
            return layer.name
        if role == Qt.ToolTipRole:
            return layer.name
        if role == Qt.CheckStateRole:
            return layer.visible
        if role == Qt.SizeHintRole:
            h = 30 if layer.is_group() else 38
            return QSize(228, h)
        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
        if role == Qt.UserRole:
            return self.getItem(index)
        if role == Qt.DecorationRole:
            thumbnail = layer.thumbnail
            image = QImage(
                thumbnail,
                thumbnail.shape[1],
                thumbnail.shape[0],
                QImage.Format_RGBA8888,
            )
            return QPixmap.fromImage(image)
        return None

    def setData(self, index: QModelIndex, value, role: int) -> bool:
        if role == Qt.CheckStateRole:
            self.getItem(index).visible = value
        elif role == Qt.EditRole:
            self.getItem(index).name = value
        else:
            return super().setData(index, value, role=role)

        self.dataChanged.emit(index, index, [role])
        return True


class LayerDelegate(QStyledItemDelegate):
    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ):
        opt = option
        self.initStyleOption(opt, index)
        w = opt.widget
        p = painter
        style: QStyle = w.style() if w else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, option, p, w)
        # style.drawPrimitive(QStyle.PE_IndicatorItemViewItemCheck, opt, p, w)

        # checkRect = style.subElementRect(
        #     QStyle.SE_ItemViewItemCheckIndicator, opt, w
        # )
        # iconRect = style.subElementRect(
        #     QStyle.SE_ItemViewItemDecoration, opt, w
        # )
        # textRect = style.subElementRect(QStyle.SE_ItemViewItemText, opt, w)

        # layer = index.internalPointer()
        # super().paint(painter, option, index)


class QtLayerTreeStyle(QCommonStyle):
    def drawPrimitive(
        self,
        pe: QStyle.PrimitiveElement,
        opt: QStyleOption,
        p: QPainter,
        widget: QWidget,
    ) -> None:
        if pe == QStyle.PE_IndicatorItemViewItemCheck:
            img = (
                'visibility'
                if (opt.state & QStyle.State_On)
                else 'visibility_off'
            )
            p.drawImage(opt.rect, QImage(f":/themes/dark/{img}.svg"))
            # if opt.state & QStyle.State_NoChange:
            #     p.setPen(opt.palette.windowText().color())
            #     p.fillRect(opt.rect, opt.palette.brush(QPalette.Button))
            #     p.drawRect(opt.rect)
            #     p.drawLine(opt.rect.topLeft(), opt.rect.bottomRight())
            # else:
            #     super().drawPrimitive(pe, opt, p, widget)
            return
        if pe == QStyle.PE_IndicatorBranch:
            pass
        return super().drawPrimitive(pe, opt, p, widget=widget)


class QtLayerTreeView(QtNodeTreeView):
    def __init__(self, root: LayerGroup = None, parent: QWidget = None):
        super().__init__(root, parent)
        self.setItemDelegate(LayerDelegate())
        self.setStyle(QtLayerTreeStyle())

    def setRoot(self, root: LayerGroup):
        self.setModel(QtLayerTreeModel(root, self))
        root.events.selection.connect(lambda e: self._select(e.index, e.value))
        # initialize selection model
        for child in root.traverse():
            selected = getattr(child, 'selected', False)
            self._select(child.index_from_root(), selected)

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
        s = getattr(QItemSelectionModel, 'Select' if selected else 'Deselect')
        self.selectionModel().select(idx, s)

    # def drawBranches(
    #     self,
    #     painter: QPainter,
    #     rect: QRect,
    #     index: QModelIndex,
    # ) -> None:
    #     """Responsible for drawing the arrows."""
    #     return super().drawBranches(painter, rect, index)
