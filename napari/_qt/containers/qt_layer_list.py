from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QSortFilterProxyModel, Qt

from ...layers import Layer
from ...utils.translations import trans
from ._base_item_model import SortRole, _BaseEventedItemModel
from ._layer_delegate import LayerDelegate
from .qt_list_view import QtListView

if TYPE_CHECKING:
    from qtpy.QtGui import QKeyEvent
    from qtpy.QtWidgets import QWidget

    from ...components.layerlist import LayerList


class ReverseProxyModel(QSortFilterProxyModel):
    """Proxy Model that reverses the view order of a _BaseEventedItemModel."""

    def __init__(self, model: _BaseEventedItemModel) -> None:
        super().__init__()
        self.setSourceModel(model)
        self.setSortRole(SortRole)
        self.sort(0, Qt.SortOrder.DescendingOrder)

    def dropMimeData(self, data, action, destRow, col, parent):
        """Handle destination row for dropping with reversed indices."""
        row = 0 if destRow == -1 else self.sourceModel().rowCount() - destRow
        return self.sourceModel().dropMimeData(data, action, row, col, parent)


class QtLayerList(QtListView[Layer]):
    """QItemView subclass specialized for the LayerList.

    This is as mostly for targetting with QSS, applying the delegate and
    reversing the view with ReverseProxyModel.
    """

    def __init__(self, root: LayerList, parent: QWidget = None):
        super().__init__(root, parent)
        self.setItemDelegate(LayerDelegate())
        self.setToolTip(trans._('Layer list'))
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)

        # This reverses the order of the items in the view,
        # so items at the end of the list are at the top.
        self.setModel(ReverseProxyModel(self.model()))

    def keyPressEvent(self, e: QKeyEvent) -> None:
        """Override Qt event to pass events to the viewer."""
        if e.key() != Qt.Key.Key_Space:
            super().keyPressEvent(e)
        if e.key() not in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            e.ignore()  # pass key events up to viewer
