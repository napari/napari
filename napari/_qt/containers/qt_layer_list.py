from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QSortFilterProxyModel, Qt

from ...layers import Layer
from ...utils.translations import trans
from ._layer_delegate import LayerDelegate
from .qt_layer_model import SortRole
from .qt_list_view import QtListView

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from ...components.layerlist import LayerList


class QtLayerList(QtListView[Layer]):
    """QItemView subclass specialized for the LayerList.

    This is as mostly for targetting with QSS, and applying the delegate
    """

    def __init__(self, root: LayerList, parent: QWidget = None):
        super().__init__(root, parent)
        self.setItemDelegate(LayerDelegate())
        self.setToolTip(trans._('Layer list'))

        # This reverses the order of the items in the view, so items at the
        # end of the list are on top.  See also the couple mimeData
        # overrides in QtLayerListModel
        self._proxy_model = QSortFilterProxyModel()
        self._proxy_model.setSourceModel(self.model())
        self._proxy_model.setSortRole(SortRole)
        self.setModel(self._proxy_model)
        self._proxy_model.sort(0, Qt.DescendingOrder)
