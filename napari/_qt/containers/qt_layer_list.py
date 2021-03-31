from __future__ import annotations

from typing import TYPE_CHECKING

from ...layers import Layer
from ...utils.translations import trans
from ._layer_delegate import LayerDelegate
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
