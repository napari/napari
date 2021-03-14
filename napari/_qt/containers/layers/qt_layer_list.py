from typing import TYPE_CHECKING

from qtpy.QtGui import QFont
from qtpy.QtWidgets import QStyleOptionViewItem, QWidget

from ....layers import Layer
from ...containers.list import QtListView
from ._layer_model import QtLayerListModel
from .delegate import LayerDelegate

if TYPE_CHECKING:

    from ....components.layerlist import LayerList


class QtLayerList(QtListView[Layer]):
    _list: 'LayerList'
    model_class = QtLayerListModel

    def __init__(self, root: 'LayerList', parent: QWidget = None):
        super().__init__(root, parent)
        self.setItemDelegate(LayerDelegate())
        fnt = QFont()
        fnt.setPixelSize(12)
        self.setFont(fnt)

    def viewOptions(self) -> QStyleOptionViewItem:
        options = super().viewOptions()
        options.decorationPosition = options.Right
        return options
