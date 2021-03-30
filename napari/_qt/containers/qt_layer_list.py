from typing import TYPE_CHECKING

from qtpy.QtGui import QFont
from qtpy.QtWidgets import QStyleOptionViewItem, QWidget

from ...layers import Layer
from ...utils import config
from ...utils.translations import trans
from ._layer_delegate import LayerDelegate
from .qt_layer_model import QtLayerListModel
from .qt_list_view import QtListView

if TYPE_CHECKING:
    from ...components.layerlist import LayerList

# translations from the old qt layer list
# tb.setToolTip(trans._('Layer thumbnail'))
# cb.setToolTip(trans._('Layer visibility'))
# ltb.setToolTip(trans._('Layer type'))
# msg = trans._('Click to select\nDrag to rearrange')


class QtLayerList(QtListView[Layer]):
    _list: 'LayerList'
    model_class = QtLayerListModel

    def __init__(self, root: 'LayerList', parent: QWidget = None):
        super().__init__(root, parent)
        self.setItemDelegate(LayerDelegate())
        fnt = QFont()
        fnt.setPixelSize(12)
        self.setFont(fnt)
        self.setToolTip(trans._('Layer list'))

        if config.async_loading:
            from ..experimental.qt_chunk_receiver import QtChunkReceiver

            # The QtChunkReceiver object allows the ChunkLoader to pass newly
            # loaded chunks to the layers that requested them.
            self.chunk_receiver = QtChunkReceiver(self)
        else:
            self.chunk_receiver = None

    def viewOptions(self) -> QStyleOptionViewItem:
        options = super().viewOptions()
        options.decorationPosition = options.Right
        return options

    @property
    def layers(self):
        # legacy API
        return self._list
