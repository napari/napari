from typing import List

from qtpy.QtCore import QModelIndex, QSize, Qt
from qtpy.QtGui import QImage

from ...layers import Layer
from .qt_list_model import QtListModel

LayerRole = Qt.UserRole
ThumbnailRole = Qt.UserRole + 1
SortRole = Qt.UserRole + 2


class QtLayerListModel(QtListModel[Layer]):
    def data(self, index: QModelIndex, role: Qt.ItemDataRole):
        """Return data stored under ``role`` for the item at ``index``."""
        layer = self.getItem(index)
        if role == Qt.DisplayRole:  # used for item text
            return layer.name
        if role == SortRole:  # used for item text
            return index.row()
        if role == Qt.TextAlignmentRole:  # alignment of the text
            return Qt.AlignCenter
        if role == Qt.EditRole:  # used to populate line edit when editing
            return layer.name
        if role == Qt.ToolTipRole:  # for tooltip
            return layer.name
        if role == Qt.CheckStateRole:  # the "checked" state of this item
            return Qt.Checked if layer.visible else Qt.Unchecked
        if role == Qt.SizeHintRole:  # determines size of item
            h = 38
            return QSize(228, h)
        if role == LayerRole:  # custom role: return the layer
            return self.getItem(index)
        if role == ThumbnailRole:  # return the thumbnail
            thumbnail = layer.thumbnail
            return QImage(
                thumbnail,
                thumbnail.shape[1],
                thumbnail.shape[0],
                QImage.Format_RGBA8888,
            )
        # normally you'd put the icon in DecorationRole, but we do that in the
        # # LayerDelegate which is aware of the theme.
        # if role == Qt.DecorationRole:  # icon to show
        #     pass
        return None

    def setData(self, index: QModelIndex, value, role: int) -> bool:
        if role == Qt.CheckStateRole:
            self.getItem(index).visible = value
        elif role == Qt.EditRole:
            self.getItem(index).name = value
            role = Qt.DisplayRole
        else:
            return super().setData(index, value, role=role)

        self.dataChanged.emit(index, index, [role])
        return True

    def _process_event(self, event):
        # The model needs to emit `dataChanged` whenever data has changed
        # for a given index, so that views can update themselves.
        # Here we convert native events to the dataChanged signal.
        if not hasattr(event, 'index'):
            return
        role = {
            'thumbnail': ThumbnailRole,
            'visible': Qt.CheckStateRole,
            'name': Qt.DisplayRole,
        }.get(event.type, None)
        roles = [role] if role is not None else []
        top = self.index(event.index)
        bot = self.index(event.index + 1)
        self.dataChanged.emit(top, bot, roles)

    # TODO:
    # These two overrides are here to handle drag/drop events because the
    # view is reversed using `QSortFilterProxyModel` in the `QtLayerList`.
    # It *should* be possible to achieve the reversal entirely on the view side
    # without the model knowing anything about it, but after a day of tinkering
    # with QAbstractProxyModel and subclasses, I haven't yet figured it out.

    def mimeData(self, indices: List[QModelIndex]):
        """Return an object containing serialized data from `indices`."""
        data = super().mimeData(indices)
        data.indices = tuple(reversed(data.indices))
        return data

    def dropMimeData(self, data, action, destRow, col, parent):
        return super().dropMimeData(data, action, destRow + 1, col, parent)
