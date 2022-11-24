import typing

from qtpy.QtCore import QModelIndex, QSize, Qt
from qtpy.QtGui import QImage

from napari._qt.containers.qt_list_model import QtListModel
from napari.layers import Layer
from napari.utils.translations import trans

ThumbnailRole = Qt.UserRole + 2
ErroredRole = Qt.UserRole + 4


class QtLayerListModel(QtListModel[Layer]):
    def data(self, index: QModelIndex, role: Qt.ItemDataRole):
        """Return data stored under ``role`` for the item at ``index``."""
        if not index.isValid():
            return None
        layer = self.getItem(index)
        if role == Qt.ItemDataRole.DisplayRole:  # used for item text
            return layer.name
        if role == Qt.ItemDataRole.TextAlignmentRole:  # alignment of the text
            return Qt.AlignCenter
        if role == Qt.ItemDataRole.EditRole:
            # used to populate line edit when editing
            return layer.name
        if role == Qt.ItemDataRole.ToolTipRole:  # for tooltip
            if layer.errored:
                failure_message = trans._("Layer failed loading")
                options_message = trans._("Try refreshing or reloading layer")
                return (
                    "<p style='white-space:pre; text-align: center;'>"
                    f"<b>{failure_message}</b>"
                    "<br>"
                    f"{options_message}"
                    "</p>"
                )
            else:
                return layer.get_source_str()
        if (
            role == Qt.ItemDataRole.CheckStateRole
        ):  # the "checked" state of this item
            return Qt.Checked if layer.visible else Qt.Unchecked
        if role == Qt.ItemDataRole.SizeHintRole:  # determines size of item
            return QSize(200, 34)
        if role == ThumbnailRole:  # return the thumbnail
            thumbnail = layer.thumbnail
            return QImage(
                thumbnail,
                thumbnail.shape[1],
                thumbnail.shape[0],
                QImage.Format_RGBA8888,
            )
        if role == ErroredRole:
            return layer.errored
        # normally you'd put the icon in DecorationRole, but we do that in the
        # # LayerDelegate which is aware of the theme.
        # if role == Qt.ItemDataRole.DecorationRole:  # icon to show
        #     pass
        return super().data(index, role)

    def setData(
        self,
        index: QModelIndex,
        value: typing.Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if role == Qt.ItemDataRole.CheckStateRole:
            self.getItem(index).visible = value
        elif role == Qt.ItemDataRole.EditRole:
            self.getItem(index).name = value
            role = Qt.ItemDataRole.DisplayRole
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
            'visible': Qt.ItemDataRole.CheckStateRole,
            'name': Qt.ItemDataRole.DisplayRole,
            'errored': ErroredRole,
        }.get(event.type)
        roles = [role] if role is not None else []
        row = self.index(event.index)
        self.dataChanged.emit(row, row, roles)
