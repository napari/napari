from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qtpy.QtCore import QModelIndex, QSize, Qt
from qtpy.QtGui import QImage

from napari import current_viewer
from napari._qt.containers.qt_list_model import QtListModel
from napari.layers import Layer
from napari.settings import get_settings
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.utils.events import Event

ThumbnailRole = Qt.ItemDataRole.UserRole + 2
LoadedRole = Qt.ItemDataRole.UserRole + 3


class QtLayerListModel(QtListModel[Layer]):
    def data(
        self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        """Return data stored under ``role`` for the item at ``index``."""
        if not index.isValid():
            return None
        item = self.getItem(index)
        layer = item if isinstance(item, Layer) else item[0]
        viewer = current_viewer()
        layer_loaded = layer.loaded
        # Playback with async slicing causes flickering between the thumbnail
        # and loading animation in some cases due quick changes in the loaded
        # state, so report as unloaded in that case to avoid that.
        if get_settings().experimental.async_ and viewer:
            viewer_playing = viewer.window._qt_viewer.dims.is_playing
            layer_loaded = layer.loaded and not viewer_playing
        if role == Qt.ItemDataRole.DisplayRole:  # used for item text
            return layer.name
        if role == Qt.ItemDataRole.TextAlignmentRole:  # alignment of the text
            return Qt.AlignmentFlag.AlignCenter
        if role == Qt.ItemDataRole.EditRole:
            # used to populate line edit when editing
            return layer.name
        if role == Qt.ItemDataRole.ToolTipRole:  # for tooltip
            layer_source_info = layer.get_source_str()
            if layer_loaded:
                return layer_source_info
            return trans._('{source} (loading)', source=layer_source_info)
        if (
            role == Qt.ItemDataRole.CheckStateRole
        ):  # the "checked" state of this item
            return (
                Qt.CheckState.Checked
                if layer.visible
                else Qt.CheckState.Unchecked
            )
        if role == Qt.ItemDataRole.SizeHintRole:  # determines size of item
            return QSize(200, 34)
        if role == ThumbnailRole:  # return the thumbnail
            thumbnail = layer.thumbnail
            return QImage(
                thumbnail.data,
                thumbnail.shape[1],
                thumbnail.shape[0],
                QImage.Format.Format_RGBA8888,
            )
        if role == LoadedRole:
            return layer_loaded
        # normally you'd put the icon in DecorationRole, but we do that in the
        # # LayerDelegate which is aware of the theme.
        # if role == Qt.ItemDataRole.DecorationRole:  # icon to show
        #     pass
        return super().data(index, role)

    def setData(
        self,
        index: QModelIndex,
        value: Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        item = self.getItem(index)
        layer = item if isinstance(item, Layer) else item[0]
        if role == Qt.ItemDataRole.CheckStateRole:
            # The item model stores a Qt.CheckState enum value that can be
            # partially checked, but we only use the unchecked and checked
            # to correspond to the layer's visibility.
            # https://doc.qt.io/qt-5/qt.html#CheckState-enum
            layer.visible = Qt.CheckState(value) == Qt.CheckState.Checked
        elif role == Qt.ItemDataRole.EditRole:
            layer.name = value
            role = Qt.ItemDataRole.DisplayRole
        else:
            return super().setData(index, value, role=role)

        self.dataChanged.emit(index, index, [role])
        return True

    def all_loaded(self) -> bool:
        """Return if all the layers are loaded."""
        return all(
            self.index(row, 0).data(LoadedRole)
            for row in range(self.rowCount())
        )

    def _process_event(self, event: Event) -> None:
        # The model needs to emit `dataChanged` whenever data has changed
        # for a given index, so that views can update themselves.
        # Here we convert native events to the dataChanged signal.
        if not hasattr(event, 'index'):
            return
        role = {
            'thumbnail': ThumbnailRole,
            'visible': Qt.ItemDataRole.CheckStateRole,
            'name': Qt.ItemDataRole.DisplayRole,
            'loaded': LoadedRole,
        }.get(event.type)
        roles = [role] if role is not None else []
        row = self.index(event.index)
        self.dataChanged.emit(row, row, roles)
