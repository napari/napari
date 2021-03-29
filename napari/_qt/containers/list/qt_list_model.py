import logging
import pickle
from typing import List, Optional

from qtpy.QtCore import QMimeData, QModelIndex, Qt

from .._base_item_model import _BaseItemModel

logger = logging.getLogger(__name__)
ListIndexMIMEType = "application/x-list-index"


class QtListModel(_BaseItemModel):
    def parent(self, index):
        return QModelIndex()

    def mimeTypes(self) -> List[str]:
        """Returns the list of allowed MIME types.

        When implementing drag and drop support in a custom model, if you will
        return data in formats other than the default internal MIME type,
        reimplement this function to return your list of MIME types.
        """
        return [ListIndexMIMEType, "text/plain"]

    def dropMimeData(
        self,
        data: QMimeData,
        action: Qt.DropAction,
        destRow: int,
        col: int,
        parent: QModelIndex,
    ) -> bool:
        """Handles `data` from a drag and drop operation ending with `action`.

        The specified row, column and parent indicate the location of an item
        in the model where the operation ended. It is the responsibility of the
        model to complete the action at the correct location.

        Returns
        -------
        bool ``True`` if the ``data`` and ``action`` were handled by the model;
            otherwise returns ``False``.
        """
        if not data or action != Qt.MoveAction:
            return False
        if not data.hasFormat(self.mimeTypes()[0]):
            return False

        if isinstance(data, ItemMimeData):
            moving_indices = data.indices

            logger.debug(f"dropMimeData: indices {moving_indices} ➡ {destRow}")

            if len(moving_indices) == 1:
                return self._root.move(moving_indices[0], destRow)
            else:
                return bool(self._root.move_multiple(moving_indices, destRow))
        return False

    def mimeData(self, indices: List[QModelIndex]) -> Optional['QMimeData']:
        """Return an object containing serialized data from `indices`.

        If the list of indexes is empty, or there are no supported MIME types,
        None is returned rather than a serialized empty list.
        """
        if not indices:
            return None
        items, indices = zip(*[(self.getItem(i), i.row()) for i in indices])
        return ItemMimeData(items, indices)


class ItemMimeData(QMimeData):
    """Custom MimeData to hold list indices."""

    def __init__(self, items, indices):
        super().__init__()
        self.items = items
        self.indices = indices
        if items:
            self.setData(ListIndexMIMEType, pickle.dumps(self.indices))
            self.setText(" ".join(str(item) for item in items))

    def formats(self) -> List[str]:
        return [ListIndexMIMEType, "text/plain"]
