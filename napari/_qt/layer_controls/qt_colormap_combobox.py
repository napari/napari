from qtpy.QtCore import QModelIndex, QRect
from qtpy.QtGui import QImage, QPainter
from qtpy.QtWidgets import (
    QComboBox,
    QListView,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from ...utils.colormaps import (
    display_name_to_name,
    ensure_colormap,
    make_colorbar,
)

COLORMAP_WIDTH = 50
TEXT_WIDTH = 130
ENTRY_HEIGHT = 20
PADDING = 1


class ColorStyledDelegate(QStyledItemDelegate):
    """Class for paint :py:class:`~.ColorComboBox` elements when list trigger

    Parameters
    ----------
    base_height : int
        Height of single list element.
    color_dict: dict
        Dict mapping name to colors.
    """

    def __init__(self, base_height: int, **kwargs):
        super().__init__(**kwargs)
        self.base_height = base_height

    def paint(
        self,
        painter: QPainter,
        style: QStyleOptionViewItem,
        model: QModelIndex,
    ):
        style2 = QStyleOptionViewItem(style)

        cbar_rect = QRect(
            style.rect.x(),
            style.rect.y() + PADDING,
            style.rect.width() - TEXT_WIDTH,
            style.rect.height() - 2 * PADDING,
        )
        text_rect = QRect(
            style.rect.width() - TEXT_WIDTH,
            style.rect.y() + PADDING,
            style.rect.width(),
            style.rect.height() - 2 * PADDING,
        )
        style2.rect = text_rect
        super().paint(painter, style2, model)
        name = display_name_to_name(model.data())
        cbar = make_colorbar(ensure_colormap(name), (18, 100))
        image = QImage(
            cbar,
            cbar.shape[1],
            cbar.shape[0],
            QImage.Format_RGBA8888,
        )
        painter.drawImage(cbar_rect, image)

    def sizeHint(self, style: QStyleOptionViewItem, model: QModelIndex):
        res = super().sizeHint(style, model)
        res.setHeight(self.base_height)
        res.setWidth(max(500, res.width()))
        return res


class QtColormapComboBox(QComboBox):
    """Combobox showing colormaps

    Parameters
    ----------
    parent : QWidget
        Parent widget of comboxbox.
    """

    def __init__(self, parent):
        super().__init__(parent)
        view = QListView()
        view.setMinimumWidth(COLORMAP_WIDTH + TEXT_WIDTH)
        view.setItemDelegate(ColorStyledDelegate(ENTRY_HEIGHT))
        self.setView(view)
