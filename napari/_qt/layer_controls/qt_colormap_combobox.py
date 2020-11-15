from qtpy.QtCore import QModelIndex, QRect, Qt
from qtpy.QtGui import QImage, QPainter, QPen
from qtpy.QtWidgets import (
    QComboBox,
    QListView,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from ...utils.colormaps import ensure_colormap, make_colorbar

COLORMAP_WIDTH = 50
TEXT_WIDTH = 100
ENTRY_HEIGHT = 18
BORDER_WIDTH = 2
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
        rect = QRect(
            style.rect.x(),
            style.rect.y() + PADDING,
            style.rect.width() - TEXT_WIDTH,
            style.rect.height() - 2 * PADDING,
        )
        rect2 = QRect(
            style.rect.width() - TEXT_WIDTH + 10,
            style.rect.y() + PADDING,
            style.rect.width(),
            style.rect.height() - 2 * PADDING,
        )
        cbar = make_colorbar(ensure_colormap(model.data()), (18, 100))
        image = QImage(
            cbar, cbar.shape[1], cbar.shape[0], QImage.Format_RGBA8888,
        )
        painter.drawImage(rect, image)
        painter.drawText(rect2, Qt.AlignCenter & Qt.AlignVCenter, model.data())
        if int(style.state & QStyle.State_HasFocus):
            painter.save()
            pen = QPen()
            pen.setWidth(BORDER_WIDTH)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.restore()

    def sizeHint(self, style: QStyleOptionViewItem, model: QModelIndex):
        res = super().sizeHint(style, model)
        # print(res)
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
