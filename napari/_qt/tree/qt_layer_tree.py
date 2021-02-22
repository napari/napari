"""

General rendering flow:

1. ``QtLayerTreeView`` needs to display or edit an index in the model...
2. It gets the ``itemDelegate``
    a. A custom delegate can be set with ``setItemDelegate``
    b. ``QStyledItemDelegate`` is the default delegate for all Qt item views,
       and is installed upon them when they are created.
3. ``itemDelegate.paint`` is called on the index being displayed
4. Each index in the model has various data elements (i.e. name, image, etc..),
   each of which has a "data role".  A model should return the appropriate data
   for each role by reimplementing ``QAbstractItemModel.data``.
    a. `QStyledItemDelegate` implements display and editing for the most common
       datatypes expected by users, including booleans, integers, and strings.
    b. If the delegate does not support painting of the data types you need or
       you want to customize the drawing of items, you need to subclass
       ``QStyledItemDelegate``, and reimplement ``paint()`` and possibly
       ``sizeHint()``.
    c. When reimplementing ``paint()``, one typically handles the datatypes
       they would like to draw and uses the superclass implementation for other
       types.
5. The default implementation of ``QStyledItemDelegate.paint`` paints the item
   using the view's ``QStyle`` (which is, by default, an OS specific style...
   but see ``QCommonStyle`` for a generic implementation)
    a. It is also possible to override the view's style, using either a
    subclass of ``QCommonStyle``, for a platform-independent look and feel, or
    ``QProxyStyle``, which let's you override only certain stylistic elements
    on any platform, falling back to the system default otherwise.
    b. ``QStyle`` paints various elements using methods like ``drawPrimitive``
       and ``drawControl``.  These can be overridden for very fine control.
6. It has hard to use stylesheets with custom ``QStyles``... but it's possible
   to style sub-controls in ``QAbstractItemView`` (such as ``QTreeView``):
   https://doc.qt.io/qt-5/stylesheet-reference.html#list-of-sub-controls

"""
from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QFont, QImage, QPixmap
from qtpy.QtWidgets import QStyledItemDelegate

from ...layers import Layer
from ..qt_resources import QColoredSVGIcon
from .qt_tree_model import QtNodeTreeModel
from .qt_tree_view import QtNodeTreeView

if TYPE_CHECKING:
    from qtpy.QtCore import QModelIndex
    from qtpy.QtGui import QPainter
    from qtpy.QtWidgets import QStyleOptionViewItem, QWidget

    from ...layers.layergroup import LayerGroup


class QtLayerTreeModel(QtNodeTreeModel[Layer]):
    LayerRole = Qt.UserRole
    ThumbnailRole = Qt.UserRole + 1

    def __init__(self, root: LayerGroup, parent: QWidget = None):
        super().__init__(root, parent)
        self.data
        self.setRoot(root)

    def data(self, index: QModelIndex, role: Qt.ItemDataRole):
        """Return data stored under ``role`` for the item at ``index``."""
        layer = self.getItem(index)
        if role == Qt.DisplayRole:  # used for item text
            return layer.name
        if role == Qt.TextAlignmentRole:  # alignment of the text
            return Qt.AlignCenter
        if role == Qt.EditRole:  # used to populate line edit when editing
            return layer.name
        if role == Qt.ToolTipRole:  # for tooltip
            return layer.name
        if role == Qt.CheckStateRole:  # the "checked" state of this item
            layer_visible = layer._visible
            parents_visible = all(p._visible for p in layer.iter_parents())
            if layer_visible:
                if parents_visible:
                    return Qt.Checked
                else:
                    return Qt.PartiallyChecked
            else:
                return Qt.Unchecked
        if role == Qt.SizeHintRole:  # determines size of item
            h = 38
            return QSize(228, h)
        if role == self.LayerRole:  # custom role: return the layer
            return self.getItem(index)
        if role == self.ThumbnailRole:  # return the thumbnail
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
            'thumbnail': self.ThumbnailRole,
            'visible': Qt.CheckStateRole,
            'name': Qt.DisplayRole,
        }.get(event.type, None)
        roles = [role] if role is not None else []
        top = self.nestedIndex(event.index)
        bot = self.index(top.row() + 1)
        self.dataChanged.emit(top, bot, roles)


class QtLayerTreeView(QtNodeTreeView):
    _root: LayerGroup
    model_class = QtLayerTreeModel

    def __init__(self, root: LayerGroup, parent: QWidget = None):
        super().__init__(root, parent)
        self.setItemDelegate(_LayerDelegate())
        self.setIndentation(18)

        # couldn't set in qss for some reason
        fnt = QFont()
        fnt.setPixelSize(12)
        self.setFont(fnt)

        # self.setStyle(QCommonStyle())

    def viewOptions(self) -> QStyleOptionViewItem:
        options = super().viewOptions()
        options.decorationPosition = options.Right
        return options


class _LayerDelegate(QStyledItemDelegate):
    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ):
        # update the icon based on layer type
        self._get_option_icon(option, index)
        # paint the standard itemView (includes layer name, icon, and visible)
        super().paint(painter, option, index)
        # paint the thumbnail
        self._paint_thumbnail(painter, option, index)

    def _get_option_icon(self, option, index):
        layer = index.data(QtLayerTreeModel.LayerRole)
        if layer.is_group():
            expanded = option.widget.isExpanded(index)
            icon_name = 'folder-open' if expanded else 'folder'
        else:
            icon_name = f'new_{layer._type_string}'
        icon = QColoredSVGIcon.from_resources(icon_name)
        # guessing theme rather than passing it through.
        bg = option.palette.color(option.palette.Background).red()
        option.icon = icon.colored(theme='dark' if bg < 128 else 'light')
        option.decorationSize = QSize(18, 18)
        option.features |= option.HasDecoration

    def _paint_thumbnail(self, painter, option, index):
        # paint the thumbnail
        # MAGICNUMBER: numbers from the margin applied in the stylesheet to
        # QtLayerTreeView::item
        thumb_rect = option.rect.translated(-2, 2)
        h = index.data(Qt.SizeHintRole).height() - 4
        thumb_rect.setWidth(h)
        thumb_rect.setHeight(h)
        image = index.data(QtLayerTreeModel.ThumbnailRole)
        painter.drawPixmap(thumb_rect, QPixmap.fromImage(image))

    def createEditor(
        self,
        parent: QWidget,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> QWidget:
        # necessary for geometry, otherwise editor takes up full space.
        self._get_option_icon(option, index)
        editor = super().createEditor(parent, option, index)
        editor.setAlignment(Qt.Alignment(index.data(Qt.TextAlignmentRole)))
        editor.setObjectName("editor")

        return editor
