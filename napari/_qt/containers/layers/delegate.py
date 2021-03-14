"""

General rendering flow:

1. The List/Tree view needs to display or edit an index in the model...
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
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QStyledItemDelegate

from ...qt_resources import QColoredSVGIcon
from ._layer_model import QtLayerListModel

if TYPE_CHECKING:
    from qtpy.QtCore import QModelIndex
    from qtpy.QtGui import QPainter
    from qtpy.QtWidgets import QStyleOptionViewItem, QWidget


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
        layer = index.data(QtLayerListModel.LayerRole)
        if hasattr(layer, 'is_group') and layer.is_group():
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
        image = index.data(QtLayerListModel.ThumbnailRole)
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
