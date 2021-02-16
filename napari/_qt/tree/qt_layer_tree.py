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

from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import (
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QSize,
    Qt,
)
from qtpy.QtGui import QColor, QImage, QPainter, QPixmap
from qtpy.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem, QWidget

from ...utils.theme import get_theme
from ..qt_resources._svg import colored_svg_icon
from .qt_tree_model import QtNodeTreeModel
from .qt_tree_view import QtNodeTreeView

if TYPE_CHECKING:
    from ...layers import Layer
    from ...layers.layergroup import LayerGroup
    from ...utils.events.containers._nested_list import MaybeNestedIndex


class QtLayerTreeModel(QtNodeTreeModel):
    LayerRole = Qt.UserRole
    ThumbnailRole = Qt.UserRole + 1
    LayerTypeRole = Qt.UserRole + 2

    def __init__(self, root: LayerGroup, parent: QWidget = None):
        super().__init__(root, parent)
        self.data
        self.setRoot(root)

    def getItem(self, index: QModelIndex) -> Layer:
        # TODO: this ignore should be fixed by making QtNodeTreeModel Generic.
        return super().getItem(index)  # type: ignore

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
            h = 32 if layer.is_group() else 38
            return QSize(228, h)
        if role == QtLayerTreeModel.LayerRole:  # custom role: return the layer
            return self.getItem(index)
        if role == QtLayerTreeModel.LayerTypeRole:  # custom: layer type string
            return self.getItem(index)._type_string
        if role == QtLayerTreeModel.ThumbnailRole:  # return the thumbnail
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
        else:
            return super().setData(index, value, role=role)

        self.dataChanged.emit(index, index, [role])
        return True


class QtLayerTreeView(QtNodeTreeView):
    def __init__(self, root: LayerGroup = None, parent: QWidget = None):
        super().__init__(root, parent)
        self.setItemDelegate(LayerDelegate())
        self.setAnimated(True)
        self.setAutoExpandDelay(300)

    def viewOptions(self) -> QStyleOptionViewItem:
        options = super().viewOptions()
        options.decorationPosition = QStyleOptionViewItem.Right
        return options

    def setRoot(self, root: LayerGroup):
        self.setModel(QtLayerTreeModel(root, self))
        self.model().rowsRemoved.connect(self._redecorate_root)
        self.model().rowsInserted.connect(self._redecorate_root)
        self._redecorate_root()
        root.events.selection.connect(lambda e: self._select(e.index, e.value))
        # initialize selection model
        for child in root.traverse():
            selected = getattr(child, 'selected', False)
            self._select(child.index_from_root(), selected)

    def _redecorate_root(self, parent=None, *_):
        """Add a branch/arrow column only if there are Groups in the root."""
        if not parent or not parent.isValid():
            self.setRootIsDecorated(self.model().hasGroups())

    def selectionChanged(
        self, selected: QItemSelection, deselected: QItemSelection
    ):
        model = self.model()
        for q_index in selected.indexes():
            model.getItem(q_index).selected = True
        for q_index in deselected.indexes():
            model.getItem(q_index).selected = False
        return super().selectionChanged(selected, deselected)

    def _select(self, nested_index: MaybeNestedIndex, selected=True):
        idx = self.model().nestedIndex(nested_index)
        if nested_index == () or not idx.isValid():
            return
        s = getattr(QItemSelectionModel, 'Select' if selected else 'Deselect')
        # TODO: figure out bug on pop(0) in examples/layer_tree
        self.selectionModel().select(idx, s)


class LayerDelegate(QStyledItemDelegate):
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
        ltype = index.data(QtLayerTreeModel.LayerTypeRole)
        icons = Path(__file__).parent.parent.parent / 'resources' / 'icons'
        if ltype == 'layergroup':
            path = icons / 'folder.svg'
        else:
            path = icons / f'new_{ltype}.svg'

        # guessing theme rather than passing it through.
        bg = option.palette.color(option.palette.Background).red()
        theme = get_theme('dark' if bg < 128 else 'light')
        icon_color = theme['icon'].strip("rgb()").split(',')
        icon_color = QColor(*map(int, icon_color)).name()

        option.icon = colored_svg_icon(path, icon_color)
        option.features |= QStyleOptionViewItem.HasDecoration

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
        editor = super().createEditor(parent, option, index)
        editor.setAlignment(Qt.Alignment(index.data(Qt.TextAlignmentRole)))
        return editor
