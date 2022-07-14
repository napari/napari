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
6. It is hard to use stylesheets with custom ``QStyles``... but it's possible
   to style sub-controls in ``QAbstractItemView`` (such as ``QTreeView``):
   https://doc.qt.io/qt-5/stylesheet-reference.html#list-of-sub-controls

"""
from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import QPoint, QSize, Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QStyledItemDelegate

from ..._app.constants import MenuId
from ..._app.context import get_context
from .._qapp_model import build_qmodel_menu
from ..qt_resources import QColoredSVGIcon
from ._base_item_model import ItemRole
from .qt_layer_model import ThumbnailRole

if TYPE_CHECKING:
    from qtpy import QtCore
    from qtpy.QtGui import QPainter
    from qtpy.QtWidgets import QStyleOptionViewItem, QWidget

    from ...components.layerlist import LayerList


class LayerDelegate(QStyledItemDelegate):
    """A QItemDelegate specialized for painting Layer objects.

    In Qt's `Model/View architecture
    <https://doc.qt.io/qt-5/model-view-programming.html>`_. A *delegate* is an
    object that controls the visual rendering (and editing widgets) of an item
    in a view. For more, see:
    https://doc.qt.io/qt-5/model-view-programming.html#delegate-classes

    This class provides the logic required to paint a Layer item in the
    :class:`napari._qt.containers.QtLayerList`.  The `QStyledItemDelegate`
    super-class provides most of the logic (including display/editing of the
    layer name, a visibility checkbox, and an icon for the layer type).  This
    subclass provides additional logic for drawing the layer thumbnail, picking
    the appropriate icon for the layer, and some additional style/UX issues.
    """

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ):
        """Paint the item in the model at `index`."""
        # update the icon based on layer type

        self.get_layer_icon(option, index)
        # paint the standard itemView (includes name, icon, and vis. checkbox)
        super().paint(painter, option, index)
        # paint the thumbnail
        self._paint_thumbnail(painter, option, index)

    def get_layer_icon(self, option, index):
        """Add the appropriate QIcon to the item based on the layer type."""
        layer = index.data(ItemRole)
        if layer is None:
            return
        if hasattr(layer, 'is_group') and layer.is_group():  # for layer trees
            expanded = option.widget.isExpanded(index)
            icon_name = 'folder-open' if expanded else 'folder'
        else:
            icon_name = f'new_{layer._type_string}'

        try:
            icon = QColoredSVGIcon.from_resources(icon_name)
        except ValueError:
            return
        # guessing theme rather than passing it through.
        bg = option.palette.color(option.palette.Background).red()
        option.icon = icon.colored(theme='dark' if bg < 128 else 'light')
        option.decorationSize = QSize(18, 18)
        option.decorationPosition = option.Right  # put icon on the right
        option.features |= option.HasDecoration

    def _paint_thumbnail(self, painter, option, index):
        """paint the layer thumbnail."""
        # paint the thumbnail
        # MAGICNUMBER: numbers from the margin applied in the stylesheet to
        # QtLayerTreeView::item
        thumb_rect = option.rect.translated(-2, 2)
        h = index.data(Qt.SizeHintRole).height() - 4
        thumb_rect.setWidth(h)
        thumb_rect.setHeight(h)
        image = index.data(ThumbnailRole)
        painter.drawPixmap(thumb_rect, QPixmap.fromImage(image))

    def createEditor(
        self,
        parent: QWidget,
        option: QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QWidget:
        """User has double clicked on layer name."""
        # necessary for geometry, otherwise editor takes up full width.
        self.get_layer_icon(option, index)
        editor = super().createEditor(parent, option, index)
        # make sure editor has same alignment as the display name
        editor.setAlignment(Qt.Alignment(index.data(Qt.TextAlignmentRole)))
        return editor

    def editorEvent(
        self,
        event: QtCore.QEvent,
        model: QtCore.QAbstractItemModel,
        option: QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> bool:
        """Called when an event has occured in the editor.

        This can be used to customize how the delegate handles mouse/key events
        """
        if (
            event.type() == event.MouseButtonRelease
            and event.button() == Qt.RightButton
        ):
            self.show_context_menu(
                index, model, event.globalPos(), option.widget
            )

        # if the user clicks quickly on the visibility checkbox, we *don't*
        # want it to be interpreted as a double-click.  We want the visibilty
        # to simply be toggled.
        if event.type() == event.MouseButtonDblClick:
            self.initStyleOption(option, index)
            style = option.widget.style()
            check_rect = style.subElementRect(
                style.SE_ItemViewItemCheckIndicator, option, option.widget
            )
            if check_rect.contains(event.pos()):
                cur_state = index.data(Qt.CheckStateRole)
                if model.flags(index) & Qt.ItemIsUserTristate:
                    state = Qt.CheckState((cur_state + 1) % 3)
                else:
                    state = Qt.Unchecked if cur_state else Qt.Checked
                return model.setData(index, state, Qt.CheckStateRole)
        # refer all other events to the QStyledItemDelegate
        return super().editorEvent(event, model, option, index)

    def show_context_menu(self, index, model, pos: QPoint, parent):
        """Show the layerlist context menu.
        To add a new item to the menu, update the _LAYER_ACTIONS dict.
        """
        if not hasattr(self, '_context_menu'):
            self._context_menu = build_qmodel_menu(MenuId.LAYERLIST_CONTEXT)

        layer_list: LayerList = model.sourceModel()._root
        self._context_menu.update_from_context(get_context(layer_list))
        self._context_menu.exec_(pos)
