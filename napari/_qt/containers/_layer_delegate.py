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
from qtpy.QtGui import QMouseEvent, QPixmap
from qtpy.QtWidgets import QStyledItemDelegate

from napari._app_model.constants import MenuId
from napari._app_model.context import get_context
from napari._qt._qapp_model import build_qmodel_menu
from napari._qt.containers._base_item_model import ItemRole
from napari._qt.containers.qt_layer_model import ThumbnailRole
from napari._qt.qt_resources import QColoredSVGIcon

if TYPE_CHECKING:
    from qtpy import QtCore
    from qtpy.QtGui import QPainter
    from qtpy.QtWidgets import QStyleOptionViewItem, QWidget

    from napari.components.layerlist import LayerList


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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layer_visibility_states = {}

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

    def get_layer_icon(
        self, option: QStyleOptionViewItem, index: QtCore.QModelIndex
    ):
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
        bg = option.palette.color(option.palette.ColorRole.Window).red()
        option.icon = icon.colored(theme='dark' if bg < 128 else 'light')
        option.decorationSize = QSize(18, 18)
        option.decorationPosition = (
            option.Position.Right
        )  # put icon on the right
        option.features |= option.ViewItemFeature.HasDecoration

    def _paint_thumbnail(self, painter, option, index):
        """paint the layer thumbnail."""
        # paint the thumbnail
        # MAGICNUMBER: numbers from the margin applied in the stylesheet to
        # QtLayerTreeView::item
        thumb_rect = option.rect.translated(-2, 2)
        h = index.data(Qt.ItemDataRole.SizeHintRole).height() - 4
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
        editor.setAlignment(
            Qt.Alignment(index.data(Qt.ItemDataRole.TextAlignmentRole))
        )
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
            event.type() == QMouseEvent.MouseButtonRelease
            and event.button() == Qt.MouseButton.RightButton
        ):
            pnt = (
                event.globalPosition().toPoint()
                if hasattr(event, "globalPosition")
                else event.globalPos()
            )

            self.show_context_menu(index, model, pnt, option.widget)

        # if the user clicks quickly on the visibility checkbox, we *don't*
        # want it to be interpreted as a double-click.  We want the visibilty
        # to simply be toggled.
        if event.type() == QMouseEvent.MouseButtonDblClick:
            self.initStyleOption(option, index)
            style = option.widget.style()
            check_rect = style.subElementRect(
                style.SubElement.SE_ItemViewItemCheckIndicator,
                option,
                option.widget,
            )
            if check_rect.contains(event.pos()):
                cur_state = index.data(Qt.ItemDataRole.CheckStateRole)
                if model.flags(index) & Qt.ItemFlag.ItemIsUserTristate:
                    state = Qt.CheckState((cur_state + 1) % 3)
                else:
                    state = (
                        Qt.CheckState.Unchecked
                        if cur_state
                        else Qt.CheckState.Checked
                    )
                return model.setData(
                    index, state, Qt.ItemDataRole.CheckStateRole
                )
        # catch alt-click on the vis checkbox and hide *other* layer visibility
        # on second alt-click, restore the visibility state of the layers
        if event.type() == QMouseEvent.MouseButtonRelease and (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() == Qt.AltModifier
        ):
            self.initStyleOption(option, index)
            style = option.widget.style()
            check_rect = style.subElementRect(
                style.SubElement.SE_ItemViewItemCheckIndicator,
                option,
                option.widget,
            )
            if check_rect.contains(event.pos()):
                cur_state = index.data(Qt.ItemDataRole.CheckStateRole)
                clicked_layer = index.data(ItemRole)
                layer_list: LayerList = model.sourceModel()._root
                other_layers = [
                    layer for layer in layer_list if layer != clicked_layer
                ]
                if not self._layer_visibility_states:
                    # first option-click, so store visibility, hide others
                    self._layer_visibility_states = {
                        layer: [layer.visible] for layer in layer_list
                    }
                    for layer in other_layers:
                        layer.visible = False
                    # show the clicked layer
                    state = Qt.CheckState.Checked
                    # make a note that this layer was alt-clicked
                    self._layer_visibility_states[clicked_layer].append(True)
                # if user clicked on the last alt-clicked layer, restore and reset
                elif len(self._layer_visibility_states[clicked_layer]) == 2:
                    # account for any added/deleted layers when restoring
                    for layer in layer_list:
                        if layer in self._layer_visibility_states:
                            layer.visible = self._layer_visibility_states[
                                layer
                            ][0]
                    # restore clicked layer to original state
                    state = (
                        Qt.CheckState.Checked
                        if self._layer_visibility_states[clicked_layer][0]
                        else Qt.CheckState.Unchecked
                    )
                    # reset visibility dict
                    self._layer_visibility_states = {}
                else:
                    # option-click on a different layer, hide others, show it
                    for layer in other_layers:
                        layer.visible = False
                        # clear the previous alt-click state
                        if len(self._layer_visibility_states[layer]) == 2:
                            self._layer_visibility_states[layer].pop()
                    state = Qt.CheckState.Checked
                    # make a note that this layer was alt-clicked
                    self._layer_visibility_states[clicked_layer].append(True)
                return model.setData(
                    index, state, Qt.ItemDataRole.CheckStateRole
                )
        # refer all other events to the QStyledItemDelegate
        return super().editorEvent(event, model, option, index)

    def show_context_menu(self, index, model, pos: QPoint, parent):
        """Show the layerlist context menu.
        To add a new item to the menu, update the _LAYER_ACTIONS dict.
        """
        if not hasattr(self, '_context_menu'):
            self._context_menu = build_qmodel_menu(
                MenuId.LAYERLIST_CONTEXT, parent=parent
            )

        layer_list: LayerList = model.sourceModel()._root
        self._context_menu.update_from_context(get_context(layer_list))
        self._context_menu.exec_(pos)
