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

from functools import partial
from typing import TYPE_CHECKING

from qtpy import QtCore
from qtpy.QtCore import QPoint, QSize, Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QMenu, QStyledItemDelegate

from ..qt_resources import QColoredSVGIcon
from ._base_item_model import ItemRole
from .qt_layer_model import ThumbnailRole

if TYPE_CHECKING:
    from qtpy.QtCore import QModelIndex
    from qtpy.QtGui import QPainter
    from qtpy.QtWidgets import QStyleOptionViewItem, QWidget


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
        index: QModelIndex,
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
        if hasattr(layer, 'is_group') and layer.is_group():  # for layer trees
            expanded = option.widget.isExpanded(index)
            icon_name = 'folder-open' if expanded else 'folder'
        else:
            icon_name = f'new_{layer._type_string}'

        icon = QColoredSVGIcon.from_resources(icon_name)
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
        index: QModelIndex,
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
            self.showContextMenu(
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

    def showContextMenu(
        self, index: QModelIndex, model, pos: QPoint, parent: QWidget
    ):

        menu = QMenu(parent)

        layer_list = model.sourceModel()._root
        layer = index.data(ItemRole)

        for name, action in get_actions(layer, layer_list):
            menu.addAction(name, partial(QTimer.singleShot, 0, action))

        menu.exec_(pos)


from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

from napari.layers.utils import stack_utils


def split_stack(layer, axis, layer_list):
    if layer.rgb:
        images = stack_utils.split_rgb(layer)
    else:
        images = stack_utils.stack_to_images(layer, axis)
    layer_list.remove(layer)
    layer_list.extend(images)
    layer_list.selection = set(images)


def merge_stack(layers, layer_list, position=0, rgb=False):
    if rgb:
        new = stack_utils.merge_rgb(layers)
    else:
        new = stack_utils.images_to_stack(layers)
    for layer in layers:
        layer_list.remove(layer)
    layer_list.insert(position, new)
    layer_list.selection.active = new


def duplicate(layer, layer_list):
    from copy import deepcopy

    new = deepcopy(layer)
    new.name += ' copy'
    layer_list.insert(layer_list.index(layer) + 1, new)


def convert(layer, new_type, layer_list):
    from napari.layers import Layer

    layer_list.remove(layer)
    data = layer.data.astype(int) if layer_list == 'labels' else layer.data
    layer_list.append(Layer.create(data, {}, new_type))


def link_layers(layers):
    from napari.experimental import link_layers

    link_layers(layers)


def get_actions(layer, layer_list):
    from napari.layers import Image, Labels

    selection = layer_list.selection
    layer_in_selection = len(selection) > 1 and layer in selection

    if len(selection) <= 1:
        yield ('Duplicate', partial(duplicate, layer, layer_list))
    else:
        yield ('Link layers', partial(link_layers, selection))
    if isinstance(layer, Image):
        if layer_in_selection:
            try:
                # not sure whether layer.data.shape is always going to work
                if (
                    len({x.data.shape for x in selection}) == 1
                    and len({type(x) for x in selection}) == 1
                ):
                    # use the layer that was clicked on as the "template" for merge
                    lst = list(selection)
                    lst.insert(0, lst.pop(lst.index(layer)))
                    yield (
                        'Stack images',
                        partial(merge_stack, lst, layer_list),
                    )
            except Exception:
                pass
            # not working
            # if len(selection) == 3:
            #     yield (
            #         'Make RGB',
            #         partial(merge_stack, lst, layer_list, rgb=True),
            #     )
        elif layer.rgb:
            yield ('Split RGB', partial(split_stack, layer, -1, layer_list))
        else:
            for ax in (i for i, s in enumerate(layer.data.shape) if s < 8):
                yield (
                    f'Split Image (axis {ax})',
                    partial(split_stack, layer, ax, layer_list),
                )
            yield (
                'Convert to Labels',
                partial(convert, layer, 'labels', layer_list),
            )
    if isinstance(layer, Labels) and not layer_in_selection:
        yield (
            'Convert to Image',
            partial(convert, layer, 'image', layer_list),
        )
