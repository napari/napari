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

from enum import Flag, auto
from functools import partial
from typing import TYPE_CHECKING

from PyQt5.QtCore import QTimer
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

from napari.experimental import link_layers, unlink_layers
from napari.layers.utils._link_layers import get_linked_layers, layer_is_linked


class LayerCtx(Flag):
    MULTIPLE_LAYERS_SELECTED = auto()
    ALL_LAYERS_ARE_LINKED = auto()
    LINKED_LAYERS_UNSELECTED = auto()
    IS_RGB = auto()
    ALL_IMAGES = auto()
    ALL_LABELS = auto()
    CAN_SPLIT_IMAGE = auto()
    CAN_STACK_IMAGES = auto()
    CAN_STACK_RGB = auto()


def _select_linked_layers(layer, layer_list):
    layer_list.selection.update(get_linked_layers(*layer_list.selection))


def _link_selection(layer, layer_list):
    link_layers(layer_list.selection)


def _unlink_selection(layer, layer_list):
    unlink_layers(layer_list.selection)


def _duplicate_layer(layer, layer_list):
    from copy import deepcopy

    for lay in list(layer_list.selection):
        new = deepcopy(lay)
        new.name += ' copy'
        layer_list.insert(layer_list.index(lay) + 1, new)


def split_stack(layer, layer_list, axis=0):
    from ...layers.utils import stack_utils

    if layer.rgb:
        images = stack_utils.split_rgb(layer)
    else:
        images = stack_utils.stack_to_images(layer, axis)
    layer_list.remove(layer)
    layer_list.extend(images)
    layer_list.selection = set(images)


def convert(layer, layer_list, type_):
    from napari.layers import Layer

    for lay in list(layer_list.selection):
        idx = layer_list.index(lay)
        data = lay.data.astype(int) if layer_list == 'labels' else lay.data
        layer_list.pop(idx)
        layer_list.insert(idx, Layer.create(data, {'name': lay.name}, type_))


def merge_stack(layer, layer_list, rgb=False):
    from ...layers.utils import stack_utils

    selection = list(layer_list.selection)
    for layer in selection:
        layer_list.remove(layer)
    if rgb:
        new = stack_utils.merge_rgb(selection)
    else:
        new = stack_utils.images_to_stack(selection)
    layer_list.append(new)


class LayerContextMenu(QMenu):
    def __init__(self):
        super().__init__()
        self._duplicate_layer = self.addAction('Duplicate Layer')
        self._duplicate_layer.setData(_duplicate_layer)
        self._convert_to_labels = self.addAction('Convert to Labels')
        self._convert_to_labels.setData(partial(convert, type_='labels'))
        self._convert_to_image = self.addAction('Convert to Image')
        self._convert_to_image.setData(partial(convert, type_='image'))

        self.addSeparator()
        self._split_stack = self.addAction('Split Stack')
        self._split_stack.setData(split_stack)
        self._split_rgb = self.addAction('Split RGB')
        self._split_rgb.setData(split_stack)
        self._merge_stack = self.addAction('Merge to Stack')
        self._merge_stack.setData(merge_stack)
        self._merge_rgb = self.addAction('Merge to RGB')
        self._merge_rgb.setData(partial(merge_stack, rgb=True))

        self.addSeparator()
        self._link_layers = self.addAction('Link Layers')
        # putting callback as data() feels weird, but it still needs to be
        # assigned a specific layer/layer_list argument
        self._link_layers.setData(_link_selection)
        self._unlink_layers = self.addAction('Unlink Layers')
        self._unlink_layers.setData(_unlink_selection)
        self._select_linked_layers = self.addAction('Select Linked Layers')
        self._select_linked_layers.setData(_select_linked_layers)

    def _update(self, c: LayerCtx) -> None:
        self._link_layers.setVisible(True)
        self._unlink_layers.setVisible(False)
        self._split_stack.setVisible(True)
        self._split_rgb.setVisible(False)

        self._convert_to_labels.setEnabled(bool(c & c.ALL_IMAGES))
        self._convert_to_image.setEnabled(bool(c & c.ALL_LABELS))

        if c & c.IS_RGB:
            self._split_rgb.setVisible(True)
            self._split_stack.setVisible(False)
        else:
            self._split_stack.setEnabled(bool(c & c.CAN_SPLIT_IMAGE))

        # self._merge_rgb.setEnabled(bool(c & c.ALL_IMAGES))
        self._merge_rgb.setEnabled(False)  # TODO
        self._merge_stack.setEnabled(bool(c & c.CAN_STACK_IMAGES))

        if c & c.ALL_LAYERS_ARE_LINKED:  # type: ignore
            self._unlink_layers.setVisible(True)
            self._unlink_layers.setEnabled(True)
            self._link_layers.setVisible(False)
        else:
            self._link_layers.setEnabled(bool(c & c.MULTIPLE_LAYERS_SELECTED))

        self._select_linked_layers.setEnabled(
            bool(c & c.LINKED_LAYERS_UNSELECTED)
        )


def get_flags(layer, layer_list):
    from ...layers import Image, Labels
    from ...utils.events.containers import Selection

    selection: Selection = layer_list.selection
    active = selection.active

    flags = LayerCtx(0)
    if len(selection) > 1:
        flags |= LayerCtx.MULTIPLE_LAYERS_SELECTED
    elif isinstance(active, Image):
        if active.rgb:
            flags |= LayerCtx.IS_RGB
            # Determine initial shape
        shp = active.data[0].shape if active.multiscale else active.data.shape
        if any(x < 16 for x in shp):
            flags |= LayerCtx.CAN_SPLIT_IMAGE

    are_linked = [layer_is_linked(x) for x in selection]
    if all(are_linked):
        flags |= LayerCtx.ALL_LAYERS_ARE_LINKED
    if any(are_linked):
        all_links = get_linked_layers(*selection)
        if len(all_links - selection):
            flags |= LayerCtx.LINKED_LAYERS_UNSELECTED
    if all(isinstance(x, Image) for x in selection):
        flags |= LayerCtx.ALL_IMAGES
        if len({x.data.shape for x in selection}) == 1 and len(selection) > 1:
            flags |= LayerCtx.CAN_STACK_IMAGES
    if all(isinstance(x, Labels) for x in selection):
        flags |= LayerCtx.ALL_LABELS
    return flags


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

    def __init__(self):
        super().__init__()
        self._context_menu = LayerContextMenu()

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

    def show_context_menu(
        self, index: QModelIndex, model, pos: QPoint, parent: QWidget
    ):
        layer_list = model.sourceModel()._root
        layer = index.data(ItemRole)
        flags = get_flags(layer, layer_list)
        self._context_menu._update(flags)
        action = self._context_menu.exec_(pos)
        if action is not None:
            QTimer.singleShot(0, partial(action.data(), layer, layer_list))
