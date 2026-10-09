from __future__ import annotations

from typing import TYPE_CHECKING, overload

from napari._qt.containers.qt_axis_model import AxisList, QtAxisListModel
from napari.components.layerlist import LayerList
from napari.utils.events import SelectableEventedList
from napari.utils.tree import Group

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget

    from napari._qt.containers import (
        QtLayerList,
        QtLayerListModel,
        QtListModel,
        QtListView,
        QtNodeTreeModel,
        QtNodeTreeView,
    )


@overload
def create_view(
    obj: LayerList, parent: QWidget | None = None
) -> QtLayerList: ...


@overload
def create_view(
    obj: Group, parent: QWidget | None = None
) -> QtNodeTreeView: ...


@overload
def create_view(
    obj: SelectableEventedList, parent: QWidget | None = None
) -> QtListView: ...


def create_view(
    obj: SelectableEventedList | Group, parent: QWidget | None = None
) -> QtLayerList | QtNodeTreeView | QtListView:
    """Create a `QtListView`, or `QtNodeTreeView` for `obj`.

    Parameters
    ----------
    obj : SelectableEventedList or Group
        The python object for which to creat a QtView.
    parent : QWidget, optional
        Optional parent widget, by default None

    Returns
    -------
    Union[QtListView, QtNodeTreeView]
        A view instance appropriate for `obj`.
    """
    from napari._qt.containers import QtLayerList, QtListView, QtNodeTreeView

    if isinstance(obj, LayerList):
        return QtLayerList(obj, parent=parent)
    if isinstance(obj, Group):
        return QtNodeTreeView(obj, parent=parent)
    if isinstance(obj, SelectableEventedList):
        return QtListView(obj, parent=parent)
    raise TypeError(f'Cannot create Qt view for obj: {obj}')


@overload
def create_model(
    obj: LayerList, parent: QWidget | None = None
) -> QtLayerListModel: ...


@overload
def create_model(
    obj: Group, parent: QWidget | None = None
) -> QtNodeTreeModel: ...


@overload
def create_model(
    obj: AxisList, parent: QWidget | None = None
) -> QtAxisListModel: ...


@overload
def create_model(
    obj: SelectableEventedList, parent: QWidget | None = None
) -> QtListModel: ...


def create_model(
    obj: SelectableEventedList | Group, parent: QWidget | None = None
) -> QtLayerListModel | QtListModel | QtNodeTreeModel:
    """Create a `QtListModel`, or `QtNodeTreeModel` for `obj`.

    Parameters
    ----------
    obj : SelectableEventedList or Group
        The python object for which to creat a QtView.
    parent : QWidget, optional
        Optional parent widget, by default None

    Returns
    -------
    Union[QtListModel, QtNodeTreeModel]
        A model instance appropriate for `obj`.
    """
    from napari._qt.containers import (
        QtLayerListModel,
        QtListModel,
        QtNodeTreeModel,
    )

    if isinstance(obj, LayerList):
        return QtLayerListModel(obj, parent=parent)
    if isinstance(obj, Group):
        return QtNodeTreeModel(obj, parent=parent)
    if isinstance(obj, AxisList):
        return QtAxisListModel(obj, parent=parent)
    if isinstance(obj, SelectableEventedList):
        return QtListModel(obj, parent=parent)
    raise TypeError(f'Cannot create Qt model for obj: {obj}')
