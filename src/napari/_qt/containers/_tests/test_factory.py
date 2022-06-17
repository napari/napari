import pytest

from napari._qt.containers import (
    QtListModel,
    QtListView,
    QtNodeTreeModel,
    QtNodeTreeView,
    create_view,
)
from napari.utils.events.containers import SelectableEventedList
from napari.utils.tree import Group, Node


class T(Node):
    def __init__(self, x):
        self.x = x


@pytest.mark.parametrize(
    'cls, exView, exModel',
    [
        (SelectableEventedList, QtListView, QtListModel),
        (Group, QtNodeTreeView, QtNodeTreeModel),
    ],
)
def test_factory(qtbot, cls, exView, exModel):
    a = cls([T(1), T(2)])
    view = create_view(a)
    qtbot.addWidget(view)
    assert isinstance(view, exView)
    assert isinstance(view.model(), exModel)
