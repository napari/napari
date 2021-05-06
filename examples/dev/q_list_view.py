"""Example of using low-level `QtListView` with SelectableEventedList

:class:`napari.utils.events.SelectableEventedList` is a mutable sequence that
emits events when modified.  It also has a selection model (tracking which
items are selected).

:class:`napari._qt.containers.QtListView` adapts the `EventedList` to the
QAbstractItemModel/QAbstractItemView interface used by the QtFramework.  This
allows you to create an interactive GUI view onto a python model that stays
up to date, and can modify the python object... while maintining the python
object as the single "source of truth".
"""
import napari
from napari.qt import get_app
from napari._qt.containers import QtListView
from napari.utils.events import SelectableEventedList


get_app()


class MyObject:
    """generic object."""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


# create our evented list
root = SelectableEventedList([MyObject(x) for x in 'abcdefg'])
# create Qt view onto the list
view = QtListView(root)
# show the view
view.show()


# spy on events
root.events.reordered.connect(lambda e: print("reordered to: ", e.value))
root.selection.events.changed.connect(
    lambda e: print(
        f"selection changed.  added: {e.added}, removed: {e.removed}"
    )
)
root.selection.events._current.connect(
    lambda e: print(f"current item changed to: {e.value}")
)


napari.run()
