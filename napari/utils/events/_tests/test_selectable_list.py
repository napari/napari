from typing import Iterable, TypeVar

from napari.utils.events.containers import SelectableEventedList

T = TypeVar('T')


def _make_selectable_list_and_select_first(
    items: Iterable[T],
) -> SelectableEventedList[T]:
    selectable_list = SelectableEventedList(items)
    first = selectable_list[0]
    selectable_list.selection = [first]
    assert first in selectable_list.selection
    return selectable_list


def test_remove_discards_from_selection():
    selectable_list = _make_selectable_list_and_select_first(['a', 'b', 'c'])
    selectable_list.remove('a')
    assert 'a' not in selectable_list.selection


def test_pop_discards_from_selection():
    selectable_list = _make_selectable_list_and_select_first(['a', 'b', 'c'])
    selectable_list.pop(0)
    assert 'a' not in selectable_list.selection


def test_del_discards_from_selection():
    selectable_list = _make_selectable_list_and_select_first(['a', 'b', 'c'])
    del selectable_list[0]
    assert 'a' not in selectable_list.selection
