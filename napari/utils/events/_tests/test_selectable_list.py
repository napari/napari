from collections.abc import Iterable
from typing import TypeVar

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


def test_select_next():
    selectable_list = _make_selectable_list_and_select_first(['a', 'b', 'c'])
    assert 'a' in selectable_list.selection
    selectable_list.select_next()
    assert 'a' not in selectable_list.selection
    assert 'b' in selectable_list.selection


def test_select_previous():
    selectable_list = _make_selectable_list_and_select_first(['a', 'b', 'c'])
    selectable_list.selection.active = 'c'
    assert 'a' not in selectable_list.selection
    assert 'c' in selectable_list.selection
    selectable_list.select_previous()
    assert 'c' not in selectable_list.selection
    assert 'b' in selectable_list.selection


def test_shift_select_next_previous():
    selectable_list = _make_selectable_list_and_select_first(['a', 'b', 'c'])
    assert 'a' in selectable_list.selection
    selectable_list.select_next(shift=True)
    assert 'a' in selectable_list.selection
    assert 'b' in selectable_list.selection
    selectable_list.select_previous(shift=True)
    assert 'a' in selectable_list.selection
    assert 'b' not in selectable_list.selection


def test_shift_select_previous_next():
    selectable_list = _make_selectable_list_and_select_first(['a', 'b', 'c'])
    selectable_list.selection.active = 'c'
    assert 'a' not in selectable_list.selection
    assert 'c' in selectable_list.selection
    selectable_list.select_previous(shift=True)
    assert 'b' in selectable_list.selection
    assert 'c' in selectable_list.selection
    selectable_list.select_next(shift=True)
    assert 'b' not in selectable_list.selection
    assert 'c' in selectable_list.selection
