from napari.utils.events.containers import SelectableEventedList


def test_remove_discards_from_selection():
    """Removing from the list should also discard from the selection."""
    selectable_list = SelectableEventedList(['a', 'b', 'c'])
    selectable_list.selection = ['a']

    assert 'a' in selectable_list.selection
    selectable_list.remove('a')
    assert 'a' not in selectable_list
    assert 'a' not in selectable_list.selection
